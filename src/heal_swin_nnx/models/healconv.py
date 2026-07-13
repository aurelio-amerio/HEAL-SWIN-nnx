"""HealConv: HEALPix U-Net with depthwise-conv window mixing (no attention).

Reuses HEAL-SWIN's traversal machinery (window partition + shift strategies);
the in-window mixer is a depthwise k x k convolution over the window's
Cartesian grid, followed by the shared per-pixel Mlp (ConvNeXt/MetaFormer
factorization). Spec: docs/superpowers/specs/2026-07-13-healconv-design.md.
"""
import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
from flax import nnx

from heal_swin_nnx.hp import shifting
from heal_swin_nnx.hp import topology as hp_topology
from heal_swin_nnx.hp.shifting import SHIFT_STRATEGIES
from heal_swin_nnx.hp.windowing import get_nest_win_idcs, window_partition, window_reverse
from heal_swin_nnx.layers import (
    LN_EPS, TRUNC_NORMAL, DropPath, FinalPatchExpand, Identity, Mlp, PatchEmbed,
    PatchExpand, PatchMerging)
from heal_swin_nnx.variables import Buffer


@dataclass
class HealConvParams:
    """Pure-data description of a HealConv model (architecture + geometry).

    Serializable: ``json.dumps(dataclasses.asdict(params))`` works, so a run's
    exact configuration can be logged and compared."""

    # data / geometry
    nside: int                       # HEALPix resolution of the input map
    in_channels: int
    out_channels: int
    base_pixels: Optional[Union[Tuple[int, ...], Sequence[int]]] = None  # None -> full sphere

    # architecture
    patch_size: int = 4
    kernel_size: int = 4             # k x k depthwise kernel; window = k^2 pixels
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 2, 2)
    mlp_ratio: float = 4.0
    conv_bias: bool = True
    patch_embed_norm: bool = False
    shift_strategy: Literal["nest_roll", "nest_grid_shift", "nest_grid_shift_exact",
                            "ring_shift"] = "nest_grid_shift_exact"

    # regularization / training
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    use_checkpoint: bool = False

    def __post_init__(self):
        if self.base_pixels is None:
            self.base_pixels = tuple(range(12))
        self.base_pixels = tuple(self.base_pixels)
        self.depths = tuple(self.depths)

        if any(not 0 <= b <= 11 for b in self.base_pixels):
            raise ValueError("base_pixels must be in [0, 11], got %r" % (self.base_pixels,))
        if any(a >= b for a, b in zip(self.base_pixels, self.base_pixels[1:])):
            raise ValueError(
                "base_pixels must be strictly increasing (canonical NEST subset order), "
                "got %r" % (self.base_pixels,))
        if self.shift_strategy not in SHIFT_STRATEGIES:
            raise ValueError("shift_strategy must be one of %r, got %r"
                             % (SHIFT_STRATEGIES, self.shift_strategy))
        if self.kernel_size < 2 or self.kernel_size & (self.kernel_size - 1):
            raise ValueError(
                "kernel_size must be a power of two >= 2: the k x k kernel spans a "
                "nested quadtree window of k^2 pixels, so k must be in {2, 4, 8, ...} "
                "(a 3x3 kernel is not expressible). Got %d" % self.kernel_size)
        if self.patch_size <= 0 or self.patch_size % 4 != 0:
            raise ValueError("patch_size must be a positive multiple of 4 "
                             "(valid nside in deeper layers), got %d" % self.patch_size)
        if self.nside <= 0 or self.nside & (self.nside - 1):
            raise ValueError("nside must be a power of two, got %d" % self.nside)
        if self.nside ** 2 % self.patch_size:
            raise ValueError("nside^2 (%d) must be divisible by patch_size (%d)"
                             % (self.nside ** 2, self.patch_size))
        n_stages = len(self.depths)
        if (self.nside ** 2 // self.patch_size) % 4 ** (n_stages - 1):
            raise ValueError(
                "nside^2/patch_size (%d) must be divisible by 4^(n_stages-1) (%d): "
                "every encoder stage needs an integer per-face nside"
                % (self.nside ** 2 // self.patch_size, 4 ** (n_stages - 1)))
        # The k x k kernel spans a whole window, so every stage must tile into
        # whole windows (a clamped window would not be a power of four).
        bottleneck = self.npix // self.patch_size // 4 ** (n_stages - 1)
        if bottleneck % self.window_size:
            raise ValueError(
                "bottleneck resolution (%d pixels = npix/patch_size/4^(n_stages-1)) "
                "must be divisible by window_size = kernel_size^2 (%d): every stage "
                "needs whole windows for the k x k kernel. Use a smaller kernel_size, "
                "fewer stages, or a larger nside." % (bottleneck, self.window_size))
        # nest_grid_shift's hierarchical index math needs the deepest (bottleneck)
        # stage to hold at least one full window per face (same rule as HealSwin).
        if self.shift_strategy == "nest_grid_shift":
            bottleneck_face_pix = (self.nside ** 2 // self.patch_size) // 4 ** (n_stages - 1)
            if bottleneck_face_pix < self.window_size:
                raise ValueError(
                    "shift_strategy='nest_grid_shift' needs the bottleneck (deepest) "
                    "stage to hold a full window: bottleneck nside^2 (%d) must be >= "
                    "window_size = kernel_size^2 (%d). Use fewer stages, a larger "
                    "nside, a smaller kernel_size, or a shift_strategy that supports "
                    "a unit bottleneck ('nest_grid_shift_exact', 'ring_shift', "
                    "'nest_roll')." % (bottleneck_face_pix, self.window_size))

    @property
    def window_size(self):
        return self.kernel_size ** 2

    @property
    def shift_size(self):
        return self.window_size // 2

    @property
    def npix(self):
        return len(self.base_pixels) * self.nside ** 2
