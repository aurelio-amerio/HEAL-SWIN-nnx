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


class HealConvBlock(nnx.Module):
    """shift -> window -> depthwise k x k conv (Cartesian grid) -> unwindow ->
    unshift, then the shared per-pixel Mlp. Post-norm residual wiring identical
    to HealSwinBlock; the conv replaces window attention.

    In shifted blocks, wrapped-in foreign pixels are zeroed before the conv
    (contribute nothing) and their update is zeroed after (residual
    pass-through), per the majority-region validity mask."""

    def __init__(self, params, dim, input_resolution, shifted, drop_path, *, rngs):
        self.input_resolution = input_resolution
        self.window_size = min(params.window_size, input_resolution)
        self.num_windows = input_resolution // self.window_size
        self.grid_size = math.isqrt(self.window_size)
        assert self.grid_size ** 2 == self.window_size, \
            "window clamped to a non-square size %d" % self.window_size
        shift_size = params.shift_size if (shifted
                                           and input_resolution > params.window_size) else 0

        grid = get_nest_win_idcs(self.window_size).reshape(-1)
        self.grid_perm = Buffer(jnp.asarray(grid))
        self.inv_perm = Buffer(jnp.asarray(np.argsort(grid)))

        self.dwconv = nnx.Conv(dim, dim, kernel_size=(self.grid_size, self.grid_size),
                               feature_group_count=dim, padding="SAME",
                               use_bias=params.conv_bias, kernel_init=TRUNC_NORMAL,
                               rngs=rngs)
        self.norm1 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * params.mlp_ratio), drop=params.drop_rate, rngs=rngs)

        nside = math.isqrt(input_resolution // len(params.base_pixels))
        assert nside * nside * len(params.base_pixels) == input_resolution, \
            "nside has to be an integer in every layer"

        if shift_size > 0:
            if params.shift_strategy == "nest_roll":
                self.shifter = shifting.NestRollShift(
                    shift_size=shift_size, input_resolution=input_resolution,
                    window_size=self.window_size)
                raw_mask = shifting.nest_roll_raw_mask(
                    input_resolution, self.window_size, shift_size)
            elif params.shift_strategy == "nest_grid_shift":
                self.shifter = shifting.NestGridShift(
                    nside=nside, base_pixels=params.base_pixels,
                    window_size=self.window_size)
                raw_mask = shifting.nest_grid_mask(
                    nside, list(params.base_pixels), self.window_size)
            elif params.shift_strategy == "nest_grid_shift_exact":
                self.shifter = shifting.NestGridShiftExact(
                    nside=nside, base_pixels=params.base_pixels,
                    window_size=self.window_size)
                _, raw_mask = hp_topology.exact_shift_idcs_and_mask(
                    list(params.base_pixels), nside, self.window_size)
            else:  # "ring_shift" — Params validated the enum
                self.shifter = shifting.RingShift(
                    nside=nside, base_pixels=params.base_pixels,
                    window_size=self.window_size, shift_size=shift_size)
                _, raw_mask = shifting.ring_shift_idcs_and_mask(
                    nside, list(params.base_pixels), self.window_size, shift_size)
            self.validity = Buffer(jnp.asarray(
                shifting.validity_from_mask(raw_mask, self.window_size))[:, :, None])
        else:
            self.shifter = shifting.NoShift()
            self.validity = None

    def _apply_validity(self, w):
        B_, ws, C = w.shape
        v = self.validity[...]                       # (nW, ws, 1)
        return (w.reshape(-1, self.num_windows, ws, C) * v[None]).reshape(B_, ws, C)

    def _mix(self, x):
        """The spatial mixer: shift -> window -> mask -> conv on the Cartesian
        grid -> mask -> unwindow -> unshift. Exact identity for a delta kernel."""
        shifted_x = self.shifter.shift(x)
        w = window_partition(shifted_x, self.window_size)      # (B*nW, ws, C) nested
        if self.validity is not None:
            w = self._apply_validity(w)                        # zero foreign IN
        B_, ws, C = w.shape
        g = w[:, self.grid_perm[...], :].reshape(B_, self.grid_size, self.grid_size, C)
        g = self.dwconv(g)
        w = g.reshape(B_, ws, C)[:, self.inv_perm[...], :]
        if self.validity is not None:
            w = self._apply_validity(w)                        # zero foreign OUT
        shifted_x = window_reverse(w, self.window_size, self.input_resolution)
        return self.shifter.shift_back(shifted_x)

    def __call__(self, x):
        x = x + self.drop_path(self.norm1(self._mix(x)))
        return x + self.drop_path(self.norm2(self.mlp(x)))
