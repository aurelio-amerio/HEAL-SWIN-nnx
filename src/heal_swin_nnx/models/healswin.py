"""HealSwin: HEALPix-native Swin V2 U-Net (diverged from the HEAL-SWIN reference)."""
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

POS_EMBEDS = ("none", "rel_bias", "rope_axial", "rope_mixed")
SHIFT_STRATEGIES = ("nest_roll", "nest_grid_shift", "nest_grid_shift_exact", "ring_shift")


@dataclass
class HealSwinParams:
    """Pure-data description of a HealSwin model (architecture + geometry).

    Serializable: ``json.dumps(dataclasses.asdict(params))`` works, so a run's
    exact configuration can be logged and compared."""

    # data / geometry
    nside: int                       # HEALPix resolution of the input map
    in_channels: int
    out_channels: int
    base_pixels: Optional[Union[Tuple[int, ...], Sequence[int]]] = None  # None -> full sphere

    # architecture
    patch_size: int = 4
    window_size: int = 4
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 2, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    pos_embed: Literal["none", "rel_bias", "rope_axial", "rope_mixed"] = "rope_mixed"
    rope_theta: float = 10.0
    patch_embed_norm: bool = False
    shift_strategy: Literal["nest_roll", "nest_grid_shift", "nest_grid_shift_exact",
                            "ring_shift"] = "nest_grid_shift_exact"

    # regularization / training
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    use_checkpoint: bool = False

    def __post_init__(self):
        if self.base_pixels is None:
            self.base_pixels = tuple(range(12))
        self.base_pixels = tuple(self.base_pixels)
        self.depths = tuple(self.depths)
        self.num_heads = tuple(self.num_heads)

        if any(not 0 <= b <= 11 for b in self.base_pixels):
            raise ValueError("base_pixels must be in [0, 11], got %r" % (self.base_pixels,))
        if any(a >= b for a, b in zip(self.base_pixels, self.base_pixels[1:])):
            raise ValueError(
                "base_pixels must be strictly increasing (canonical NEST subset order), "
                "got %r" % (self.base_pixels,))
        if self.pos_embed not in POS_EMBEDS:
            raise ValueError("pos_embed must be one of %r, got %r"
                             % (POS_EMBEDS, self.pos_embed))
        if self.shift_strategy not in SHIFT_STRATEGIES:
            raise ValueError("shift_strategy must be one of %r, got %r"
                             % (SHIFT_STRATEGIES, self.shift_strategy))
        if len(self.depths) != len(self.num_heads):
            raise ValueError("depths (%d) and num_heads (%d) must have equal length"
                             % (len(self.depths), len(self.num_heads)))
        if self.patch_size <= 0 or self.patch_size % 4 != 0:
            raise ValueError("patch_size must be a positive multiple of 4 "
                             "(valid nside in deeper layers), got %d" % self.patch_size)
        if self.window_size <= 0 or self.window_size & (self.window_size - 1):
            raise ValueError("window_size must be a power of two, got %d" % self.window_size)
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
        for i, heads in enumerate(self.num_heads):
            dim = self.embed_dim * 2 ** i
            if dim % heads:
                raise ValueError("stage %d: dim %d not divisible by num_heads %d"
                                 % (i, dim, heads))
            if self.pos_embed in ("rope_axial", "rope_mixed") and (dim // heads) % 4:
                raise ValueError(
                    "stage %d: head_dim %d must be divisible by 4 for RoPE "
                    "(2D frequency split)" % (i, dim // heads))

    @property
    def npix(self):
        return len(self.base_pixels) * self.nside ** 2

    @property
    def shift_size(self):
        return self.window_size // 2
