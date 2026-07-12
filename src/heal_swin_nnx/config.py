"""Serializable mirrors of the reference model configs.

Differences from the reference (all agreed in the spec):
- ``norm_layer``/``patch_embed_norm_layer`` are string literals, not classes.
- ``decoder_class`` removed (dead extension hook, only ``UnetDecoder`` existed).
- ``patch_norm`` and ``dev_mode`` removed (unused / debug scaffolding).
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union


@dataclass
class DataSpec:
    dim_in: Union[int, Tuple[int, int]]  # int (=npix) for HP, (H, W) for flat
    f_in: int
    f_out: int
    base_pix: Optional[int] = None           # legacy; derived from base_pixels after init
    base_pixels: Optional[List[int]] = None  # HEALPix faces used; None -> full sphere
    class_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.base_pixels is None:
            self.base_pixels = list(range(12 if self.base_pix is None else self.base_pix))
        self.base_pixels = list(self.base_pixels)
        if any(not 0 <= b <= 11 for b in self.base_pixels):
            raise ValueError("base_pixels must be in [0, 11], got %r" % (self.base_pixels,))
        if any(a >= b for a, b in zip(self.base_pixels, self.base_pixels[1:])):
            raise ValueError(
                "base_pixels must be strictly increasing (canonical NEST subset order), "
                "got %r" % (self.base_pixels,))
        if self.base_pix is not None and self.base_pix != len(self.base_pixels):
            raise ValueError("base_pix=%d inconsistent with base_pixels of length %d"
                             % (self.base_pix, len(self.base_pixels)))
        self.base_pix = len(self.base_pixels)


@dataclass
class SwinHPTransformerConfig:
    patch_size: int = 4
    window_size: int = 4
    shift_size: int = 2
    shift_strategy: Literal["nest_roll", "nest_grid_shift", "ring_shift"] = "nest_roll"
    rel_pos_bias: Optional[Literal["flat"]] = None
    embed_dim: int = 96
    patch_embed_norm_layer: Optional[Literal["layernorm"]] = None
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_heads: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    use_cos_attn: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: Literal["layernorm"] = "layernorm"
    use_v2_norm_placement: bool = False
    ape: bool = False
    use_checkpoint: bool = False


@dataclass
class SwinTransformerConfig:
    patch_size: Union[int, Tuple[int, int]] = (4, 4)
    window_size: Union[int, Tuple[int, int]] = (4, 4)
    shift_size: Union[int, Tuple[int, int]] = -1  # -1 -> window_size // 2
    embed_dim: int = 96
    patch_embed_norm_layer: Optional[Literal["layernorm"]] = None
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_heads: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    use_cos_attn: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: Literal["layernorm"] = "layernorm"
    use_v2_norm_placement: bool = False
    ape: bool = False
    use_checkpoint: bool = False
    final_upsample: Literal["expand_first"] = "expand_first"
    use_masking: bool = True
    use_rel_pos_bias: bool = True

    def __post_init__(self):
        if isinstance(self.patch_size, int):
            self.patch_size = (self.patch_size, self.patch_size)
        self.patch_size = tuple(self.patch_size)
        if isinstance(self.window_size, int):
            self.window_size = (self.window_size, self.window_size)
        self.window_size = tuple(self.window_size)
        if self.shift_size == -1:
            self.shift_size = (self.window_size[0] // 2, self.window_size[1] // 2)
        elif isinstance(self.shift_size, int):
            self.shift_size = (self.shift_size, self.shift_size)
        self.shift_size = tuple(self.shift_size)
