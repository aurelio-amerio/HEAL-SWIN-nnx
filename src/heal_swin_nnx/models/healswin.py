"""HealSwin: HEALPix-native Swin V2 U-Net (diverged from the HEAL-SWIN reference)."""
import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from heal_swin_nnx.hp import shifting
from heal_swin_nnx.hp.shifting import SHIFT_STRATEGIES
from heal_swin_nnx.hp.windowing import (
    nest_relative_position_index, nest_win_coords, window_partition, window_reverse)
from heal_swin_nnx.layers import (
    LN_EPS, TRUNC_NORMAL, DropPath, FinalPatchExpand, Identity, Mlp, PatchEmbed,
    PatchExpand, PatchMerging, apply_rope, canonical_float_dtype, init_rope_freqs,
    l2_normalize, rope_rotation_table)
from heal_swin_nnx.variables import Buffer

POS_EMBEDS = ("none", "rel_bias", "rope_axial", "rope_mixed")


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

    # precision
    param_dtype: str = "float32"     # parameter storage; any DTypeLike, stored as name
    dtype: str = "float32"           # compute/matmul dtype; "float32" is a staging
                                     # default — flipped to "bfloat16" in the final
                                     # task of the compute-dtype plan

    def __post_init__(self):
        if self.base_pixels is None:
            self.base_pixels = tuple(range(12))
        self.base_pixels = tuple(self.base_pixels)
        self.depths = tuple(self.depths)
        self.num_heads = tuple(self.num_heads)
        self.param_dtype = canonical_float_dtype(self.param_dtype, "param_dtype")
        self.dtype = canonical_float_dtype(self.dtype, "dtype")

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
        ps = int(round(self.patch_size ** 0.5)) if self.patch_size > 0 else 0
        if self.patch_size <= 0 or ps * ps != self.patch_size or ps & (ps - 1):
            raise ValueError(
                "patch_size must be a power of four (1 = no regrouping; the patched "
                "grid needs an integer nside), got %d" % self.patch_size)
        s = int(round(self.window_size ** 0.5))
        if self.window_size <= 0 or s * s != self.window_size or s & (s - 1):
            raise ValueError(
                "window_size must be a power of four (square nested window), "
                "got %d" % self.window_size)
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
        # nest_grid_shift's hierarchical index math needs the deepest (bottleneck)
        # stage to hold at least one full window per face — its per-face pixel count
        # (nside_bottleneck^2) must be >= window_size, else base_pix_len rounds to 0
        # and construction divides by zero. The other strategies handle a unit
        # bottleneck, so this constraint is specific to nest_grid_shift.
        if self.shift_strategy == "nest_grid_shift":
            bottleneck_face_pix = (self.nside ** 2 // self.patch_size) // 4 ** (n_stages - 1)
            if bottleneck_face_pix < self.window_size:
                bottleneck_nside = int(round(bottleneck_face_pix ** 0.5))
                raise ValueError(
                    "shift_strategy='nest_grid_shift' needs the bottleneck (deepest) "
                    "stage to hold a full window: bottleneck nside^2 (%d) must be >= "
                    "window_size (%d), i.e. bottleneck nside >= %d. Got bottleneck "
                    "nside=%d from nside=%d, patch_size=%d, %d stages. Use fewer stages, "
                    "a larger nside, a smaller window_size, or a shift_strategy that "
                    "supports a unit bottleneck ('nest_grid_shift_exact', 'ring_shift', "
                    "'nest_roll')."
                    % (bottleneck_face_pix, self.window_size,
                       int(round(self.window_size ** 0.5)), bottleneck_nside,
                       self.nside, self.patch_size, n_stages))
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


class WindowAttention(nnx.Module):
    """Swin V2 window attention: cosine similarity with learned logit scale,
    positional encoding selected by ``params.pos_embed``."""

    def __init__(self, params, dim, num_heads, window_size, *, rngs):
        self.num_heads = num_heads
        self.pos_embed = params.pos_embed
        self.dtype = params.dtype
        head_dim = dim // num_heads
        self.logit_scale = nnx.Param(
            jnp.full((num_heads, 1, 1), jnp.log(10.0), dtype=params.param_dtype))

        if self.pos_embed == "rel_bias":
            s = int(round(window_size ** 0.5))
            assert s * s == window_size, "rel_bias needs a square (power-of-4) window"
            self.relative_position_bias_table = nnx.Param(
                TRUNC_NORMAL(rngs.params(), ((2 * s - 1) ** 2, num_heads),
                             params.param_dtype))
            self.relative_position_index = Buffer(
                jnp.asarray(nest_relative_position_index(window_size)))
        elif self.pos_embed in ("rope_axial", "rope_mixed"):
            coords = jnp.asarray(nest_win_coords(window_size))  # (2, window_size)
            if self.pos_embed == "rope_mixed":
                # rope_freqs stays f32: it feeds the f32 angle computation (see apply_rope)
                self.rope_freqs = nnx.Param(init_rope_freqs(
                    head_dim, num_heads, params.rope_theta, key=rngs.params()))
                self.rope_coords = Buffer(coords)
            else:
                freqs = init_rope_freqs(head_dim, num_heads, params.rope_theta)
                self.rope_table = Buffer(rope_rotation_table(freqs, coords[0], coords[1]))

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=params.qkv_bias,
                              kernel_init=TRUNC_NORMAL, dtype=params.dtype,
                              param_dtype=params.param_dtype, rngs=rngs)
        self.attn_drop = nnx.Dropout(params.attn_drop_rate, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=TRUNC_NORMAL, dtype=params.dtype,
                               param_dtype=params.param_dtype, rngs=rngs)
        self.proj_drop = nnx.Dropout(params.drop_rate, rngs=rngs)

    def __call__(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = l2_normalize(q)
        k = l2_normalize(k)
        if self.pos_embed == "rope_mixed":
            coords = self.rope_coords[...]
            table = rope_rotation_table(self.rope_freqs[...], coords[0], coords[1])
            q, k = apply_rope(q, k, table)
        elif self.pos_embed == "rope_axial":
            q, k = apply_rope(q, k, self.rope_table[...])
        # fp32 logits island: bf16 operands with fp32 accumulation (free on
        # tensor cores), and scale/bias/mask/softmax stay fp32. Cosine logits
        # are bounded (|logit| <= 100), so this is about bf16's 8-bit mantissa
        # resolution near the softmax operating point, not overflow.
        attn = jnp.einsum("bhnd,bhmd->bhnm", q, k,
                          preferred_element_type=jnp.float32)
        logit_scale = jnp.exp(jnp.minimum(
            self.logit_scale[...].astype(jnp.float32), jnp.log(1.0 / 0.01)))
        attn = attn * logit_scale

        if self.pos_embed == "rel_bias":
            bias = self.relative_position_bias_table[...][self.relative_position_index[...]]
            attn = attn + bias.transpose(2, 0, 1)[None].astype(jnp.float32)

        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.reshape(B_ // nW, nW, self.num_heads, N, N)
                    + mask.astype(attn.dtype)[None, :, None])
            attn = attn.reshape(-1, self.num_heads, N, N)
        attn = jax.nn.softmax(attn, axis=-1).astype(self.dtype)  # island exit
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class HealSwinBlock(nnx.Module):
    def __init__(self, params, dim, input_resolution, num_heads, shifted, drop_path, *, rngs):
        self.input_resolution = input_resolution
        self.window_size = min(params.window_size, input_resolution)
        shift_size = params.shift_size if (shifted
                                           and input_resolution > params.window_size) else 0

        self.dtype = params.dtype
        self.norm1 = nnx.LayerNorm(dim, epsilon=LN_EPS, dtype=jnp.float32,
                                   param_dtype=params.param_dtype, rngs=rngs)
        self.attn = WindowAttention(params, dim, num_heads, self.window_size, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(dim, epsilon=LN_EPS, dtype=jnp.float32,
                                   param_dtype=params.param_dtype, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * params.mlp_ratio), drop=params.drop_rate,
                       dtype=params.dtype, param_dtype=params.param_dtype, rngs=rngs)

        nside = math.isqrt(input_resolution // len(params.base_pixels))
        assert nside * nside * len(params.base_pixels) == input_resolution, \
            "nside has to be an integer in every layer"

        if shift_size > 0:
            if params.shift_strategy == "nest_roll":
                self.shifter = shifting.NestRollShift(
                    shift_size=shift_size, input_resolution=input_resolution,
                    window_size=self.window_size)
            elif params.shift_strategy == "nest_grid_shift":
                self.shifter = shifting.NestGridShift(
                    nside=nside, base_pixels=params.base_pixels,
                    window_size=self.window_size)
            elif params.shift_strategy == "nest_grid_shift_exact":
                self.shifter = shifting.NestGridShiftExact(
                    nside=nside, base_pixels=params.base_pixels,
                    window_size=self.window_size)
            else:  # "ring_shift" — Params validated the enum
                self.shifter = shifting.RingShift(
                    nside=nside, base_pixels=params.base_pixels,
                    window_size=self.window_size, shift_size=shift_size)
        else:
            self.shifter = shifting.NoShift()

    def __call__(self, x):
        shortcut = x
        shifted_x = self.shifter.shift(x)
        x_windows = window_partition(shifted_x, self.window_size)
        mask = None if self.shifter.attn_mask is None else self.shifter.attn_mask[...]
        attn_windows = self.attn(x_windows, mask=mask)
        shifted_x = window_reverse(attn_windows, self.window_size, self.input_resolution)
        x = self.shifter.shift_back(shifted_x)

        # fp32 norm islands exit here: cast BEFORE the residual add — adds do
        # not self-heal, one fp32 summand re-promotes the whole downstream stream
        x = shortcut + self.drop_path(self.norm1(x).astype(self.dtype))
        return x + self.drop_path(self.norm2(self.mlp(x)).astype(self.dtype))


def _make_blocks(params, dim, input_resolution, depth, num_heads, drop_path, rngs):
    return [HealSwinBlock(params, dim, input_resolution, num_heads,
                          shifted=(i % 2 == 1), drop_path=drop_path[i], rngs=rngs)
            for i in range(depth)]


class EncoderStage(nnx.Module):
    def __init__(self, params, dim, input_resolution, depth, num_heads, drop_path,
                 downsample, *, rngs):
        self.use_checkpoint = params.use_checkpoint
        self.blocks = nnx.List(_make_blocks(params, dim, input_resolution, depth,
                                            num_heads, drop_path, rngs))
        self.downsample = (PatchMerging(dim=dim, dtype=params.dtype,
                                        param_dtype=params.param_dtype,
                                        rngs=rngs) if downsample else None)

    def __call__(self, x):
        for blk in self.blocks:
            x = nnx.remat(type(blk).__call__)(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class DecoderStage(nnx.Module):
    def __init__(self, params, dim, input_resolution, depth, num_heads, drop_path,
                 upsample, *, rngs):
        self.use_checkpoint = params.use_checkpoint
        self.blocks = nnx.List(_make_blocks(params, dim, input_resolution, depth,
                                            num_heads, drop_path, rngs))
        self.upsample = (PatchExpand(dim=dim, dim_scale=2, dtype=params.dtype,
                                     param_dtype=params.param_dtype, rngs=rngs)
                         if upsample else None)

    def __call__(self, x):
        for blk in self.blocks:
            x = nnx.remat(type(blk).__call__)(blk, x) if self.use_checkpoint else blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


def _drop_path_schedule(params):
    return [float(v) for v in np.linspace(0, params.drop_path_rate, sum(params.depths))]


class HealSwinEncoder(nnx.Module):
    """Compression-only backbone: patch embed + encoder stages + final norm.
    Standalone-usable (tokenizer / embedder); allocates no decoder parameters."""

    def __init__(self, params, *, rngs):
        self.params = params
        self.num_layers = len(params.depths)
        self.num_features = int(params.embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(params.npix, params.patch_size, params.in_channels,
                                      params.embed_dim, params.patch_embed_norm,
                                      dtype=params.dtype,
                                      param_dtype=params.param_dtype, rngs=rngs)
        self.pos_drop = nnx.Dropout(params.drop_rate, rngs=rngs)

        num_patches = self.patch_embed.num_patches
        dpr = _drop_path_schedule(params)
        layers = []
        for i in range(self.num_layers):
            layers.append(EncoderStage(
                params, dim=int(params.embed_dim * 2 ** i),
                input_resolution=num_patches // 4 ** i,
                depth=params.depths[i], num_heads=params.num_heads[i],
                drop_path=dpr[sum(params.depths[:i]):sum(params.depths[:i + 1])],
                downsample=i < self.num_layers - 1, rngs=rngs))
        self.layers = nnx.List(layers)
        self.norm = nnx.LayerNorm(self.num_features, epsilon=LN_EPS, dtype=jnp.float32,
                                  param_dtype=params.param_dtype, rngs=rngs)

    def __call__(self, x):
        x = jnp.asarray(x, dtype=jnp.float32)
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        skips = []
        for layer in self.layers:
            skips.append(x)
            x = layer(x)
        return self.norm(x), skips


class HealSwinDecoder(nnx.Module):
    """UNet decoder head producing dense per-pixel outputs."""

    def __init__(self, params, *, rngs):
        self.num_layers = len(params.depths)
        num_patches = params.npix // params.patch_size
        dpr = _drop_path_schedule(params)
        layers_up = []
        concat_back_dim = []
        for i_layer in range(self.num_layers):
            down_idx = self.num_layers - 1 - i_layer
            dim = int(params.embed_dim * 2 ** down_idx)
            concat_back_dim.append(
                nnx.Linear(2 * dim, dim, kernel_init=TRUNC_NORMAL, dtype=params.dtype,
                           param_dtype=params.param_dtype, rngs=rngs)
                if i_layer > 0 else Identity())
            if i_layer == 0:
                layers_up.append(PatchExpand(dim=dim, dim_scale=2, dtype=params.dtype,
                                             param_dtype=params.param_dtype, rngs=rngs))
            else:
                layers_up.append(DecoderStage(
                    params, dim=dim, input_resolution=num_patches // 4 ** down_idx,
                    depth=params.depths[down_idx], num_heads=params.num_heads[down_idx],
                    drop_path=dpr[sum(params.depths[:down_idx]):
                                  sum(params.depths[:down_idx + 1])],
                    upsample=down_idx > 0, rngs=rngs))
        self.layers_up = nnx.List(layers_up)
        self.concat_back_dim = nnx.List(concat_back_dim)
        self.norm_up = nnx.LayerNorm(params.embed_dim, epsilon=LN_EPS, dtype=jnp.float32,
                                     param_dtype=params.param_dtype, rngs=rngs)
        self.up = FinalPatchExpand(patch_size=params.patch_size, dim=params.embed_dim,
                                   dtype=params.dtype,
                                   param_dtype=params.param_dtype, rngs=rngs)
        # emit-fp32 endpoint: constructed fp32 (never a post-hoc astype)
        self.output = nnx.Conv(params.embed_dim, params.out_channels, kernel_size=(1,),
                               use_bias=False, dtype=jnp.float32,
                               param_dtype=params.param_dtype, rngs=rngs)

    def __call__(self, x, skips):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = jnp.concatenate([x, skips[self.num_layers - 1 - inx]], axis=-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)
        x = self.up(x)
        return self.output(x)  # (B, npix, out_channels) channels-last


class HealSwin(nnx.Module):
    """HEALPix Swin V2 U-Net: HealSwinEncoder + HealSwinDecoder."""

    def __init__(self, params, *, rngs):
        self.encoder = HealSwinEncoder(params, rngs=rngs)
        self.decoder = HealSwinDecoder(params, rngs=rngs)

    def __call__(self, x):
        tokens, skips = self.encoder(x)
        return self.decoder(tokens, skips)
