"""Flat 2D Swin V2 U-Net (SwinUnet) — the planar sibling of HealSwin."""
from dataclasses import dataclass
from typing import Literal, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx

from heal_swin_nnx.layers import (
    LN_EPS, TRUNC_NORMAL, DropPath, Identity, Mlp, apply_rope, init_rope_freqs,
    l2_normalize, rope_rotation_table)
from heal_swin_nnx.models.healswin import POS_EMBEDS
from heal_swin_nnx.variables import Buffer


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


@dataclass
class SwinParams:
    """Pure-data description of a flat SwinUnet model. Serializable."""

    img_size: Union[int, Tuple[int, int]]
    in_channels: int
    out_channels: int

    patch_size: Union[int, Tuple[int, int]] = (4, 4)
    window_size: Union[int, Tuple[int, int]] = (4, 4)
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 2, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    pos_embed: Literal["none", "rel_bias", "rope_axial", "rope_mixed"] = "rel_bias"
    rope_theta: float = 10.0
    use_masking: bool = True
    patch_embed_norm: bool = False

    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    use_checkpoint: bool = False

    def __post_init__(self):
        self.img_size = _pair(self.img_size)
        self.patch_size = _pair(self.patch_size)
        self.window_size = _pair(self.window_size)
        self.depths = tuple(self.depths)
        self.num_heads = tuple(self.num_heads)

        if self.pos_embed not in POS_EMBEDS:
            raise ValueError("pos_embed must be one of %r, got %r"
                             % (POS_EMBEDS, self.pos_embed))
        if len(self.depths) != len(self.num_heads):
            raise ValueError("depths (%d) and num_heads (%d) must have equal length"
                             % (len(self.depths), len(self.num_heads)))
        merge = 2 ** (len(self.depths) - 1)
        for a in range(2):
            div = merge * self.patch_size[a] * self.window_size[a]
            if self.img_size[a] % div:
                raise ValueError(
                    "img_size[%d]=%d must be divisible by patch*window*2^(n_stages-1)=%d"
                    % (a, self.img_size[a], div))
        for i, heads in enumerate(self.num_heads):
            dim = self.embed_dim * 2 ** i
            if dim % heads:
                raise ValueError("stage %d: dim %d not divisible by num_heads %d"
                                 % (i, dim, heads))
            if self.pos_embed in ("rope_axial", "rope_mixed") and (dim // heads) % 4:
                raise ValueError(
                    "stage %d: head_dim %d must be divisible by 4 for RoPE"
                    % (i, dim // heads))

    @property
    def patches_resolution(self):
        return (self.img_size[0] // self.patch_size[0],
                self.img_size[1] // self.patch_size[1])

    @property
    def shift_size(self):
        return (self.window_size[0] // 2, self.window_size[1] // 2)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    return x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size[0], window_size[1], C)


def window_reverse(windows, window_size, H, W):
    B = windows.shape[0] // ((H // window_size[0]) * (W // window_size[1]))
    x = windows.reshape(B, H // window_size[0], W // window_size[1],
                        window_size[0], window_size[1], -1)
    return x.transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, x.shape[-1])


def flat_relative_position_index(window_size):
    coords = np.stack(np.meshgrid(np.arange(window_size[0]), np.arange(window_size[1]),
                                  indexing="ij"))
    flat = coords.reshape(2, -1)
    rel = flat[:, :, None] - flat[:, None, :]
    rel = rel.transpose(1, 2, 0).astype(np.int64)
    rel[:, :, 0] += window_size[0] - 1
    rel[:, :, 1] += window_size[1] - 1
    rel[:, :, 0] *= 2 * window_size[1] - 1
    return rel.sum(-1)


def flat_win_coords(window_size):
    """(2, Wh*Ww) float32: (x, y) of each token in a row-major-flattened
    (Wh, Ww) window. x = column, y = row (rope-vit init_t_xy convention)."""
    n = np.arange(window_size[0] * window_size[1])
    return np.stack([(n % window_size[1]).astype(np.float32),
                     (n // window_size[1]).astype(np.float32)])


def flat_shift_mask(input_resolution, window_size, shift_size):
    H, W = input_resolution
    img_mask = np.zeros((1, H, W, 1), dtype=np.float32)
    h_slices = (slice(0, -window_size[0]), slice(-window_size[0], -shift_size[0]),
                slice(-shift_size[0], None))
    w_slices = (slice(0, -window_size[1]), slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mw = img_mask.reshape(1, H // window_size[0], window_size[0],
                          W // window_size[1], window_size[1], 1)
    mw = mw.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size[0] * window_size[1])
    attn_mask = mw[:, None, :] - mw[:, :, None]
    return np.where(attn_mask != 0, np.float32(-100.0), np.float32(0.0))


class WindowAttention(nnx.Module):
    """Swin V2 window attention (flat 2D windows), positional encoding
    selected by ``params.pos_embed``."""

    def __init__(self, params, dim, num_heads, window_size, *, rngs):
        self.window_size = tuple(window_size)
        self.num_heads = num_heads
        self.pos_embed = params.pos_embed
        head_dim = dim // num_heads
        self.logit_scale = nnx.Param(jnp.log(10.0 * jnp.ones((num_heads, 1, 1))))

        if self.pos_embed == "rel_bias":
            n_rel = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
            self.relative_position_bias_table = nnx.Param(
                TRUNC_NORMAL(rngs.params(), (n_rel, num_heads)))
            self.relative_position_index = Buffer(
                jnp.asarray(flat_relative_position_index(self.window_size)))
        elif self.pos_embed in ("rope_axial", "rope_mixed"):
            coords = jnp.asarray(flat_win_coords(self.window_size))
            if self.pos_embed == "rope_mixed":
                self.rope_freqs = nnx.Param(init_rope_freqs(
                    head_dim, num_heads, params.rope_theta, key=rngs.params()))
                self.rope_coords = Buffer(coords)
            else:
                freqs = init_rope_freqs(head_dim, num_heads, params.rope_theta)
                self.rope_table = Buffer(rope_rotation_table(freqs, coords[0], coords[1]))

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=params.qkv_bias,
                              kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.attn_drop = nnx.Dropout(params.attn_drop_rate, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=TRUNC_NORMAL, rngs=rngs)
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
        attn = q @ k.swapaxes(-2, -1)
        logit_scale = jnp.exp(jnp.minimum(self.logit_scale[...], jnp.log(1.0 / 0.01)))
        attn = attn * logit_scale

        if self.pos_embed == "rel_bias":
            ws_area = self.window_size[0] * self.window_size[1]
            bias = self.relative_position_bias_table[...][
                self.relative_position_index[...].reshape(-1)].reshape(ws_area, ws_area, -1)
            attn = attn + bias.transpose(2, 0, 1)[None]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + mask[None, :, None]
            attn = attn.reshape(-1, self.num_heads, N, N)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).swapaxes(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinBlock(nnx.Module):
    def __init__(self, params, dim, input_resolution, num_heads, shifted, drop_path,
                 *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.window_size = tuple(params.window_size)
        shift_size = params.shift_size if shifted else (0, 0)
        if (self.input_resolution[0] <= self.window_size[0]
                or self.input_resolution[1] <= self.window_size[1]):
            shift_size = (0, 0)
            self.window_size = self.input_resolution
        self.shift_size = shift_size

        self.norm1 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.attn = WindowAttention(params, dim, num_heads, self.window_size, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * params.mlp_ratio), drop=params.drop_rate, rngs=rngs)

        if params.use_masking and (shift_size[0] > 0 or shift_size[1] > 0):
            self.attn_mask = Buffer(jnp.asarray(flat_shift_mask(
                self.input_resolution, self.window_size, shift_size)))
        else:
            self.attn_mask = None

    def __call__(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.reshape(B, H, W, C)

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            shifted_x = jnp.roll(x, (-self.shift_size[0], -self.shift_size[1]), axis=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size[0] * self.window_size[1], C)
        mask = None if self.attn_mask is None else self.attn_mask[...]
        attn_windows = self.attn(x_windows, mask=mask)
        attn_windows = attn_windows.reshape(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x = jnp.roll(shifted_x, (self.shift_size[0], self.shift_size[1]), axis=(1, 2))
        else:
            x = shifted_x
        x = x.reshape(B, H * W, C)

        x = shortcut + self.drop_path(self.norm1(x))
        return x + self.drop_path(self.norm2(self.mlp(x)))


class PatchMerging(nnx.Module):
    def __init__(self, input_resolution, dim, *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.reduction = nnx.Linear(4 * dim, 2 * dim, use_bias=False,
                                    kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.norm = nnx.LayerNorm(4 * dim, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W and H % 2 == 0 and W % 2 == 0
        x = x.reshape(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = jnp.concatenate([x0, x1, x2, x3], axis=-1).reshape(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class PatchExpand(nnx.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.expand = (nnx.Linear(dim, 2 * dim, use_bias=False, kernel_init=TRUNC_NORMAL,
                                  rngs=rngs) if dim_scale == 2 else Identity())
        self.norm = nnx.LayerNorm(dim // dim_scale, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.reshape(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        return self.norm(x.reshape(B, -1, C // 4))


class FinalPatchExpand(nnx.Module):
    def __init__(self, input_resolution, patch_size, dim, *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.patch_size = tuple(patch_size)
        self.output_dim = dim
        self.expand = nnx.Linear(dim, self.patch_size[0] * self.patch_size[1] * dim,
                                 use_bias=False, kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.reshape(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c",
                      p1=self.patch_size[0], p2=self.patch_size[1],
                      c=C // (self.patch_size[0] * self.patch_size[1]))
        return self.norm(x.reshape(B, -1, self.output_dim))


class PatchEmbed(nnx.Module):
    def __init__(self, params, *, rngs):
        self.img_size = params.img_size
        self.num_patches = params.patches_resolution[0] * params.patches_resolution[1]
        self.proj = nnx.Conv(params.in_channels, params.embed_dim,
                             kernel_size=tuple(params.patch_size),
                             strides=tuple(params.patch_size), padding="VALID", rngs=rngs)
        self.norm = (nnx.LayerNorm(params.embed_dim, epsilon=LN_EPS, rngs=rngs)
                     if params.patch_embed_norm else None)

    def __call__(self, x):  # (B, H, W, in_channels) channels-last
        B, H, W, C = x.shape
        assert (H, W) == self.img_size
        x = self.proj(x)                   # (B, Ph, Pw, embed_dim)
        x = x.reshape(B, -1, x.shape[-1])  # row-major flatten
        if self.norm is not None:
            x = self.norm(x)
        return x


def _make_blocks(params, dim, input_resolution, depth, num_heads, drop_path, rngs):
    return [SwinBlock(params, dim, input_resolution, num_heads,
                      shifted=(i % 2 == 1), drop_path=drop_path[i], rngs=rngs)
            for i in range(depth)]


class EncoderStage(nnx.Module):
    def __init__(self, params, dim, input_resolution, depth, num_heads, drop_path,
                 downsample, *, rngs):
        self.use_checkpoint = params.use_checkpoint
        self.blocks = nnx.List(_make_blocks(params, dim, input_resolution, depth,
                                            num_heads, drop_path, rngs))
        self.downsample = (PatchMerging(input_resolution, dim=dim, rngs=rngs)
                           if downsample else None)

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
        self.upsample = (PatchExpand(input_resolution, dim=dim, dim_scale=2, rngs=rngs)
                         if upsample else None)

    def __call__(self, x):
        for blk in self.blocks:
            x = nnx.remat(type(blk).__call__)(blk, x) if self.use_checkpoint else blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


def _drop_path_schedule(params):
    return [float(v) for v in np.linspace(0, params.drop_path_rate, sum(params.depths))]


class SwinEncoder(nnx.Module):
    def __init__(self, params, *, rngs):
        self.params = params
        self.num_layers = len(params.depths)
        self.num_features = int(params.embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(params, rngs=rngs)
        self.pos_drop = nnx.Dropout(params.drop_rate, rngs=rngs)

        pr = params.patches_resolution
        dpr = _drop_path_schedule(params)
        layers = []
        for i in range(self.num_layers):
            layers.append(EncoderStage(
                params, dim=int(params.embed_dim * 2 ** i),
                input_resolution=(pr[0] // 2 ** i, pr[1] // 2 ** i),
                depth=params.depths[i], num_heads=params.num_heads[i],
                drop_path=dpr[sum(params.depths[:i]):sum(params.depths[:i + 1])],
                downsample=i < self.num_layers - 1, rngs=rngs))
        self.layers = nnx.List(layers)
        self.norm = nnx.LayerNorm(self.num_features, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        skips = []
        for layer in self.layers:
            skips.append(x)
            x = layer(x)
        return self.norm(x), skips


class SwinDecoder(nnx.Module):
    def __init__(self, params, *, rngs):
        self.params = params
        self.num_layers = len(params.depths)
        pr = params.patches_resolution
        dpr = _drop_path_schedule(params)
        layers_up = []
        concat_back_dim = []
        for i_layer in range(self.num_layers):
            down_idx = self.num_layers - 1 - i_layer
            dim = int(params.embed_dim * 2 ** down_idx)
            res = (pr[0] // 2 ** down_idx, pr[1] // 2 ** down_idx)
            concat_back_dim.append(
                nnx.Linear(2 * dim, dim, kernel_init=TRUNC_NORMAL, rngs=rngs)
                if i_layer > 0 else Identity())
            if i_layer == 0:
                layers_up.append(PatchExpand(res, dim=dim, dim_scale=2, rngs=rngs))
            else:
                layers_up.append(DecoderStage(
                    params, dim=dim, input_resolution=res, depth=params.depths[down_idx],
                    num_heads=params.num_heads[down_idx],
                    drop_path=dpr[sum(params.depths[:down_idx]):
                                  sum(params.depths[:down_idx + 1])],
                    upsample=i_layer < self.num_layers - 1, rngs=rngs))
        self.layers_up = nnx.List(layers_up)
        self.concat_back_dim = nnx.List(concat_back_dim)
        self.norm_up = nnx.LayerNorm(params.embed_dim, epsilon=LN_EPS, rngs=rngs)
        self.up = FinalPatchExpand(pr, patch_size=params.patch_size,
                                   dim=params.embed_dim, rngs=rngs)
        self.output = nnx.Conv(params.embed_dim, params.out_channels, kernel_size=(1, 1),
                               use_bias=False, rngs=rngs)

    def __call__(self, x, skips):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = jnp.concatenate([x, skips[self.num_layers - 1 - inx]], axis=-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)
        x = self.up(x)  # (B, H*W, embed_dim) at full resolution
        H, W = self.params.img_size
        x = x.reshape(x.shape[0], H, W, x.shape[-1])
        return self.output(x)  # (B, H, W, out_channels)


class SwinUnet(nnx.Module):
    """Flat 2D Swin V2 U-Net: SwinEncoder + SwinDecoder."""

    def __init__(self, params, *, rngs):
        self.encoder = SwinEncoder(params, rngs=rngs)
        self.decoder = SwinDecoder(params, rngs=rngs)

    def __call__(self, x):
        tokens, skips = self.encoder(x)
        return self.decoder(tokens, skips)
