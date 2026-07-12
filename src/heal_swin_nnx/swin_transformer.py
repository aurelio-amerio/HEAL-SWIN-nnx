"""Flat 2D SWIN-UNet baseline. Port of models_torch/swin_transformer.py."""
import numpy as np

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx

from heal_swin_nnx.config import DataSpec, SwinTransformerConfig
from heal_swin_nnx.layers import TRUNC_NORMAL, DropPath, Identity, Mlp
from heal_swin_nnx.variables import Buffer

LN_EPS = 1e-5


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


class WindowAttention(nnx.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, use_cos_attn=False, use_rel_pos_bias=True,
                 *, rngs):
        self.dim = dim
        self.window_size = tuple(window_size)
        self.num_heads = num_heads
        self.use_cos_attn = use_cos_attn
        self.use_rel_pos_bias = use_rel_pos_bias
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if use_cos_attn:
            self.logit_scale = nnx.Param(jnp.log(10.0 * jnp.ones((num_heads, 1, 1))))
        # table always exists in the reference, even with use_rel_pos_bias=False
        n_rel = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        self.relative_position_bias_table = nnx.Param(
            TRUNC_NORMAL(rngs.params(), (n_rel, num_heads)))
        self.relative_position_index = Buffer(
            jnp.asarray(flat_relative_position_index(self.window_size)))

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.attn_drop = nnx.Dropout(attn_drop, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)

    def __call__(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_cos_attn:
            qn = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
            kn = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-12)
            attn = qn @ kn.swapaxes(-2, -1)
            logit_scale = jnp.exp(jnp.minimum(self.logit_scale.value, jnp.log(1.0 / 0.01)))
            attn = attn * logit_scale
        else:
            attn = (q * self.scale) @ k.swapaxes(-2, -1)

        if self.use_rel_pos_bias:
            ws_area = self.window_size[0] * self.window_size[1]
            bias = self.relative_position_bias_table.value[
                self.relative_position_index.value.reshape(-1)].reshape(ws_area, ws_area, -1)
            attn = attn + bias.transpose(2, 0, 1)[None]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + mask[None, :, None]
            attn = attn.reshape(-1, self.num_heads, N, N)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).swapaxes(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


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


class SwinTransformerBlock(nnx.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=(4, 4), shift_size=(0, 0),
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, use_masking=True, use_cos_attn=False,
                 use_v2_norm_placement=False, use_rel_pos_bias=True, *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.window_size = tuple(window_size)
        self.shift_size = tuple(shift_size)
        self.use_v2_norm_placement = use_v2_norm_placement
        if (self.input_resolution[0] <= self.window_size[0]
                or self.input_resolution[1] <= self.window_size[1]):
            self.shift_size = (0, 0)
            self.window_size = self.input_resolution
        assert 0 <= self.shift_size[0] < self.window_size[0]
        assert 0 <= self.shift_size[1] < self.window_size[1]

        self.norm1 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                    proj_drop=drop, use_rel_pos_bias=use_rel_pos_bias,
                                    use_cos_attn=use_cos_attn, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop, rngs=rngs)

        if use_masking and (self.shift_size[0] > 0 or self.shift_size[1] > 0):
            self.attn_mask = Buffer(jnp.asarray(flat_shift_mask(
                self.input_resolution, self.window_size, self.shift_size)))
        else:
            self.attn_mask = None

    def __call__(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        if not self.use_v2_norm_placement:
            x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            # bug-for-bug with reference (swin_transformer.py:366-368): shift_size[0] twice
            shifted_x = jnp.roll(x, (-self.shift_size[0], -self.shift_size[0]), axis=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size[0] * self.window_size[1], C)
        mask = None if self.attn_mask is None else self.attn_mask.value
        attn_windows = self.attn(x_windows, mask=mask)
        attn_windows = attn_windows.reshape(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x = jnp.roll(shifted_x, (self.shift_size[0], self.shift_size[1]), axis=(1, 2))
        else:
            x = shifted_x
        x = x.reshape(B, H * W, C)

        if self.use_v2_norm_placement:
            x = shortcut + self.drop_path(self.norm1(x))
            x = x + self.drop_path(self.norm2(self.mlp(x)))
        else:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nnx.Module):
    def __init__(self, input_resolution, dim, *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.patch_size = 4
        self.reduction = nnx.Linear(self.patch_size * dim, 2 * dim, use_bias=False,
                                    kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.norm = nnx.LayerNorm(self.patch_size * dim, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W and H % 2 == 0 and W % 2 == 0
        x = x.reshape(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = jnp.concatenate([x0, x1, x2, x3], axis=-1).reshape(B, -1, self.patch_size * C)
        return self.reduction(self.norm(x))


class PatchExpand(nnx.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.expand = (nnx.Linear(dim, 2 * dim, use_bias=False, kernel_init=TRUNC_NORMAL,
                                  rngs=rngs) if dim_scale == 2 else Identity())
        self.norm = nnx.LayerNorm(dim // dim_scale, epsilon=LN_EPS, rngs=rngs)
        self.dim_scale = 4

    def __call__(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.reshape(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2,
                      c=C // self.dim_scale)
        return self.norm(x.reshape(B, -1, C // self.dim_scale))


class FinalPatchExpand_X4(nnx.Module):
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
    def __init__(self, config, data_spec, *, rngs):
        self.dim_in = tuple(data_spec.dim_in)
        self.patches_resolution = (data_spec.dim_in[0] // config.patch_size[0],
                                   data_spec.dim_in[1] // config.patch_size[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.proj = nnx.Conv(data_spec.f_in, config.embed_dim,
                             kernel_size=tuple(config.patch_size),
                             strides=tuple(config.patch_size), padding="VALID", rngs=rngs)
        self.norm = (nnx.LayerNorm(config.embed_dim, epsilon=LN_EPS, rngs=rngs)
                     if config.patch_embed_norm_layer == "layernorm" else None)

    def __call__(self, x):  # (B, H, W, f_in) channels-last
        B, H, W, C = x.shape
        assert (H, W) == self.dim_in
        x = self.proj(x)                   # (B, Ph, Pw, embed_dim)
        x = x.reshape(B, -1, x.shape[-1])  # (B, Ph*Pw, embed_dim); row-major == torch flatten(2)
        if self.norm is not None:
            x = self.norm(x)
        return x
