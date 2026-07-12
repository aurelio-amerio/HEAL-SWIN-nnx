"""HEAL-SWIN on the HEALPix grid. Port of models_torch/swin_hp_transformer.py."""
import math

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx

from heal_swin_nnx import hp_shifting
from heal_swin_nnx.config import DataSpec, SwinHPTransformerConfig
from heal_swin_nnx.hp_windowing import (
    nest_relative_position_index, window_partition, window_reverse)
from heal_swin_nnx.layers import TRUNC_NORMAL, DropPath, Identity, Mlp
from heal_swin_nnx.variables import Buffer

LN_EPS = 1e-5  # torch nn.LayerNorm default; flax default (1e-6) breaks parity


class WindowAttention(nnx.Module):
    def __init__(self, dim, window_size, num_heads, rel_pos_bias=None, qkv_bias=True,
                 qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_cos_attn=False, *, rngs):
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.use_cos_attn = use_cos_attn
        self.rel_pos_bias = rel_pos_bias
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if use_cos_attn:
            self.logit_scale = nnx.Param(jnp.log(10.0 * jnp.ones((num_heads, 1, 1))))
        if rel_pos_bias == "flat":
            s = int(round(window_size ** 0.5))
            # zeros init: the reference's trunc_normal_ call for this table is commented out
            self.relative_position_bias_table = nnx.Param(
                jnp.zeros(((2 * s - 1) ** 2, num_heads)))
            self.relative_position_index = Buffer(
                jnp.asarray(nest_relative_position_index(window_size)))

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

        if self.rel_pos_bias is not None:
            bias = self.relative_position_bias_table.value[self.relative_position_index.value]
            attn = attn + bias.transpose(2, 0, 1)[None]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + mask[None, :, None]
            attn = attn.reshape(-1, self.num_heads, N, N)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class PatchMerging(nnx.Module):
    def __init__(self, dim, dim_scale=2, *, rngs):
        self.reduction = nnx.Linear(4 * dim, dim_scale * dim, use_bias=False,
                                    kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.norm = nnx.LayerNorm(4 * dim, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        assert N % 4 == 0, "x size %d is not divisible by 4 as necessary for patching." % N
        x = jnp.concatenate([x[:, 0::4], x[:, 1::4], x[:, 2::4], x[:, 3::4]], axis=-1)
        return self.reduction(self.norm(x))


class PatchExpand(nnx.Module):
    def __init__(self, dim, dim_scale=2, *, rngs):
        self.expand = (nnx.Linear(dim, dim_scale * dim, use_bias=False,
                                  kernel_init=TRUNC_NORMAL, rngs=rngs)
                       if dim_scale != 1 else Identity())
        self.norm = nnx.LayerNorm(dim * dim_scale // 4, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        x = self.expand(x)
        C = x.shape[-1]
        x = rearrange(x, "b n (p c) -> b (n p) c", p=4, c=C // 4)
        return self.norm(x)


class FinalPatchExpand_X4(nnx.Module):
    def __init__(self, patch_size, dim, *, rngs):
        self.patch_size = patch_size
        self.expand = nnx.Linear(dim, patch_size * dim, use_bias=False,
                                 kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        x = self.expand(x)
        C = x.shape[-1]
        x = rearrange(x, "b n (p c) -> b (n p) c", p=self.patch_size, c=C // self.patch_size)
        return self.norm(x)


class SwinTransformerBlock(nnx.Module):
    def __init__(self, dim, input_resolution, base_pix, num_heads, window_size=4, shift_size=0,
                 shift_strategy="nest_roll", rel_pos_bias=None, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 use_v2_norm_placement=False, use_cos_attn=False, *, rngs):
        self.input_resolution = input_resolution
        self.use_v2_norm_placement = use_v2_norm_placement
        self.window_size = window_size
        self.shift_size = shift_size
        if self.input_resolution <= self.window_size:
            self.shift_size = 0
            self.window_size = self.input_resolution

        self.norm1 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads,
                                    rel_pos_bias=rel_pos_bias, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                                    use_cos_attn=use_cos_attn, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop, rngs=rngs)

        nside = math.sqrt(input_resolution // base_pix)
        assert nside % 1 == 0, "nside has to be an integer in every layer"
        nside = int(nside)

        if self.shift_size > 0:
            if shift_strategy == "nest_roll":
                self.shifter = hp_shifting.NestRollShift(
                    shift_size=self.shift_size, input_resolution=self.input_resolution,
                    window_size=self.window_size)
            elif shift_strategy == "nest_grid_shift":
                self.shifter = hp_shifting.NestGridShift(
                    nside=nside, base_pix=base_pix, window_size=self.window_size)
            elif shift_strategy == "ring_shift":
                self.shifter = hp_shifting.RingShift(
                    nside=nside, base_pix=base_pix, window_size=self.window_size,
                    shift_size=self.shift_size)
            else:
                raise ValueError("unknown shift_strategy %r" % shift_strategy)
        else:
            self.shifter = hp_shifting.NoShift()

    def __call__(self, x):
        shortcut = x
        if not self.use_v2_norm_placement:
            x = self.norm1(x)

        shifted_x = self.shifter.shift(x)
        x_windows = window_partition(shifted_x, self.window_size)
        mask = None if self.shifter.attn_mask is None else self.shifter.attn_mask.value
        attn_windows = self.attn(x_windows, mask=mask)
        shifted_x = window_reverse(attn_windows, self.window_size, self.input_resolution)
        x = self.shifter.shift_back(shifted_x)

        if self.use_v2_norm_placement:
            x = shortcut + self.drop_path(self.norm1(x))
            x = x + self.drop_path(self.norm2(self.mlp(x)))
        else:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _make_blocks(dim, input_resolution, base_pix, depth, num_heads, window_size, shift_size,
                 shift_strategy, rel_pos_bias, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                 drop_path, use_v2_norm_placement, use_cos_attn, rngs):
    return [SwinTransformerBlock(
        dim=dim, input_resolution=input_resolution, base_pix=base_pix, num_heads=num_heads,
        window_size=window_size, shift_size=0 if (i % 2 == 0) else shift_size,
        shift_strategy=shift_strategy, rel_pos_bias=rel_pos_bias, mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        use_v2_norm_placement=use_v2_norm_placement, use_cos_attn=use_cos_attn, rngs=rngs)
        for i in range(depth)]


class BasicLayer(nnx.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, base_pix,
                 shift_size, shift_strategy, rel_pos_bias, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, downsample=False,
                 use_checkpoint=False, use_v2_norm_placement=False, use_cos_attn=False, *, rngs):
        self.use_checkpoint = use_checkpoint
        self.blocks = _make_blocks(dim, input_resolution, base_pix, depth, num_heads,
                                   window_size, shift_size, shift_strategy, rel_pos_bias,
                                   mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path,
                                   use_v2_norm_placement, use_cos_attn, rngs)
        self.downsample = PatchMerging(dim=dim, rngs=rngs) if downsample else None

    def __call__(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = nnx.remat(type(blk).__call__)(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayer_up(nnx.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, base_pix,
                 shift_size, shift_strategy, rel_pos_bias, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, upsample=False,
                 use_checkpoint=False, use_v2_norm_placement=False, use_cos_attn=False, *, rngs):
        self.use_checkpoint = use_checkpoint
        self.blocks = _make_blocks(dim, input_resolution, base_pix, depth, num_heads,
                                   window_size, shift_size, shift_strategy, rel_pos_bias,
                                   mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path,
                                   use_v2_norm_placement, use_cos_attn, rngs)
        self.upsample = PatchExpand(dim=dim, dim_scale=2, rngs=rngs) if upsample else None

    def __call__(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = nnx.remat(type(blk).__call__)(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x
