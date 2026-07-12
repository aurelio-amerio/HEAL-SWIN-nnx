"""HEAL-SWIN on the HEALPix grid. Port of models_torch/swin_hp_transformer.py."""
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
