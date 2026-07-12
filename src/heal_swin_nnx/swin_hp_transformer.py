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
            logit_scale = jnp.exp(jnp.minimum(self.logit_scale[...], jnp.log(1.0 / 0.01)))
            attn = attn * logit_scale
        else:
            attn = (q * self.scale) @ k.swapaxes(-2, -1)

        if self.rel_pos_bias is not None:
            bias = self.relative_position_bias_table[...][self.relative_position_index[...]]
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
                # base_pix here is still the legacy pixel *count*; the 0..base_pix-1
                # face-id sequence matches the reference 8-base-pixel fisheye subset.
                # Task 6+ threads an explicit base_pixels sequence through this class.
                self.shifter = hp_shifting.NestGridShift(
                    nside=nside, base_pixels=list(range(base_pix)), window_size=self.window_size)
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
        mask = None if self.shifter.attn_mask is None else self.shifter.attn_mask[...]
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
        self.blocks = nnx.List(_make_blocks(dim, input_resolution, base_pix, depth, num_heads,
                                            window_size, shift_size, shift_strategy, rel_pos_bias,
                                            mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                            drop_path, use_v2_norm_placement, use_cos_attn, rngs))
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
        self.blocks = nnx.List(_make_blocks(dim, input_resolution, base_pix, depth, num_heads,
                                            window_size, shift_size, shift_strategy, rel_pos_bias,
                                            mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                            drop_path, use_v2_norm_placement, use_cos_attn, rngs))
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


class PatchEmbed(nnx.Module):
    def __init__(self, config, data_spec, *, rngs):
        assert config.patch_size % 4 == 0, "required for valid nside in deeper layers"
        self.dim_in = data_spec.dim_in
        self.num_patches = data_spec.dim_in // config.patch_size
        self.proj = nnx.Conv(data_spec.f_in, config.embed_dim,
                             kernel_size=(config.patch_size,), strides=(config.patch_size,),
                             padding="VALID", rngs=rngs)
        self.norm = (nnx.LayerNorm(config.embed_dim, epsilon=LN_EPS, rngs=rngs)
                     if config.patch_embed_norm_layer == "layernorm" else None)

    def __call__(self, x):  # (B, N, f_in) channels-last
        assert x.shape[1] == self.dim_in, (
            "Input image size (%d) doesn't match model (%d)." % (x.shape[1], self.dim_in))
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinHPEncoder(nnx.Module):
    """Compression-only backbone: patch embed + APE + encoder stages + final norm.
    Standalone-usable (tokenizer / embedder); allocates no decoder parameters."""

    def __init__(self, config: SwinHPTransformerConfig, data_spec: DataSpec, *, rngs):
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(config, data_spec, rngs=rngs)
        num_patches = self.patch_embed.num_patches
        if config.ape:
            self.absolute_pos_embed = nnx.Param(
                TRUNC_NORMAL(rngs.params(), (1, num_patches, config.embed_dim)))
        else:
            self.absolute_pos_embed = None
        self.pos_drop = nnx.Dropout(config.drop_rate, rngs=rngs)

        dpr = [float(v) for v in np.linspace(0, config.drop_path_rate, sum(config.depths))]
        layers = []
        for i_layer in range(self.num_layers):
            layers.append(BasicLayer(
                dim=int(config.embed_dim * 2 ** i_layer),
                input_resolution=num_patches // (4 ** i_layer),
                depth=config.depths[i_layer], num_heads=config.num_heads[i_layer],
                window_size=config.window_size, base_pix=data_spec.base_pix,
                shift_size=config.shift_size, shift_strategy=config.shift_strategy,
                rel_pos_bias=config.rel_pos_bias, mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                use_cos_attn=config.use_cos_attn, drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])],
                use_v2_norm_placement=config.use_v2_norm_placement,
                downsample=i_layer < self.num_layers - 1,
                use_checkpoint=config.use_checkpoint, rngs=rngs))
        self.layers = nnx.List(layers)
        self.norm = nnx.LayerNorm(self.num_features, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed[...]
        x = self.pos_drop(x)
        skips = []
        for layer in self.layers:
            skips.append(x)
            x = layer(x)
        return self.norm(x), skips


class HPUnetDecoder(nnx.Module):
    """UNet decoder head (dense per-pixel outputs). Named UnetDecoder in the reference."""

    def __init__(self, config: SwinHPTransformerConfig, data_spec: DataSpec, *, rngs):
        self.num_layers = len(config.depths)
        num_patches = data_spec.dim_in // config.patch_size
        dpr = [float(v) for v in np.linspace(0, config.drop_path_rate, sum(config.depths))]
        layers_up = []
        concat_back_dim = []
        for i_layer in range(self.num_layers):
            down_idx = self.num_layers - 1 - i_layer
            concat_out = int(config.embed_dim * 2 ** down_idx)
            concat_back_dim.append(
                nnx.Linear(2 * concat_out, concat_out, kernel_init=TRUNC_NORMAL, rngs=rngs)
                if i_layer > 0 else Identity())
            if i_layer == 0:
                layers_up.append(PatchExpand(dim=concat_out, dim_scale=2, rngs=rngs))
            else:
                layers_up.append(BasicLayer_up(
                    dim=concat_out, input_resolution=num_patches // (4 ** down_idx),
                    depth=config.depths[down_idx], num_heads=config.num_heads[down_idx],
                    window_size=config.window_size, base_pix=data_spec.base_pix,
                    shift_size=config.shift_size, shift_strategy=config.shift_strategy,
                    rel_pos_bias=config.rel_pos_bias, mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                    use_cos_attn=config.use_cos_attn, drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path=dpr[sum(config.depths[:down_idx]):sum(config.depths[:down_idx + 1])],
                    use_v2_norm_placement=config.use_v2_norm_placement,
                    upsample=down_idx > 0, use_checkpoint=config.use_checkpoint, rngs=rngs))
        self.layers_up = nnx.List(layers_up)
        self.concat_back_dim = nnx.List(concat_back_dim)
        self.up = FinalPatchExpand_X4(patch_size=config.patch_size, dim=config.embed_dim,
                                      rngs=rngs)
        self.output = nnx.Conv(config.embed_dim, data_spec.f_out, kernel_size=(1,),
                               use_bias=False, rngs=rngs)
        self.norm_up = nnx.LayerNorm(config.embed_dim, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x, skips, return_intermediates=False):
        intermediates = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = jnp.concatenate([x, skips[self.num_layers - 1 - inx]], axis=-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
            if return_intermediates:
                intermediates.append(x)
        x = self.norm_up(x)
        x = self.up(x)
        x = self.output(x)  # (B, N, f_out) channels-last
        return (x, intermediates) if return_intermediates else x


class SwinHPTransformerSys(nnx.Module):
    def __init__(self, config: SwinHPTransformerConfig, data_spec: DataSpec, *, rngs):
        self.encoder = SwinHPEncoder(config, data_spec, rngs=rngs)
        self.decoder = HPUnetDecoder(config, data_spec, rngs=rngs)

    def __call__(self, x):
        tokens, skips = self.encoder(x)
        return self.decoder(tokens, skips)
