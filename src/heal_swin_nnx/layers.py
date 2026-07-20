"""Shared leaf modules and RoPE primitives used by both models."""
import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx

# Distributional mirror of timm's trunc_normal_(std=0.02). (jax truncates at
# +-2 sigma, timm at absolute +-2 — an init-distribution detail, not a
# functional difference: only affects the tails of the initial parameter
# distribution.)
TRUNC_NORMAL = nnx.initializers.truncated_normal(stddev=0.02)


def l2_normalize(x, axis=-1, eps=1e-12):
    """L2-normalize with a finite gradient at x = 0.

    ``x / max(||x||, eps)`` NaNs in the backward pass for exactly-zero vectors
    (d/dx ||x|| is 0/0 at x = 0, and the clamp doesn't block the NaN), which
    zero-background inputs reach through the zero-initialized biases ahead of
    the first attention block. ``rsqrt(sum(x^2) + eps)`` is smooth at 0 and
    matches the clamped division everywhere else.
    """
    return x * jax.lax.rsqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=True) + eps)


def canonical_float_dtype(value):
    """Canonicalize a DTypeLike into its dtype name ("float32", "bfloat16", ...).

    Params dataclasses store dtypes as canonical strings so
    ``json.dumps(dataclasses.asdict(params))`` keeps working; every jnp/nnx
    API accepts the string form. Floating dtypes only."""
    try:
        dt = jnp.dtype(value)
    except TypeError as e:
        raise ValueError("param_dtype must be a floating DTypeLike, got %r"
                         % (value,)) from e
    if not jnp.issubdtype(dt, jnp.floating):
        raise ValueError("param_dtype must be a floating dtype, got %r" % (value,))
    return dt.name


class Identity(nnx.Module):
    def __call__(self, x):
        return x


class DropPath(nnx.Module):
    """Stochastic depth per sample (port of timm 0.4.12 drop_path)."""

    def __init__(self, rate, *, rngs):
        self.rate = rate
        self.deterministic = False
        self.rngs = rngs

    def __call__(self, x):
        if self.deterministic or self.rate == 0.0:
            return x
        keep_prob = 1.0 - self.rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = jax.random.bernoulli(self.rngs.dropout(), keep_prob, shape)
        return jnp.where(mask, x / keep_prob, jnp.zeros_like(x))


class Mlp(nnx.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, *, rngs):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nnx.Linear(in_features, hidden_features, kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_features, out_features, kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        x = self.fc1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


LN_EPS = 1e-5  # torch nn.LayerNorm default; flax default (1e-6) differs


def init_rope_freqs(head_dim, num_heads, theta, key=None):
    """(2, num_heads, head_dim // 2) x/y RoPE frequency magnitudes.

    rope-vit init_random_2d_freqs: ``key=None`` gives axis-aligned frequencies
    (rope_axial — first D/4 pairs rotate with x, second D/4 with y, all heads
    identical); a PRNG key applies a random per-head rotation of the two axes
    (rope_mixed init)."""
    assert head_dim % 4 == 0, "RoPE needs head_dim divisible by 4"
    mag = 1.0 / theta ** (
        jnp.arange(0, head_dim, 4, dtype=jnp.float32)[: head_dim // 4] / head_dim)
    if key is None:
        angles = jnp.zeros((num_heads, 1), dtype=jnp.float32)
    else:
        angles = jax.random.uniform(key, (num_heads, 1)) * 2 * jnp.pi
    fx = jnp.concatenate([mag * jnp.cos(angles), mag * jnp.cos(jnp.pi / 2 + angles)], axis=-1)
    fy = jnp.concatenate([mag * jnp.sin(angles), mag * jnp.sin(jnp.pi / 2 + angles)], axis=-1)
    return jnp.stack([fx, fy])


def rope_rotation_table(freqs, t_x, t_y):
    """freqs (2, H, D/2) + coords (N,) -> (H, N, D/2, 2, 2) rotation matrices."""
    angles = (t_x[None, :, None] * freqs[0][:, None, :]
              + t_y[None, :, None] * freqs[1][:, None, :])  # (H, N, D/2)
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    return jnp.stack([cos, -sin, sin, cos], axis=-1).reshape(*angles.shape, 2, 2)


def apply_rope(q, k, table):
    """Rotate q, k (B, H, N, D) by table (H, N, D/2, 2, 2). Computed in f32
    (angle precision), returned in the input dtype. Norm-preserving."""
    def rot(x):
        xr = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 1, 2)
        out = table[..., 0] * xr[..., 0] + table[..., 1] * xr[..., 1]
        return out.reshape(*x.shape).astype(x.dtype)
    return rot(q), rot(k)


class PatchMerging(nnx.Module):
    """Merge 4 nested pixels into 1: (B, N, C) -> (B, N/4, dim_scale*C)."""

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
    """Expand 1 pixel into 4 nested pixels: (B, N, C) -> (B, 4N, C*dim_scale/4)."""

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


class FinalPatchExpand(nnx.Module):
    """Undo the patch embedding's downsampling: (B, N, C) -> (B, N*patch_size, C)."""

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


class PatchEmbed(nnx.Module):
    """Non-overlapping 1D patch embedding over the nested pixel sequence."""

    def __init__(self, npix, patch_size, in_channels, embed_dim, norm=False, *, rngs):
        self.npix = npix
        self.num_patches = npix // patch_size
        self.proj = nnx.Conv(in_channels, embed_dim,
                             kernel_size=(patch_size,), strides=(patch_size,),
                             padding="VALID", rngs=rngs)
        self.norm = nnx.LayerNorm(embed_dim, epsilon=LN_EPS, rngs=rngs) if norm else None

    def __call__(self, x):  # (B, N, in_channels) channels-last
        assert x.shape[1] == self.npix, (
            "Input map size (%d) doesn't match model (%d)." % (x.shape[1], self.npix))
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
