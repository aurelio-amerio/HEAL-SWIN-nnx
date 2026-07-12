"""Shared leaf modules and RoPE primitives used by both models."""
import jax
import jax.numpy as jnp
from flax import nnx

# Distributional mirror of timm's trunc_normal_(std=0.02). (jax truncates at
# +-2 sigma, timm at absolute +-2 — irrelevant for parity tests, which always
# use transferred weights.)
TRUNC_NORMAL = nnx.initializers.truncated_normal(stddev=0.02)


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
