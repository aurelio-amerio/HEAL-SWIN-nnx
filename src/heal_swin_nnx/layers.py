"""Shared leaf modules (identical between HP and flat models in the reference)."""
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
