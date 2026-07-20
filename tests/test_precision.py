"""Mixed-precision contract tests: fp32 master weights, bf16 compute knob,
fp32 islands, leak locks (spy tests), and the calibrated drift lock.

Spec: docs/superpowers/specs/2026-07-20-compute-dtype-design.md
"""
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from heal_swin_nnx.layers import (FinalPatchExpand, Mlp, PatchEmbed, PatchExpand,
                                  PatchMerging, l2_normalize)


def test_l2_normalize_is_fp32_island():
    x = jax.random.normal(jax.random.key(0), (2, 3, 8)).astype(jnp.bfloat16)
    y = l2_normalize(x)
    assert y.dtype == jnp.bfloat16                       # emits input dtype
    ref = l2_normalize(x.astype(jnp.float32))
    assert float(jnp.max(jnp.abs(y.astype(jnp.float32) - ref))) < 1e-2
    assert float(jnp.max(jnp.abs(
        jnp.sum(ref * ref, axis=-1) - 1.0))) < 1e-5      # actually normalizes


def test_shared_blocks_emit_compute_dtype():
    rngs = nnx.Rngs(0)
    x = jnp.ones((2, 16, 8), jnp.bfloat16)
    assert Mlp(8, 16, dtype="bfloat16", rngs=rngs)(x).dtype == jnp.bfloat16
    assert PatchMerging(8, dtype="bfloat16", rngs=rngs)(x).dtype == jnp.bfloat16
    assert PatchExpand(8, dtype="bfloat16", rngs=rngs)(x).dtype == jnp.bfloat16
    # FinalPatchExpand's sole consumer is the fp32 output conv: deliberate fp32 tail
    assert FinalPatchExpand(4, 8, dtype="bfloat16", rngs=rngs)(x).dtype == jnp.float32
    xe = jnp.ones((2, 64, 3), jnp.bfloat16)
    pe = PatchEmbed(64, 4, 3, 8, norm=True, dtype="bfloat16", rngs=rngs)
    assert pe(xe).dtype == jnp.bfloat16                  # norm exit cast
    pe2 = PatchEmbed(64, 4, 3, 8, norm=False, dtype="bfloat16", rngs=rngs)
    assert pe2(xe).dtype == jnp.bfloat16                 # conv computes bf16


def test_shared_blocks_master_weights_follow_param_dtype_not_dtype():
    rngs = nnx.Rngs(0)
    m = Mlp(8, 16, param_dtype="float32", dtype="bfloat16", rngs=rngs)
    for path, v in nnx.to_flat_state(nnx.state(m, nnx.Param)):
        assert v[...].dtype == jnp.float32, path
