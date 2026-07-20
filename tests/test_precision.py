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


# --- HealSwin ---------------------------------------------------------------

from heal_swin_nnx import HealSwin, HealSwinParams
from heal_swin_nnx.models.healswin import HealSwinBlock


def make_healswin(**over):
    kw = dict(nside=16, in_channels=3, out_channels=2, base_pixels=(0, 1, 2, 3),
              embed_dim=16, depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0)
    kw.update(over)
    p = HealSwinParams(**kw)
    return HealSwin(p, rngs=nnx.Rngs(0)), p


def _smooth_input(key, npix, channels, batch=2):
    """Deterministic smooth test signal: a few seeded sinusoids over the pixel
    index (closer to real fields than white noise; error behavior differs)."""
    k1, k2 = jax.random.split(key)
    t = jnp.linspace(0.0, 1.0, npix)[None, :, None, None]        # (1, npix, 1, 1)
    amp = jax.random.normal(k1, (batch, 1, channels, 4))
    phase = jax.random.uniform(k2, (batch, 1, channels, 4), maxval=2 * jnp.pi)
    freqs = 2 * jnp.pi * jnp.arange(1.0, 5.0)                    # (4,)
    return jnp.sum(amp * jnp.sin(freqs * t + phase), axis=-1)    # (batch, npix, C)


def _rel_err(a, b):
    a = jnp.asarray(a, jnp.float32)
    b = jnp.asarray(b, jnp.float32)
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(a)) + 1e-12))


@pytest.mark.parametrize("pos_embed", ["rope_mixed", "rel_bias"])
def test_healswin_master_weights_fp32_under_bf16_compute(pos_embed):
    model, _ = make_healswin(dtype="bfloat16", pos_embed=pos_embed)
    for path, v in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        assert v[...].dtype == jnp.float32, path


def test_healswin_output_and_grads_fp32():
    model, p = make_healswin(dtype="bfloat16")
    model.eval()
    x = _smooth_input(jax.random.key(0), p.npix, 3)
    y = model(x)
    assert y.dtype == jnp.float32 and bool(jnp.isfinite(y).all())
    tokens, _ = model.encoder(x)
    assert tokens.dtype == jnp.float32                 # standalone-encoder endpoint
    grads = nnx.grad(lambda m: jnp.mean(m(x) ** 2))(model)
    for path, g in nnx.to_flat_state(grads):
        joined = "/".join(str(q) for q in path)
        assert g[...].dtype == jnp.float32, joined
        assert bool(jnp.isfinite(g[...]).all()), joined


def _block_entry_spy(monkeypatch, cls):
    seen = []
    orig = cls.__call__

    def spy(self, x, *args, **kw):
        seen.append(x.dtype)
        return orig(self, x, *args, **kw)

    monkeypatch.setattr(cls, "__call__", spy)
    return seen


@pytest.mark.parametrize("patch_embed_norm", [False, True])
def test_healswin_stream_is_bf16_inside_blocks(monkeypatch, patch_embed_norm):
    """Leak lock: the dtype entering EVERY block is the compute dtype. Block i's
    entry is block i-1's residual output, so this sweeps all post-norm residual
    casts, PatchMerging's self-heal, PatchExpand's exit cast, the concat path,
    and (parametrized) the PatchEmbed-norm exit cast."""
    seen = _block_entry_spy(monkeypatch, HealSwinBlock)
    model, p = make_healswin(dtype="bfloat16", patch_embed_norm=patch_embed_norm)
    model.eval()
    model(_smooth_input(jax.random.key(0), p.npix, 3))
    assert len(seen) >= 6            # 4 encoder blocks + decoder-stage blocks
    assert all(dt == jnp.bfloat16 for dt in seen), seen


def test_healswin_softmax_island_is_fp32(monkeypatch):
    seen = []
    orig = jax.nn.softmax

    def spy(x, axis=-1, **kw):
        seen.append(x.dtype)
        return orig(x, axis=axis, **kw)

    monkeypatch.setattr(jax.nn, "softmax", spy)
    model, p = make_healswin(dtype="bfloat16")
    model.eval()
    model(_smooth_input(jax.random.key(0), p.npix, 3))
    assert seen and all(dt == jnp.float32 for dt in seen), seen


# --- SwinUnet ---------------------------------------------------------------

from heal_swin_nnx import SwinParams, SwinUnet
from heal_swin_nnx.models.swin import SwinBlock


def make_flat(**over):
    kw = dict(img_size=(32, 64), in_channels=2, out_channels=3, embed_dim=16,
              depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0)
    kw.update(over)
    p = SwinParams(**kw)
    return SwinUnet(p, rngs=nnx.Rngs(0)), p


def _smooth_input_2d(key, img_size, channels, batch=2):
    H, W = img_size
    flat = _smooth_input(key, H * W, channels, batch=batch)
    return flat.reshape(batch, H, W, channels)


def test_swin_local_blocks_emit_compute_dtype():
    """Block-level emit-dtype guard for swin.py's OWN local patch blocks
    (separate classes from layers.py's shared ones — mirrors
    test_shared_blocks_emit_compute_dtype above)."""
    from heal_swin_nnx.models.swin import FinalPatchExpand, PatchExpand, PatchMerging
    rngs = nnx.Rngs(0)
    x = jnp.ones((2, 16, 8), jnp.bfloat16)   # 4x4 resolution, C=8
    assert PatchMerging((4, 4), 8, dtype="bfloat16", rngs=rngs)(x).dtype == jnp.bfloat16
    assert PatchExpand((4, 4), 8, dtype="bfloat16", rngs=rngs)(x).dtype == jnp.bfloat16
    # FinalPatchExpand's sole consumer is the fp32 output conv: deliberate fp32 tail
    assert (FinalPatchExpand((4, 4), (2, 2), 8, dtype="bfloat16", rngs=rngs)(x).dtype
            == jnp.float32)


def test_flat_master_weights_fp32_under_bf16_compute():
    model, _ = make_flat(dtype="bfloat16")
    for path, v in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        assert v[...].dtype == jnp.float32, path


def test_flat_output_and_grads_fp32():
    model, p = make_flat(dtype="bfloat16")
    model.eval()
    x = _smooth_input_2d(jax.random.key(0), p.img_size, 2)
    y = model(x)
    assert y.dtype == jnp.float32 and bool(jnp.isfinite(y).all())
    tokens, _ = model.encoder(x)
    assert tokens.dtype == jnp.float32
    grads = nnx.grad(lambda m: jnp.mean(m(x) ** 2))(model)
    for path, g in nnx.to_flat_state(grads):
        joined = "/".join(str(q) for q in path)
        assert g[...].dtype == jnp.float32, joined


@pytest.mark.parametrize("patch_embed_norm", [False, True])
def test_flat_stream_is_bf16_inside_blocks(monkeypatch, patch_embed_norm):
    seen = _block_entry_spy(monkeypatch, SwinBlock)
    model, p = make_flat(dtype="bfloat16", patch_embed_norm=patch_embed_norm)
    model.eval()
    model(_smooth_input_2d(jax.random.key(0), p.img_size, 2))
    assert len(seen) >= 6
    assert all(dt == jnp.bfloat16 for dt in seen), seen


def test_flat_softmax_island_is_fp32(monkeypatch):
    seen = []
    orig = jax.nn.softmax

    def spy(x, axis=-1, **kw):
        seen.append(x.dtype)
        return orig(x, axis=axis, **kw)

    monkeypatch.setattr(jax.nn, "softmax", spy)
    model, p = make_flat(dtype="bfloat16")
    model.eval()
    model(_smooth_input_2d(jax.random.key(0), p.img_size, 2))
    assert seen and all(dt == jnp.float32 for dt in seen), seen
