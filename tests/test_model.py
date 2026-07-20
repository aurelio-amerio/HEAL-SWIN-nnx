import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from heal_swin_nnx import Buffer, HealSwin, HealSwinEncoder, HealSwinParams, SwinParams, SwinUnet
from heal_swin_nnx.layers import (DropPath, FinalPatchExpand, Identity, Mlp, PatchEmbed,
                                  PatchExpand, PatchMerging)


def test_identity():
    x = jnp.ones((2, 3))
    assert np.array_equal(Identity()(x), x)


def test_mlp_shapes():
    m = Mlp(8, 32, rngs=nnx.Rngs(0))
    y = m(jnp.ones((2, 5, 8)))
    assert y.shape == (2, 5, 8)


def test_droppath_eval_is_identity():
    dp = DropPath(0.5, rngs=nnx.Rngs(0))
    dp.eval()
    x = jnp.ones((4, 3, 2))
    assert np.array_equal(dp(x), x)


def test_droppath_train_drops_whole_samples():
    dp = DropPath(0.5, rngs=nnx.Rngs(0))
    dp.train()
    x = jnp.ones((512, 4))
    y = np.asarray(dp(x))
    per_sample = y.sum(axis=1)
    assert set(np.round(per_sample, 3)).issubset({0.0, 8.0})  # 4 * 1/keep, keep=0.5
    dropped = float((per_sample == 0).mean())
    assert 0.3 < dropped < 0.7  # ~Bernoulli(0.5)


def tiny_params(**over):
    kw = dict(nside=16, in_channels=3, out_channels=5, base_pixels=tuple(range(8)),
              embed_dim=16, depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0)
    kw.update(over)
    return HealSwinParams(**kw)


def tiny_hp(**over):
    p = tiny_params(**over)
    return HealSwin(p, rngs=nnx.Rngs(0)), p


def test_jit_matches_eager():
    model, p = tiny_hp()
    model.eval()
    x = jax.random.normal(jax.random.key(0), (2, p.npix, 3))
    # tolerance covers float32 jit-fusion reduction-order drift inherent to the
    # 8-block V2 cosine-attention stack (~5e-5 observed); real bugs are >> 1e-4
    np.testing.assert_allclose(np.asarray(nnx.jit(lambda m, x: m(x))(model, x)),
                               np.asarray(model(x)), rtol=1e-4, atol=1e-4)


def test_batch_independence():
    model, p = tiny_hp()
    model.eval()
    x = jax.random.normal(jax.random.key(0), (3, p.npix, 3))
    full = np.asarray(model(x))
    single = np.asarray(model(x[1:2]))
    # tolerance covers float32 batch-tiling reduction-order drift inherent to the
    # 8-block V2 cosine-attention stack (~5e-5 observed); real bugs are >> 1e-4
    np.testing.assert_allclose(full[1:2], single, rtol=1e-4, atol=1e-4)


def test_patch_size_one_forward_shape():
    # patch_size=1: no regrouping, one pixel = one token at stage 0
    model, p = tiny_hp(nside=8, patch_size=1)
    model.eval()
    x = jax.random.normal(jax.random.key(0), (1, p.npix, 3))
    assert model(x).shape == (1, p.npix, 5)


def test_remat_matches_no_remat():
    m1, p = tiny_hp()
    m2, _ = tiny_hp(use_checkpoint=True)
    # same rngs seed -> same weights
    m1.eval(); m2.eval()
    x = jax.random.normal(jax.random.key(0), (1, p.npix, 3))
    np.testing.assert_allclose(np.asarray(m2(x)), np.asarray(m1(x)), rtol=1e-6, atol=1e-6)


def test_encoder_standalone_no_decoder_params():
    p = tiny_params()
    enc = HealSwinEncoder(p, rngs=nnx.Rngs(0))
    tokens, skips = enc(jnp.ones((1, p.npix, 3)))
    assert tokens.shape == (1, p.npix // 4 // 4, 32)   # N/(patch*4^(L-1)), embed*2^(L-1)
    assert len(skips) == 2
    paths = [tuple(str(q) for q in path)
             for path, _ in nnx.to_flat_state(nnx.state(enc, nnx.Param))]
    assert not any("decoder" in q for path in paths for q in path)


@pytest.mark.parametrize("strategy",
                         ["nest_roll", "nest_grid_shift", "nest_grid_shift_exact",
                          "ring_shift"])
@pytest.mark.parametrize("base_pixels", [tuple(range(12)), (8, 9, 10, 11)])
def test_forward_full_sphere_and_south_cap(base_pixels, strategy):
    model, p = tiny_hp(base_pixels=base_pixels, shift_strategy=strategy)
    model.eval()
    x = jax.random.normal(jax.random.key(0), (2, p.npix, 3))
    y = model(x)
    assert y.shape == (2, p.npix, 5)
    assert np.isfinite(np.asarray(y)).all()


def test_grads_finite_on_all_zero_input():
    # Zero-background inputs (e.g. sparse projections onto the sphere) produce
    # exactly-zero q/k vectors in the first block (zero-init biases, post-norm).
    # The cosine-attention normalization must have a finite gradient there:
    # d/dx ||x|| is 0/0 = NaN at x = 0, and clamping the norm does not fix the
    # backward pass.
    model, p = tiny_hp()
    model.train()
    x = jnp.zeros((2, p.npix, 3))
    grads = nnx.grad(lambda m: jnp.mean(m(x) ** 2))(model)
    for path, g in nnx.to_flat_state(grads):
        joined = "/".join(str(q) for q in path)
        assert np.isfinite(np.asarray(g)).all(), f"non-finite grad at {joined}"


def test_flat_grads_finite_on_all_zero_input():
    model, p = tiny_flat()
    model.train()
    x = jnp.zeros((2, *p.img_size, 2))
    grads = nnx.grad(lambda m: jnp.mean(m(x) ** 2))(model)
    for path, g in nnx.to_flat_state(grads):
        joined = "/".join(str(q) for q in path)
        assert np.isfinite(np.asarray(g)).all(), f"non-finite grad at {joined}"


def test_params_are_json_loggable_next_to_model():
    import dataclasses, json
    _, p = tiny_hp()
    json.dumps(dataclasses.asdict(p))


def test_no_buffer_is_a_param():
    model, _ = tiny_hp(pos_embed="rel_bias")
    params = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))
    for path in params:
        joined = "/".join(str(q) for q in path)
        assert "attn_mask" not in joined and "relative_position_index" not in joined
        assert "shift_idcs" not in joined


def test_rope_buffers_and_params_sorted_correctly():
    mixed, _ = tiny_hp(pos_embed="rope_mixed")
    param_paths = ["/".join(str(q) for q in path)
                   for path, _ in nnx.to_flat_state(nnx.state(mixed, nnx.Param))]
    assert any("rope_freqs" in p for p in param_paths)      # learned freqs train
    assert not any("rope_coords" in p for p in param_paths)  # coords are Buffers

    axial, _ = tiny_hp(pos_embed="rope_axial")
    param_paths = ["/".join(str(q) for q in path)
                   for path, _ in nnx.to_flat_state(nnx.state(axial, nnx.Param))]
    assert not any("rope_table" in p for p in param_paths)   # fixed table is a Buffer


def tiny_flat_params(**over):
    kw = dict(img_size=(32, 64), in_channels=2, out_channels=3, embed_dim=16,
              depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0)
    kw.update(over)
    return SwinParams(**kw)


def tiny_flat(**over):
    p = tiny_flat_params(**over)
    return SwinUnet(p, rngs=nnx.Rngs(0)), p


def test_flat_no_buffer_is_a_param():
    model, _ = tiny_flat(pos_embed="rel_bias")
    params = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))
    for path in params:
        joined = "/".join(str(q) for q in path)
        assert "attn_mask" not in joined and "relative_position_index" not in joined


def test_flat_rope_buffers_and_params_sorted_correctly():
    mixed, _ = tiny_flat(pos_embed="rope_mixed")
    param_paths = ["/".join(str(q) for q in path)
                   for path, _ in nnx.to_flat_state(nnx.state(mixed, nnx.Param))]
    assert any("rope_freqs" in p for p in param_paths)      # learned freqs train
    assert not any("rope_coords" in p for p in param_paths)  # coords are Buffers

    axial, _ = tiny_flat(pos_embed="rope_axial")
    param_paths = ["/".join(str(q) for q in path)
                   for path, _ in nnx.to_flat_state(nnx.state(axial, nnx.Param))]
    assert not any("rope_table" in p for p in param_paths)   # fixed table is a Buffer


def test_shared_layers_accept_param_dtype():
    rngs = nnx.Rngs(0)
    mods = [Mlp(8, 32, param_dtype="bfloat16", rngs=rngs),
            PatchMerging(8, param_dtype="bfloat16", rngs=rngs),
            PatchExpand(8, param_dtype="bfloat16", rngs=rngs),
            FinalPatchExpand(4, 8, param_dtype="bfloat16", rngs=rngs),
            PatchEmbed(64, 4, 3, 8, norm=True, param_dtype="bfloat16", rngs=rngs)]
    for m in mods:
        flat = list(nnx.to_flat_state(nnx.state(m, nnx.Param)))
        assert len(flat) > 0
        for path, v in flat:
            # v[...] (not .value — deprecated in this flax version) reads the array
            assert v[...].dtype == jnp.bfloat16, (type(m).__name__, path)


def _param_dtypes(model):
    return {"/".join(str(q) for q in path): v[...].dtype
            for path, v in nnx.to_flat_state(nnx.state(model, nnx.Param))}


def _buffer_dtypes(model):
    return {"/".join(str(q) for q in path): v[...].dtype
            for path, v in nnx.to_flat_state(nnx.state(model, Buffer))}


@pytest.mark.parametrize("pos_embed", ["rope_mixed", "rope_axial", "rel_bias"])
def test_healswin_param_dtype_propagates(pos_embed):
    model, p = tiny_hp(param_dtype="bfloat16", pos_embed=pos_embed)
    model.eval()
    for path, dtype in _param_dtypes(model).items():
        # rope_freqs feeds the f32 RoPE angle computation and stays f32 by design
        expected = jnp.float32 if "rope_freqs" in path else jnp.bfloat16
        assert dtype == expected, path

    ref, _ = tiny_hp(pos_embed=pos_embed)          # buffers ignore param_dtype
    assert _buffer_dtypes(model) == _buffer_dtypes(ref)

    x = jax.random.normal(jax.random.key(0), (2, p.npix, 3))
    y = model(x)
    assert y.dtype == jnp.bfloat16 and y.shape == (2, p.npix, 5)
    assert bool(jnp.isfinite(y).all())
