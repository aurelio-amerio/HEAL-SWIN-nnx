import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from heal_swin_nnx import (
    DataSpec, SwinHPEncoder, SwinHPTransformerConfig, SwinHPTransformerSys)
from heal_swin_nnx.layers import DropPath, Identity, Mlp


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


def tiny_hp(base_pix=8, **over):
    cfg = SwinHPTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4],
                                  drop_path_rate=0.0, **over)
    ds = DataSpec(dim_in=base_pix * 16 ** 2, f_in=3, f_out=5, base_pix=base_pix)
    return SwinHPTransformerSys(cfg, ds, rngs=nnx.Rngs(0)), ds


def test_jit_matches_eager():
    model, ds = tiny_hp()
    model.eval()
    x = jax.random.normal(jax.random.key(0), (2, ds.dim_in, 3))
    np.testing.assert_allclose(np.asarray(nnx.jit(lambda m, x: m(x))(model, x)),
                               np.asarray(model(x)), rtol=1e-6, atol=1e-6)


def test_batch_independence():
    model, ds = tiny_hp()
    model.eval()
    x = jax.random.normal(jax.random.key(0), (3, ds.dim_in, 3))
    full = np.asarray(model(x))
    single = np.asarray(model(x[1:2]))
    np.testing.assert_allclose(full[1:2], single, rtol=1e-5, atol=1e-6)


def test_remat_matches_no_remat():
    m1, ds = tiny_hp()
    m2, _ = tiny_hp(use_checkpoint=True)
    # same rngs seed -> same weights
    m1.eval(); m2.eval()
    x = jax.random.normal(jax.random.key(0), (1, ds.dim_in, 3))
    np.testing.assert_allclose(np.asarray(m2(x)), np.asarray(m1(x)), rtol=1e-6, atol=1e-6)


def test_encoder_standalone_no_decoder_params():
    cfg = SwinHPTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4],
                                  drop_path_rate=0.0)
    ds = DataSpec(dim_in=2048, f_in=3, f_out=5, base_pix=8)
    enc = SwinHPEncoder(cfg, ds, rngs=nnx.Rngs(0))
    tokens, skips = enc(jnp.ones((1, 2048, 3)))
    assert tokens.shape == (1, 2048 // 4 // 4, 24)   # N/(patch*4^(L-1)), embed*2^(L-1)
    assert len(skips) == 2
    paths = [tuple(str(p) for p in path) for path, _ in nnx.to_flat_state(nnx.state(enc, nnx.Param))]
    assert not any("decoder" in p for path in paths for p in path)


def tiny_hp_pixels(base_pixels, strategy, nside=16):
    cfg = SwinHPTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4],
                                  drop_path_rate=0.0, shift_strategy=strategy)
    ds = DataSpec(dim_in=len(base_pixels) * nside ** 2, f_in=3, f_out=5,
                  base_pixels=base_pixels)
    return SwinHPTransformerSys(cfg, ds, rngs=nnx.Rngs(0)), ds


@pytest.mark.parametrize("strategy",
                         ["nest_roll", "nest_grid_shift", "nest_grid_shift_exact",
                          "ring_shift"])
@pytest.mark.parametrize("base_pixels", [list(range(12)), [8, 9, 10, 11]])
def test_forward_full_sphere_and_south_cap(base_pixels, strategy):
    model, ds = tiny_hp_pixels(base_pixels, strategy)
    model.eval()
    x = jax.random.normal(jax.random.key(0), (2, ds.dim_in, 3))
    y = model(x)
    assert y.shape == (2, ds.dim_in, 5)
    assert np.isfinite(np.asarray(y)).all()


def test_legacy_8pix_path_still_works():
    model, ds = tiny_hp()  # existing helper, base_pix=8
    model.eval()
    x = jax.random.normal(jax.random.key(0), (1, ds.dim_in, 3))
    assert model(x).shape == (1, ds.dim_in, 5)


def test_no_buffer_is_a_param():
    model, _ = tiny_hp(rel_pos_bias="flat")
    params = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))
    for path in params:
        joined = "/".join(str(p) for p in path)
        assert "attn_mask" not in joined and "relative_position_index" not in joined
        assert "shift_idcs" not in joined
