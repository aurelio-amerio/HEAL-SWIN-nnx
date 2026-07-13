import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from heal_swin_nnx.models.healconv import HealConvBlock, HealConvParams


def tiny_conv_params(**over):
    kw = dict(nside=16, in_channels=3, out_channels=5, base_pixels=tuple(range(8)),
              embed_dim=16, depths=(2, 2), drop_path_rate=0.0)
    kw.update(over)
    return HealConvParams(**kw)


def make_block(shifted, **over):
    p = tiny_conv_params(**over)
    N = p.npix // p.patch_size            # first-stage resolution
    blk = HealConvBlock(p, dim=8, input_resolution=N, shifted=shifted,
                        drop_path=0.0, rngs=nnx.Rngs(0))
    blk.eval()
    return blk, p, N


def test_grid_perm_round_trip():
    blk, _, _ = make_block(shifted=False)
    perm = np.asarray(blk.grid_perm[...])
    inv = np.asarray(blk.inv_perm[...])
    ws = blk.window_size
    np.testing.assert_array_equal(perm[inv], np.arange(ws))
    np.testing.assert_array_equal(inv[perm], np.arange(ws))
    assert blk.grid_size ** 2 == ws


def _set_delta_kernel(blk, dim):
    """Depthwise kernel that makes the conv an exact identity under SAME padding.

    lax 'SAME' pads (k-1)//2 low, so tap index (k-1)//2 has zero spatial offset
    (k is even here: 2, 4, 8 — there is no 'center' tap)."""
    k = blk.grid_size
    kern = np.zeros((k, k, 1, dim), dtype=np.float32)
    tap = (k - 1) // 2
    kern[tap, tap, 0, :] = 1.0
    blk.dwconv.kernel.value = jnp.asarray(kern)
    if blk.dwconv.bias is not None:
        blk.dwconv.bias.value = jnp.zeros_like(blk.dwconv.bias.value)


def test_identity_kernel_mix_is_identity_unshifted():
    # With a delta kernel, shift->window->grid->conv->ungrid->unwindow->unshift
    # must be an EXACT identity: catches any permutation/reshape/padding bug.
    blk, p, N = make_block(shifted=False)
    _set_delta_kernel(blk, dim=8)
    x = jax.random.normal(jax.random.key(0), (2, N, 8))
    np.testing.assert_array_equal(np.asarray(blk._mix(x)), np.asarray(x))


STRATEGIES = ["nest_roll", "nest_grid_shift", "nest_grid_shift_exact", "ring_shift"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_cross_region_independence_shifted(strategy):
    # full sphere so every strategy produces a nontrivial region mask
    blk, p, N = make_block(shifted=True, base_pixels=tuple(range(12)),
                           shift_strategy=strategy)
    assert blk.validity is not None
    v = np.asarray(blk.validity[...])[:, :, 0].reshape(-1)   # (N,) in shifted coords
    foreign = np.where(v == 0.0)[0]
    dominant = np.where(v == 1.0)[0]
    if foreign.size == 0:
        pytest.skip("mask is trivial for this geometry")
    # map shifted positions -> original sequence positions via the shifter itself
    idx = np.asarray(blk.shifter.shift(
        jnp.arange(N, dtype=jnp.int32)[None, :, None]))[0, :, 0]

    x = jax.random.normal(jax.random.key(0), (1, N, 8))
    bump = 10.0 * jax.random.normal(jax.random.key(1), (foreign.size, 8))
    x_pert = x.at[0, idx[foreign], :].add(bump)

    y = np.asarray(blk._mix(x))
    y_pert = np.asarray(blk._mix(x_pert))
    # 1) foreign inputs are zeroed IN: dominant outputs are bit-identical
    np.testing.assert_array_equal(y[0, idx[dominant]], y_pert[0, idx[dominant]])
    # 2) foreign updates are zeroed OUT: _mix output is exactly 0 there
    #    (in the block, those pixels then pass through the residual; norm1 adds
    #    only a data-independent bias)
    np.testing.assert_array_equal(y[0, idx[foreign]], np.zeros((foreign.size, 8)))


def test_unshifted_block_has_no_validity_and_no_mask_cost():
    blk, _, _ = make_block(shifted=False)
    assert blk.validity is None
    from heal_swin_nnx.hp.shifting import NoShift
    assert isinstance(blk.shifter, NoShift)


def test_block_output_shape_and_finite():
    blk, p, N = make_block(shifted=True)
    x = jax.random.normal(jax.random.key(0), (2, N, 8))
    y = np.asarray(blk(x))
    assert y.shape == (2, N, 8)
    assert np.isfinite(y).all()


# --- model level -------------------------------------------------------------
from heal_swin_nnx import HealConv, HealConvEncoder


def tiny_conv(**over):
    p = tiny_conv_params(**over)
    return HealConv(p, rngs=nnx.Rngs(0)), p


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("base_pixels", [tuple(range(12)), (8, 9, 10, 11)])
def test_conv_forward_full_sphere_and_south_cap(base_pixels, strategy):
    model, p = tiny_conv(base_pixels=base_pixels, shift_strategy=strategy)
    model.eval()
    x = jax.random.normal(jax.random.key(0), (2, p.npix, 3))
    y = model(x)
    assert y.shape == (2, p.npix, 5)
    assert np.isfinite(np.asarray(y)).all()


def test_conv_jit_matches_eager():
    model, p = tiny_conv()
    model.eval()
    x = jax.random.normal(jax.random.key(0), (2, p.npix, 3))
    # tolerance covers float32 jit-fusion reduction-order drift; real bugs >> 1e-4
    np.testing.assert_allclose(np.asarray(nnx.jit(lambda m, x: m(x))(model, x)),
                               np.asarray(model(x)), rtol=1e-4, atol=1e-4)


def test_conv_batch_independence():
    model, p = tiny_conv()
    model.eval()
    x = jax.random.normal(jax.random.key(0), (3, p.npix, 3))
    full = np.asarray(model(x))
    single = np.asarray(model(x[1:2]))
    np.testing.assert_allclose(full[1:2], single, rtol=1e-4, atol=1e-4)


def test_conv_remat_matches_no_remat():
    m1, p = tiny_conv()
    m2, _ = tiny_conv(use_checkpoint=True)
    m1.eval(); m2.eval()
    x = jax.random.normal(jax.random.key(0), (1, p.npix, 3))
    np.testing.assert_allclose(np.asarray(m2(x)), np.asarray(m1(x)), rtol=1e-6, atol=1e-6)


def test_conv_encoder_standalone_no_decoder_params():
    p = tiny_conv_params()
    enc = HealConvEncoder(p, rngs=nnx.Rngs(0))
    tokens, skips = enc(jnp.ones((1, p.npix, 3)))
    assert tokens.shape == (1, p.npix // 4 // 4, 32)   # N/(patch*4^(L-1)), embed*2^(L-1)
    assert len(skips) == 2
    paths = [tuple(str(q) for q in path)
             for path, _ in nnx.to_flat_state(nnx.state(enc, nnx.Param))]
    assert not any("decoder" in q for path in paths for q in path)


def test_conv_no_buffer_is_a_param():
    model, _ = tiny_conv()
    params = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))
    for path in params:
        joined = "/".join(str(q) for q in path)
        for banned in ("attn_mask", "shift_idcs", "grid_perm", "inv_perm", "validity"):
            assert banned not in joined
    # and the depthwise kernels DO train
    assert any("dwconv" in "/".join(str(q) for q in path) for path in params)


def test_conv_params_are_json_loggable_next_to_model():
    import dataclasses, json
    _, p = tiny_conv()
    json.dumps(dataclasses.asdict(p))
