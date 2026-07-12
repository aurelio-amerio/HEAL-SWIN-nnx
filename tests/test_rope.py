import jax
import jax.numpy as jnp
import numpy as np
import pytest

from heal_swin_nnx.hp.windowing import get_nest_win_idcs, nest_win_coords
from heal_swin_nnx.layers import apply_rope, init_rope_freqs, rope_rotation_table


@pytest.mark.parametrize("window_size", [4, 16, 64])
def test_nest_win_coords_roundtrips_grid(window_size):
    # derive-then-verify: coords must invert the independently tested nested->
    # Cartesian window map (get_nest_win_idcs is pinned to healpy sky adjacency
    # by test_seam_geometry.py)
    grid = get_nest_win_idcs(window_size)
    coords = nest_win_coords(window_size)
    assert coords.shape == (2, window_size) and coords.dtype == np.float32
    for i in range(window_size):
        x, y = int(coords[0, i]), int(coords[1, i])
        assert grid[x, y] == i


def test_init_rope_freqs_shapes_and_axial_alignment():
    axial = init_rope_freqs(16, 3, 10.0)
    assert axial.shape == (2, 3, 8)
    # axial: first D/4 pairs are pure-x (fy == 0), second D/4 pure-y (fx == 0)
    np.testing.assert_allclose(axial[1, :, :4], 0.0, atol=1e-7)
    np.testing.assert_allclose(axial[0, :, 4:], 0.0, atol=1e-7)
    mixed = init_rope_freqs(16, 3, 10.0, key=jax.random.key(0))
    assert mixed.shape == (2, 3, 8)
    # mixed init: heads got distinct random rotations
    assert not np.allclose(mixed[:, 0], mixed[:, 1])


def test_apply_rope_preserves_norms():
    q = jax.random.normal(jax.random.key(0), (2, 3, 8, 16))
    k = jax.random.normal(jax.random.key(1), (2, 3, 8, 16))
    freqs = init_rope_freqs(16, 3, 10.0, key=jax.random.key(2))
    t = jnp.arange(8, dtype=jnp.float32)
    table = rope_rotation_table(freqs, t, t[::-1])
    q2, k2 = apply_rope(q, k, table)
    assert q2.shape == q.shape and k2.shape == k.shape
    np.testing.assert_allclose(np.linalg.norm(np.asarray(q2), axis=-1),
                               np.linalg.norm(np.asarray(q), axis=-1), rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(k2), axis=-1),
                               np.linalg.norm(np.asarray(k), axis=-1), rtol=1e-5)


def test_rope_logits_depend_only_on_coordinate_offset():
    # (R(ti)q)^T (R(tj)k) must be invariant under a global coordinate shift —
    # holds for any freqs, so test with the mixed (random-rotation) init
    freqs = init_rope_freqs(16, 2, 10.0, key=jax.random.key(0))
    t_x = jnp.array([0.0, 1.0, 2.0, 5.0])
    t_y = jnp.array([3.0, 0.0, 2.0, 1.0])
    q = jax.random.normal(jax.random.key(1), (1, 2, 4, 16))
    k = jax.random.normal(jax.random.key(2), (1, 2, 4, 16))

    def logits(dx, dy):
        table = rope_rotation_table(freqs, t_x + dx, t_y + dy)
        q2, k2 = apply_rope(q, k, table)
        return np.asarray(q2 @ k2.swapaxes(-2, -1))

    np.testing.assert_allclose(logits(0.0, 0.0), logits(3.0, 7.0),
                               rtol=1e-4, atol=1e-5)


def test_healswin_forward_all_pos_embeds():
    from flax import nnx
    from heal_swin_nnx import HealSwin, HealSwinParams
    for pos_embed in ["none", "rel_bias", "rope_axial", "rope_mixed"]:
        p = HealSwinParams(nside=16, in_channels=2, out_channels=3,
                           base_pixels=(8, 9, 10, 11), embed_dim=16,
                           depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0,
                           pos_embed=pos_embed)
        model = HealSwin(p, rngs=nnx.Rngs(0))
        model.eval()
        y = model(jnp.ones((1, p.npix, 2)))
        assert y.shape == (1, p.npix, 3), pos_embed
        assert np.isfinite(np.asarray(y)).all(), pos_embed


def test_swinunet_forward_all_pos_embeds():
    from flax import nnx
    from heal_swin_nnx import SwinParams, SwinUnet
    for pos_embed in ["none", "rel_bias", "rope_axial", "rope_mixed"]:
        p = SwinParams(img_size=(32, 64), in_channels=2, out_channels=3,
                       embed_dim=16, depths=(2, 2), num_heads=(2, 4),
                       drop_path_rate=0.0, pos_embed=pos_embed)
        model = SwinUnet(p, rngs=nnx.Rngs(0))
        model.eval()
        y = model(jnp.ones((1, 32, 64, 2)))
        assert y.shape == (1, 32, 64, 3), pos_embed
        assert np.isfinite(np.asarray(y)).all(), pos_embed
