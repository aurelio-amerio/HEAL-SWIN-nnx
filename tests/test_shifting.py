import jax.numpy as jnp
import numpy as np
import pytest

from heal_swin_nnx import hp_shifting as hps
from tests.parity_utils import load_case


def test_nest_roll_mask_bit_exact():
    npz, _ = load_case("indices")
    for nside in (4, 8, 16):
        for ws in (4, 16):
            key = "nest_roll/mask/ns%d_ws%d" % (nside, ws)
            if key not in npz.files:
                continue
            got = hps.nest_roll_mask(8 * nside ** 2, ws, ws // 2)
            assert np.array_equal(got, npz[key]), key


def test_nest_roll_shift_roundtrip():
    sh = hps.NestRollShift(shift_size=2, input_resolution=64, window_size=4)
    x = jnp.arange(2 * 64 * 3, dtype=jnp.float32).reshape(2, 64, 3)
    assert np.array_equal(sh.shift_back(sh.shift(x)), x)
    assert np.array_equal(np.asarray(sh.shift(x)), np.roll(np.asarray(x), -2, axis=1))


def test_noshift_is_identity():
    sh = hps.NoShift()
    x = jnp.ones((1, 8, 2))
    assert sh.attn_mask is None
    assert np.array_equal(sh.shift(x), x) and np.array_equal(sh.shift_back(x), x)


def _grid_combos(npz):
    for nside in (4, 8, 16):
        for ws in (4, 16):
            if "nest_grid/idcs/ns%d_ws%d" % (nside, ws) in npz.files:
                yield nside, ws


def test_nest_grid_idcs_bit_exact():
    npz, _ = load_case("indices")
    for nside, ws in _grid_combos(npz):
        tag = "ns%d_ws%d" % (nside, ws)
        got = hps.nest_grid_shift_idcs(nside, 8, ws)
        assert np.array_equal(got, npz["nest_grid/idcs/%s" % tag]), tag
        assert np.array_equal(np.argsort(got), npz["nest_grid/back/%s" % tag]), tag


def test_nest_grid_masks_bit_exact():
    npz, _ = load_case("indices")
    for nside, ws in _grid_combos(npz):
        tag = "ns%d_ws%d" % (nside, ws)
        raw = hps.nest_grid_mask(nside, 8, ws)
        assert np.array_equal(raw, npz["nest_grid/mask_raw/%s" % tag]), tag
        attn = hps.get_attn_mask_from_mask(raw, ws)
        assert np.array_equal(attn, npz["nest_grid/attn_mask/%s" % tag]), tag


def test_nest_grid_module_and_unsupported_base_pix():
    sh = hps.NestGridShift(nside=16, base_pix=8, window_size=4)
    x = jnp.arange(1 * 2048 * 2, dtype=jnp.float32).reshape(1, 2048, 2)
    assert np.array_equal(sh.shift_back(sh.shift(x)), x)
    with pytest.raises(NotImplementedError):
        hps.NestGridShift(nside=16, base_pix=12, window_size=4)


def test_ring_idcs_and_masks_bit_exact():
    npz, _ = load_case("indices")
    for nside in (4, 8, 16):
        for ws in (4, 16):
            tag = "ns%d_ws%d" % (nside, ws)
            if "ring/idcs/%s" % tag not in npz.files:
                continue
            idcs, raw = hps.ring_shift_idcs_and_mask(nside, 8, ws, ws // 2)
            assert np.array_equal(idcs, npz["ring/idcs/%s" % tag]), tag
            assert np.array_equal(np.argsort(idcs), npz["ring/back/%s" % tag]), tag
            assert np.array_equal(raw, npz["ring/mask_raw/%s" % tag]), tag
            attn = hps.get_attn_mask_from_mask(raw, ws)
            assert np.array_equal(attn, npz["ring/attn_mask/%s" % tag]), tag


def test_ring_module_and_unsupported_base_pix():
    sh = hps.RingShift(nside=16, base_pix=8, window_size=4, shift_size=2)
    x = jnp.arange(1 * 2048 * 2, dtype=jnp.float32).reshape(1, 2048, 2)
    assert np.array_equal(sh.shift_back(sh.shift(x)), x)
    with pytest.raises(NotImplementedError):
        hps.RingShift(nside=16, base_pix=12, window_size=4, shift_size=2)
