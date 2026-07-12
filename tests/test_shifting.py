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
