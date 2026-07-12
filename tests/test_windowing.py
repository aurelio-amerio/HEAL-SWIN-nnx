import jax.numpy as jnp
import numpy as np

from heal_swin_nnx.hp_windowing import (
    get_nest_win_idcs, nest_relative_position_index, window_partition, window_reverse)
from tests.parity_utils import load_case


def test_nest_win_idcs_bit_exact():
    npz, _ = load_case("indices")
    for ws in (4, 16):
        assert np.array_equal(get_nest_win_idcs(ws), npz["nest_win_idcs/ws%d" % ws])


def test_nest_relative_position_index_bit_exact():
    npz, _ = load_case("indices")
    for ws in (4, 16):
        assert np.array_equal(nest_relative_position_index(ws), npz["hp_rel_pos_index/ws%d" % ws])


def test_window_roundtrip():
    x = jnp.arange(2 * 64 * 3, dtype=jnp.float32).reshape(2, 64, 3)
    w = window_partition(x, 4)
    assert w.shape == (2 * 16, 4, 3)
    assert np.array_equal(window_reverse(w, 4, 64), x)
