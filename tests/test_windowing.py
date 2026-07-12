import jax.numpy as jnp
import numpy as np

from heal_swin_nnx.hp_windowing import (
    get_nest_win_idcs, nest_relative_position_index, window_partition, window_reverse)


def test_window_roundtrip():
    x = jnp.arange(2 * 64 * 3, dtype=jnp.float32).reshape(2, 64, 3)
    w = window_partition(x, 4)
    assert w.shape == (2 * 16, 4, 3)
    assert np.array_equal(window_reverse(w, 4, 64), x)
