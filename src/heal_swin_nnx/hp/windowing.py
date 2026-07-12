"""1D windowing on the HEALPix nested pixel sequence.

Port of references/HEAL-SWIN/heal_swin/models_torch/hp_windowing.py.
Runtime functions use jnp; index helpers are numpy and run at construction.
"""
import math

import jax.numpy as jnp
import numpy as np


def window_partition(x, window_size):
    """(B, N, C) -> (num_windows*B, window_size, C). window_size: power of 2."""
    assert (math.log(window_size) / math.log(2)) % 1 == 0
    B, N, C = x.shape
    return x.reshape(B * (N // window_size), window_size, C)


def window_reverse(windows, window_size, N):
    """(num_windows*B, window_size, C) -> (B, N, C)."""
    assert (math.log(window_size) / math.log(2)) % 1 == 0
    B = windows.shape[0] // (N // window_size)
    return windows.reshape(B, N, windows.shape[-1])


def get_nest_win_idcs(window_size):
    """(sqrt(ws), sqrt(ws)) int64 grid holding the nested-scheme index of each
    Cartesian position inside one window."""
    s = int(round(window_size ** 0.5))
    assert s * s == window_size
    result = np.zeros((s, s), dtype=np.int64)

    def fill_quadrant(idx, x, y, size):
        if size == 2:
            result[x, y + 1] = idx
            result[x, y] = idx + 1
            result[x + 1, y + 1] = idx + 2
            result[x + 1, y] = idx + 3
        else:
            fill_quadrant(idx, x, y + size // 2, size // 2)
            fill_quadrant(idx + size ** 2 // 4, x, y, size // 2)
            fill_quadrant(idx + 2 * (size ** 2 // 4), x + size // 2, y + size // 2, size // 2)
            fill_quadrant(idx + 3 * (size ** 2 // 4), x + size // 2, y, size // 2)

    fill_quadrant(0, 0, 0, s)
    return result


def nest_relative_position_index(window_size):
    """(ws, ws) int64 relative-position index for HP window attention:
    the standard 2D SWIN index, re-ordered from Cartesian to nested scheme."""
    s = int(round(window_size ** 0.5))
    coords = np.stack(np.meshgrid(np.arange(s), np.arange(s), indexing="ij"))  # 2, s, s
    flat = coords.reshape(2, -1)
    rel = flat[:, :, None] - flat[:, None, :]  # 2, ws, ws
    rel = rel.transpose(1, 2, 0).astype(np.int64)
    rel[:, :, 0] += s - 1
    rel[:, :, 1] += s - 1
    rel[:, :, 0] *= 2 * s - 1
    idx = rel.sum(-1)
    inv = np.argsort(get_nest_win_idcs(window_size).reshape(-1))
    return idx[inv][:, inv]


def nest_win_coords(window_size):
    """(2, window_size) float32: Cartesian (x, y) of each nested-scheme index
    inside one window — the inverse of ``get_nest_win_idcs``. Used as RoPE
    token coordinates; same flat-grid approximation as the rel-pos bias."""
    grid = get_nest_win_idcs(window_size)
    s = grid.shape[0]
    coords = np.zeros((2, window_size), dtype=np.float32)
    xs, ys = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")
    coords[0, grid.reshape(-1)] = xs.reshape(-1).astype(np.float32)
    coords[1, grid.reshape(-1)] = ys.reshape(-1).astype(np.float32)
    return coords
