"""The central invariant of the full-sphere extension (spec 1.1 and 6.1):
pixels adjacent in a window's canonical 2D layout must be adjacent on the sky.

Unit-offset checks (horizontal, vertical, diagonal) against healpy pin the
window's local isometry; consistency of larger offsets follows by composition
along grid paths, so unit steps are the complete geometric claim."""
import healpy as hp
import numpy as np
import pytest

from heal_swin_nnx.hp import topology as hpt
from heal_swin_nnx.hp.windowing import get_nest_win_idcs

# expect_cross_face: whether any unmasked cross-face slot pair can survive.
# True for full-sphere configs (north->equatorial and equatorial->south seams are
# pure translations, so cross-seam content survives unmasked). False for the south
# cap alone: all its seams are south-south (90-degree-rotated frames), every
# cross-seam source is already claimed in-face, so all cross-seam dests are
# backfilled and masked apart (see hp_topology.exact_shift_idcs_and_mask docstring).
CONFIGS = [
    (list(range(12)), 8, 4, True),
    (list(range(12)), 8, 16, True),
    ([8, 9, 10, 11], 8, 4, False),
]


def _healpy_neighbours(nside, p):
    return set(int(v) for v in hp.get_all_neighbours(nside, p, nest=True) if v >= 0)


@pytest.mark.parametrize("base_pixels,nside,ws,expect_cross_face", CONFIGS)
def test_exact_shift_seam_correctness(base_pixels, nside, ws, expect_cross_face):
    idcs, raw = hpt.exact_shift_idcs_and_mask(base_pixels, nside, ws)
    grid = get_nest_win_idcs(ws)
    s = grid.shape[0]
    n_windows = len(idcs) // ws
    checked_cross_face = 0
    for w in range(n_windows):
        win_src = hpt.local_to_global(base_pixels, nside, idcs[w * ws:(w + 1) * ws])
        win_lbl = raw[w * ws:(w + 1) * ws]
        for gx in range(s):
            for gy in range(s):
                a = int(grid[gx, gy])
                for nx, ny in ((gx + 1, gy), (gx, gy + 1), (gx + 1, gy + 1)):
                    if nx >= s or ny >= s:
                        continue
                    b = int(grid[nx, ny])
                    if win_lbl[a] != win_lbl[b]:
                        continue  # masked apart (backfilled/pinch) — exempt
                    p, q = int(win_src[a]), int(win_src[b])
                    assert q in _healpy_neighbours(nside, p), (
                        "window %d slots %d-%d: pixels %d and %d are laid out "
                        "adjacent but are not sky-adjacent" % (w, a, b, p, q))
                    if p // nside ** 2 != q // nside ** 2:
                        checked_cross_face += 1
    if expect_cross_face:
        assert checked_cross_face > 0, "no cross-face pair was exercised — config too small"
    else:
        assert checked_cross_face == 0, (
            "south-cap subsets have only rotated seams; a surviving unmasked "
            "cross-face pair means the collision handling regressed")


@pytest.mark.parametrize("base_pixels,nside,ws,expect_cross_face", CONFIGS)
def test_unmasked_windows_are_single_component(base_pixels, nside, ws, expect_cross_face):
    """Every window without mask labels holds one geographically contiguous patch."""
    idcs, raw = hpt.exact_shift_idcs_and_mask(base_pixels, nside, ws)
    for w in range(len(idcs) // ws):
        win_lbl = raw[w * ws:(w + 1) * ws]
        if win_lbl.any():
            continue
        comp = hpt._window_components(base_pixels, nside, ws, idcs[w * ws:(w + 1) * ws],
                                      np.zeros(ws, dtype=bool))
        assert comp.max() == 0, "window %d splits into %d components" % (w, int(comp.max()) + 1)
