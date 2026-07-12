import healpy as hp
import numpy as np
import pytest

from heal_swin_nnx import hp_topology as hpt


@pytest.mark.parametrize("nside", [1, 2, 4, 8, 16])
def test_pix2xyf_matches_healpy(nside):
    pix = np.arange(12 * nside ** 2)
    x, y, f = hpt.pix2xyf(nside, pix)
    hx, hy, hf = hp.pix2xyf(nside, pix, nest=True)
    assert np.array_equal(x, hx) and np.array_equal(y, hy) and np.array_equal(f, hf)


@pytest.mark.parametrize("nside", [1, 2, 4, 8, 16])
def test_xyf2pix_roundtrip(nside):
    pix = np.arange(12 * nside ** 2)
    assert np.array_equal(hpt.xyf2pix(nside, *hpt.pix2xyf(nside, pix)), pix)


def test_local_global_roundtrip():
    base_pixels, nside = [8, 9, 10, 11], 4
    loc = np.arange(len(base_pixels) * nside ** 2)
    glob = hpt.local_to_global(base_pixels, nside, loc)
    assert glob[0] == 8 * nside ** 2 and glob[-1] == 12 * nside ** 2 - 1
    assert np.array_equal(hpt.global_to_local(base_pixels, nside, glob), loc)
    assert hpt.global_to_local(base_pixels, nside, np.array([0])) == -1


def test_topology_matrices_shape_and_symmetry():
    for m in (hpt.neighbours_matrix, hpt.side_matrix,
              hpt.corner_faces_matrix, hpt.corner_sides_matrix):
        assert m.shape == (12, 4)
    # edge adjacency is symmetric: g in neighbours(f) <-> f in neighbours(g)
    for f in range(12):
        for g in hpt.neighbours_matrix[f]:
            assert f in hpt.neighbours_matrix[g]
    # 8 pinch corners, each incident to 3 faces -> 24 blocked corner entries,
    # aligned between the two corner matrices
    assert (hpt.corner_faces_matrix == -1).sum() == 24
    assert np.array_equal(hpt.corner_sides_matrix == -1, hpt.corner_faces_matrix == -1)


@pytest.mark.parametrize("nside", [2, 4, 8])
def test_grid_neighbours_match_healpy_exhaustively(nside):
    """Every pixel's neighbour set must equal healpy's (edges, corners,
    orientation, and pinch points all validated in one sweep)."""
    for pix in range(12 * nside ** 2):
        ours = hpt.grid_neighbours(nside, pix)
        theirs = set(int(v) for v in hp.get_all_neighbours(nside, pix, nest=True) if v >= 0)
        assert ours == theirs, "pixel %d: %r != %r" % (pix, sorted(ours), sorted(theirs))


def test_derive_offset_tables_reproduces_reference_8pix():
    off1, off2 = hpt.derive_offset_tables(list(range(8)))
    assert off1 == {0: 2, 1: 2, 2: 2, 3: 6, 4: 3, 5: 3, 6: 3, 7: 3}
    assert off2 == {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3}


def test_derive_offset_tables_full_sphere():
    off1, off2 = hpt.derive_offset_tables(list(range(12)))
    assert off1 == {0: 6, 1: 6, 2: 6, 3: 10, 4: 3, 5: 3, 6: 3, 7: 3,
                    8: 8, 9: 0, 10: 0, 11: 0}
    assert off2 == {0: 7, 1: 7, 2: 7, 3: 7, 4: 3, 5: 3, 6: 3, 7: 3,
                    8: 10, 9: 10, 10: 10, 11: 2}


def test_derive_offset_tables_south_cap():
    off1, off2 = hpt.derive_offset_tables([8, 9, 10, 11])
    assert off1 == {0: 0, 1: 0, 2: 0, 3: 0}
    assert off2 == {0: 2, 1: 2, 2: 2, 3: 2}
