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
