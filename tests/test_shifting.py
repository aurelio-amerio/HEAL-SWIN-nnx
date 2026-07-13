import jax.numpy as jnp
import numpy as np
import pytest

from heal_swin_nnx.hp import shifting as hps
from heal_swin_nnx.hp import topology as hpt


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


def test_nest_grid_module_roundtrip_8pix():
    sh = hps.NestGridShift(nside=16, base_pixels=list(range(8)), window_size=4)
    x = jnp.arange(1 * 2048 * 2, dtype=jnp.float32).reshape(1, 2048, 2)
    assert np.array_equal(sh.shift_back(sh.shift(x)), x)


def test_nest_grid_idcs_valid_permutation_matrix():
    # Brief specified [0, 4, 8] as the fourth case; substituted with [0, 1, 2, 3, 8, 9,
    # 10, 11] (north + south caps, no equatorial band) — see task-5-report.md for the
    # verified proof that [0, 4, 8] is *not* a valid permutation under the current
    # hp.topology.derive_offset_tables fallback formula (a mixed real-edge/fallback
    # collision, not an artifact of this task's changes).
    for base_pixels in (list(range(12)), list(range(8)), [8, 9, 10, 11],
                        [0, 1, 2, 3, 8, 9, 10, 11]):
        for nside in (8, 16):
            for ws in (4, 16):
                # nest_grid_shift_idcs asserts permutation validity internally
                idcs = hps.nest_grid_shift_idcs(nside, base_pixels, ws)
                assert idcs.shape == (len(base_pixels) * nside ** 2,)


def test_ring_module_roundtrip_full_sphere_and_subsets():
    for base_pixels in (list(range(12)), list(range(8)), [8, 9, 10, 11], [0, 4, 8]):
        npix = len(base_pixels) * 16 ** 2
        sh = hps.RingShift(nside=16, base_pixels=base_pixels, window_size=4, shift_size=2)
        x = jnp.arange(1 * npix * 2, dtype=jnp.float32).reshape(1, npix, 2)
        assert np.array_equal(sh.shift_back(sh.shift(x)), x), base_pixels


def test_nest_grid_module_roundtrip_full_sphere_and_subsets():
    # Brief specified [0, 4, 8] as a subset case; substituted with [0, 1, 2, 3, 8, 9,
    # 10, 11] per controller adjudication — [0, 4, 8] is unsatisfiable for the
    # nest-grid strategy (self-map fallback for local face 0 collides with face 4's
    # genuine pull from face 0), consistent with the exclusion already made in
    # test_nest_grid_idcs_valid_permutation_matrix (Task 5).
    for base_pixels in (list(range(12)), [8, 9, 10, 11], [0, 1, 2, 3, 8, 9, 10, 11]):
        npix = len(base_pixels) * 16 ** 2
        sh = hps.NestGridShift(nside=16, base_pixels=base_pixels, window_size=4)
        x = jnp.arange(1 * npix * 2, dtype=jnp.float32).reshape(1, npix, 2)
        assert np.array_equal(sh.shift_back(sh.shift(x)), x), base_pixels


def _slot_grid(ws):
    from heal_swin_nnx.hp.windowing import get_nest_win_idcs
    return get_nest_win_idcs(ws)


def test_nest_grid_mask_ground_truth():
    """Unmasked canonically-adjacent slot pairs must be sky-adjacent (spec 6.2)."""
    import healpy as hp
    for base_pixels in (list(range(12)), [8, 9, 10, 11], [0, 1, 2, 3],
                        [0, 1, 2, 3, 8, 9, 10, 11]):
        nside, ws = 8, 4
        idcs = hps.nest_grid_shift_idcs(nside, base_pixels, ws)
        raw = hps.nest_grid_mask(nside, base_pixels, ws)
        grid = _slot_grid(ws)
        s = grid.shape[0]
        for w in range(len(idcs) // ws):
            win_src = hpt.local_to_global(base_pixels, nside, idcs[w * ws:(w + 1) * ws])
            win_lbl = raw[w * ws:(w + 1) * ws]
            for gx in range(s):
                for gy in range(s):
                    a = grid[gx, gy]
                    for nx, ny in ((gx + 1, gy), (gx, gy + 1)):
                        if nx >= s or ny >= s:
                            continue
                        b = grid[nx, ny]
                        if win_lbl[a] != win_lbl[b]:
                            continue  # masked apart — no geometric claim
                        p, q = int(win_src[a]), int(win_src[b])
                        neigh = set(int(v) for v in
                                    hp.get_all_neighbours(nside, p, nest=True) if v >= 0)
                        assert q in neigh, "window %d slots %d-%d: %d !~ %d" % (w, a, b, p, q)


def test_ring_full_sphere_is_pure_permutation_no_mask():
    idcs, raw = hps.ring_shift_idcs_and_mask(8, list(range(12)), 4, 2)
    assert np.array_equal(np.sort(idcs), np.arange(12 * 8 ** 2))
    assert not raw.any()


def test_ring_subset_masks_out_of_domain_sources():
    import healpy as hp
    nside, base_pixels = 8, [8, 9, 10, 11]
    idcs, raw = hps.ring_shift_idcs_and_mask(nside, base_pixels, 4, 2)
    # every masked pixel is one whose true ring-shift source lies outside the subset
    ring = np.arange(12 * nside ** 2)
    full = hp.pixelfunc.ring2nest(nside, np.roll(ring, 2))[
        hp.pixelfunc.nest2ring(nside, np.arange(12 * nside ** 2))]
    sel = hpt.local_to_global(base_pixels, nside, np.arange(len(idcs)))
    out_of_domain = hpt.global_to_local(base_pixels, nside, full[sel]) < 0
    assert np.array_equal(raw > 0, out_of_domain)


def test_exact_shift_valid_permutation_and_backfill_accounting():
    # The plan's original expectation (south cap closes on itself -> zero holes,
    # zero mask) was geometrically wrong: south-south seam crossings swap x and y
    # (a 90-degree frame rotation, side_matrix rows 8-11), so the uniform (-d, -d)
    # walk collides along those seams and near the pole (verified against healpy).
    # Count-free invariants instead; per-pair geometric correctness is arbitrated
    # by test_exact_shift_colabelled_pairs_sky_adjacent below.
    for base_pixels in (list(range(12)), [8, 9, 10, 11], list(range(8))):
        nside, ws = 8, 4
        d = int(round(ws ** 0.5)) // 2
        idcs, raw = hpt.exact_shift_idcs_and_mask(base_pixels, nside, ws)
        npix = len(base_pixels) * nside ** 2
        assert np.array_equal(np.sort(idcs), np.arange(npix)), base_pixels
        if base_pixels == list(range(12)):
            # pinch floor: the four south-pointing pinch corners are backfilled
            # and masked; rotated-seam masking legitimately adds more windows
            n_masked_windows = len(np.unique(np.nonzero(raw)[0] // ws))
            assert n_masked_windows >= 4
            assert (raw > 0).sum() >= 4 * d * d
        if base_pixels == [8, 9, 10, 11]:
            # rotated south-south seams force masking even on the closed cap
            assert (raw > 0).any()


def test_exact_shift_colabelled_pairs_sky_adjacent():
    """Equal-label canonically-adjacent slot pairs must hold sky-adjacent sources
    (spec 6.1, Task 9's headline invariant pulled forward as this task's gate)."""
    import healpy as hp
    for base_pixels in (list(range(12)), [8, 9, 10, 11]):
        nside, ws = 8, 4
        idcs, raw = hpt.exact_shift_idcs_and_mask(base_pixels, nside, ws)
        grid = _slot_grid(ws)
        s = grid.shape[0]
        for w in range(len(idcs) // ws):
            win_src = hpt.local_to_global(base_pixels, nside, idcs[w * ws:(w + 1) * ws])
            win_lbl = raw[w * ws:(w + 1) * ws]
            for gx in range(s):
                for gy in range(s):
                    a = grid[gx, gy]
                    for nx, ny in ((gx + 1, gy), (gx, gy + 1)):
                        if nx >= s or ny >= s:
                            continue
                        b = grid[nx, ny]
                        if win_lbl[a] != win_lbl[b]:
                            continue  # masked apart — no geometric claim
                        p, q = int(win_src[a]), int(win_src[b])
                        neigh = set(int(v) for v in
                                    hp.get_all_neighbours(nside, p, nest=True) if v >= 0)
                        assert q in neigh, "window %d slots %d-%d: %d !~ %d" % (w, a, b, p, q)


def test_exact_matches_approximate_on_face_interiors():
    for base_pixels in (list(range(12)), list(range(8)), [8, 9, 10, 11]):
        nside, ws = 8, 4
        d = int(round(ws ** 0.5)) // 2
        approx = hps.nest_grid_shift_idcs(nside, base_pixels, ws)
        exact, _ = hpt.exact_shift_idcs_and_mask(base_pixels, nside, ws)
        npix = len(base_pixels) * nside ** 2
        x, y, _f = hpt.pix2xyf(nside, hpt.local_to_global(base_pixels, nside, np.arange(npix)))
        interior = (x >= d) & (y >= d)
        assert np.array_equal(exact[interior], approx[interior]), base_pixels


def test_exact_shift_module_roundtrip():
    for base_pixels in (list(range(12)), [8, 9, 10, 11]):
        npix = len(base_pixels) * 16 ** 2
        sh = hps.NestGridShiftExact(nside=16, base_pixels=base_pixels, window_size=4)
        x = jnp.arange(1 * npix * 2, dtype=jnp.float32).reshape(1, npix, 2)
        assert np.array_equal(sh.shift_back(sh.shift(x)), x), base_pixels


# --- raw masks + majority-region validity (HealConv support) ---------------


def test_nest_roll_raw_mask_matches_previous_inline_construction():
    input_resolution, ws, ss = 256, 16, 8
    img_mask = np.zeros(input_resolution, dtype=np.float32)
    for cnt, s in enumerate((slice(0, -ws), slice(-ws, -ss), slice(-ss, None))):
        img_mask[s] = cnt
    np.testing.assert_array_equal(
        hps.nest_roll_raw_mask(input_resolution, ws, ss), img_mask)
    # the public pairwise mask must be exactly the wrapper composition
    np.testing.assert_array_equal(
        hps.nest_roll_mask(input_resolution, ws, ss),
        hps.get_attn_mask_from_mask(
            hps.nest_roll_raw_mask(input_resolution, ws, ss), ws))


def test_validity_from_mask_majority_and_deterministic_ties():
    # ws=4: window 0 has majority label 0; window 1 is a 2-2 tie -> lowest label (2) wins
    raw = np.array([0, 0, 0, 1, 3, 3, 2, 2], dtype=np.float32)
    v = hps.validity_from_mask(raw, 4)
    np.testing.assert_array_equal(v, np.array([[1, 1, 1, 0], [0, 0, 1, 1]], dtype=np.float32))
    assert v.dtype == np.float32


def _raw_mask_for(strategy, nside, base_pixels, ws, ss):
    if strategy == "nest_roll":
        return hps.nest_roll_raw_mask(len(base_pixels) * nside ** 2, ws, ss)
    if strategy == "nest_grid_shift":
        return hps.nest_grid_mask(nside, list(base_pixels), ws)
    if strategy == "nest_grid_shift_exact":
        return hpt.exact_shift_idcs_and_mask(list(base_pixels), nside, ws)[1]
    return hps.ring_shift_idcs_and_mask(nside, list(base_pixels), ws, ss)[1]


@pytest.mark.parametrize("strategy", ["nest_roll", "nest_grid_shift",
                                      "nest_grid_shift_exact", "ring_shift"])
def test_validity_consistent_with_attn_mask(strategy):
    # a pixel is invalid (0) iff the pairwise attention mask forbids it (-100)
    # against its window's majority pixels
    nside, base_pixels, ws, ss = 8, tuple(range(12)), 16, 8
    raw = _raw_mask_for(strategy, nside, base_pixels, ws, ss)
    attn = hps.get_attn_mask_from_mask(raw, ws)
    v = hps.validity_from_mask(raw, ws)
    assert v.shape == attn.shape[:2]
    for w in range(v.shape[0]):
        maj = int(np.argmax(v[w]))            # some majority pixel (v==1 exists by construction)
        assert v[w, maj] == 1.0
        for p in range(ws):
            assert (v[w, p] == 0.0) == (attn[w, p, maj] == -100.0)
