"""HEALPix base-pixel (face) topology and derivation functions.

Face adjacency/orientation matrices are copied from Map2Patches (A. Amerio,
references/Map2Patches) with permission; there is no runtime dependency on it.

This module must never import healpy: healpy is the *independent* ground truth
used by the test suite, so deriving and verifying with the same library would
be circular. Everything here is construction-time numpy; per-pixel Python
loops are acceptable (only the model forward path is performance-critical).

Face-local coordinate conventions (identical to healpy's pix2xyf):
  x, y in [0, nside); the in-face NEST index is the Morton (Z-order)
  interleave with x on even bits, y on odd bits. Exit directions map to
  neighbour columns [TL, BL, BR, TR] as:
    +x (x >= nside) -> column 0 (TL);  -y (y < 0) -> column 1 (BL)
    -x (x < 0)      -> column 2 (BR);  +y (y >= nside) -> column 3 (TR)
  Corner column i sits between edge columns i and (i+1) % 4.
"""
import numpy as np

# Row f = corner-neighbour faces of face f in [top-left, bottom-left,
# bottom-right, top-right] order (anticlockwise from upper left).
neighbours_matrix = np.array([
    [1, 5, 4, 3], [2, 6, 5, 0], [3, 7, 6, 1], [0, 4, 7, 2],      # north polar 0-3
    [0, 8, 11, 3], [1, 9, 8, 0], [2, 10, 9, 1], [3, 11, 10, 2],  # equatorial  4-7
    [5, 9, 11, 4], [6, 10, 8, 5], [7, 11, 9, 6], [4, 8, 10, 7],  # south polar 8-11
], dtype=np.int64)

# side_matrix[f][i]: which edge of neighbours_matrix[f][i] touches face f.
# Edge labels: 0 = its x = nside-1 edge, 1 = its y = 0 edge,
#              2 = its x = 0 edge,       3 = its y = nside-1 edge.
side_matrix = np.array([
    [3, 3, 0, 0], [3, 3, 0, 0], [3, 3, 0, 0], [3, 3, 0, 0],
    [2, 3, 0, 1], [2, 3, 0, 1], [2, 3, 0, 1], [2, 3, 0, 1],
    [2, 2, 1, 1], [2, 2, 1, 1], [2, 2, 1, 1], [2, 2, 1, 1],
], dtype=np.int64)

# corner_faces_matrix[f][i]: face diagonally across corner i (between edge
# columns i and i+1 mod 4); -1 at the 8 pinch points where only 3 faces meet.
corner_faces_matrix = np.array([
    [-1, 8, -1, 2], [-1, 9, -1, 3], [-1, 10, -1, 0], [-1, 11, -1, 1],
    [5, -1, 7, -1], [6, -1, 4, -1], [7, -1, 5, -1], [4, -1, 6, -1],
    [-1, 10, -1, 0], [-1, 11, -1, 1], [-1, 8, -1, 2], [-1, 9, -1, 3],
], dtype=np.int64)

# corner_sides_matrix[f][i]: which corner of the diagonal face touches ours.
# 0 = its (nside-1, nside-1), 1 = its (0, 0), 2 = its (nside-1, 0),
# 3 = its (0, nside-1); -1 where corner_faces_matrix is -1.
corner_sides_matrix = np.array([
    [-1, 0, -1, 0], [-1, 0, -1, 0], [-1, 0, -1, 0], [-1, 0, -1, 0],
    [3, -1, 2, -1], [3, -1, 2, -1], [3, -1, 2, -1], [3, -1, 2, -1],
    [-1, 1, -1, 1], [-1, 1, -1, 1], [-1, 1, -1, 1], [-1, 1, -1, 1],
], dtype=np.int64)


def pix2xyf(nside, pix):
    """Global NEST pixel index -> (x, y, face). Matches healpy pix2xyf(nest=True)."""
    pix = np.asarray(pix, dtype=np.int64)
    face, p = np.divmod(pix, nside * nside)
    x = np.zeros_like(p)
    y = np.zeros_like(p)
    for i in range(int(nside).bit_length() - 1):
        x |= ((p >> (2 * i)) & 1) << i
        y |= ((p >> (2 * i + 1)) & 1) << i
    return x, y, face


def xyf2pix(nside, x, y, face):
    """(x, y, face) -> global NEST pixel index. Inverse of pix2xyf."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    p = np.zeros_like(x)
    for i in range(int(nside).bit_length() - 1):
        p |= ((x >> i) & 1) << (2 * i)
        p |= ((y >> i) & 1) << (2 * i + 1)
    return np.asarray(face, dtype=np.int64) * nside * nside + p


def local_to_global(base_pixels, nside, idx):
    """Model-layout pixel index (concatenated selected faces) -> global NEST index."""
    face_len = nside * nside
    i, p = np.divmod(np.asarray(idx, dtype=np.int64), face_len)
    return np.asarray(base_pixels, dtype=np.int64)[i] * face_len + p


def global_to_local(base_pixels, nside, idx):
    """Global NEST index -> model-layout index; -1 where the face is not selected."""
    face_len = nside * nside
    lut = np.full(12, -1, dtype=np.int64)
    lut[np.asarray(base_pixels, dtype=np.int64)] = np.arange(len(base_pixels))
    f, p = np.divmod(np.asarray(idx, dtype=np.int64), face_len)
    lf = lut[f]
    return np.where(lf < 0, np.int64(-1), lf * face_len + p)


def _enter(nside, side, depth, run):
    """Coords inside the neighbour entered through `side` at `depth` past the
    seam, `run` along it. Along-edge orientation is identity (Map2Patches
    convention, validated in production padding); the exhaustive healpy
    neighbour test arbitrates — a failing seam would be fixed here with a
    per-(face, column) run reversal."""
    if side == 0:
        return nside - 1 - depth, run
    if side == 1:
        return run, depth
    if side == 2:
        return depth, run
    return run, nside - 1 - depth  # side == 3


def _cross_x(nside, x, y, face):
    col, depth = (0, x - nside) if x >= nside else (2, -1 - x)
    g = int(neighbours_matrix[face][col])
    xr, yr = _enter(nside, int(side_matrix[face][col]), depth, y)
    return xr, yr, g


def _cross_y(nside, x, y, face):
    col, depth = (3, y - nside) if y >= nside else (1, -1 - y)
    g = int(neighbours_matrix[face][col])
    xr, yr = _enter(nside, int(side_matrix[face][col]), depth, x)
    return xr, yr, g


def walk(nside, x, y, face):
    """Resolve possibly out-of-range face-local coords to on-sphere (x, y, face).

    Returns None at a pinch corner (no diagonal face there). Supports at most
    one crossing per axis (|overflow| < nside), which every caller satisfies.
    Corner crossings are composed from two edge crossings; both orders must
    agree, which fails exactly at pinch corners (checked via the corner matrix
    first, asserted for regular corners).
    """
    x, y, face = int(x), int(y), int(face)
    x_in, y_in = 0 <= x < nside, 0 <= y < nside
    if x_in and y_in:
        return x, y, face
    if not x_in and not y_in:
        col = {(True, True): 3, (True, False): 0,
               (False, False): 1, (False, True): 2}[(x >= nside, y >= nside)]
        if corner_faces_matrix[face][col] == -1:
            return None
        r1 = walk(nside, *_cross_x(nside, x, y, face))
        r2 = walk(nside, *_cross_y(nside, x, y, face))
        assert r1 == r2, "corner walk inconsistent at face %d col %d" % (face, col)
        assert r1[2] == corner_faces_matrix[face][col]
        return r1
    if not x_in:
        return walk(nside, *_cross_x(nside, x, y, face))
    return walk(nside, *_cross_y(nside, x, y, face))


def grid_neighbours(nside, pix):
    """The <=8 sky-adjacent pixels of a global NEST pixel (pinch diagonals absent)."""
    x, y, f = pix2xyf(nside, int(pix))
    out = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            r = walk(nside, int(x) + dx, int(y) + dy, int(f))
            if r is not None:
                out.add(int(xyf2pix(nside, r[0], r[1], r[2])))
    return out


# Which neighbour column is the dir1/dir2 shift source for each face; the
# south polar ring is reflected relative to the north, hence the swap.
DIR1_COL = [1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2]
DIR2_COL = [2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1]


def _window_slot_grid_ok(base_pixels, nside, window_size, local_sources):
    """True iff every canonically-adjacent slot pair in this window, other than
    pairs touching a flat slot in [0, window_size // 4), holds sky-adjacent
    source pixels (the seam-correctness criterion). The first quarter-window
    slots are exempt because they are exactly the positions the carry-over
    write of nest_grid_mask can re-label: a bad glue there is repaired by the
    carry label (which separates those slots from the rest of the window in
    the attention mask), so it does not force masking the whole face."""
    from heal_swin_nnx.hp.windowing import get_nest_win_idcs
    grid = get_nest_win_idcs(window_size)
    s = grid.shape[0]
    qws = window_size // 4
    local_sources = np.asarray(local_sources)
    glob = local_to_global(base_pixels, nside, local_sources)
    for gx in range(s):
        for gy in range(s):
            a = grid[gx, gy]
            for nx, ny in ((gx + 1, gy), (gx, gy + 1)):
                if nx < s and ny < s:
                    b = grid[nx, ny]
                    if a < qws or b < qws:
                        continue
                    if int(glob[b]) not in grid_neighbours(nside, int(glob[a])):
                        return False
    return True


def derive_mask_faces(base_pixels, nside, window_size, shift_idcs):
    """Face-level (masked, carry_over) lists for nest_grid_mask, decided by
    geometry: a face is masked iff its first (boundary) window glues content
    that is not sky-adjacent (spec 3.3). carry_over[k] is the face whose dir2
    boundary content comes *from* masked face k (it holds k's carried pixels).

    Known theoretical residual: a face could in principle have both a real
    (non-self) taker and a self-pull (its own dir2 source maps to itself,
    i.e. two "takers" of face k under dir2_source, one of which is k itself).
    The single-carry contract here only separates one taker's content (a
    non-self taker is preferred; see nest_grid_mask's self-carry label for the
    self-pull case). No currently tested base-pixel subset produces a face
    with both a real taker and a self-pull simultaneously; the ground-truth
    test (test_nest_grid_mask_ground_truth) is the backstop for this."""
    base_pixels = list(base_pixels)
    n = len(base_pixels)
    face_len = nside * nside
    masked = [i for i in range(n)
              if not _window_slot_grid_ok(base_pixels, nside, window_size,
                                          shift_idcs[i * face_len:i * face_len + window_size])]
    _, off2 = derive_offset_tables(base_pixels)
    dir2_source = {j: (j - off2[j] - 1) % n for j in range(n)}
    carry = []
    for b in masked:
        takers = [j for j in range(n) if dir2_source[j] == b]
        non_self = [j for j in takers if j != b]
        carry.append(non_self[0] if non_self else (takers[0] if takers else None))
    return masked, carry


def derive_ring_lost_from(base_pixels):
    """RingShift backfill donor map (local positions): each face takes lost
    pixels from its same-latitude-row cyclic predecessor, except faces in the
    northernmost selected row, which are filled from the leftover pool. This
    reproduces the reference's {4:7, 5:4, 6:5, 7:6} for [0..7]."""
    base_pixels = list(base_pixels)
    rows = sorted({b // 4 for b in base_pixels})
    out = {}
    for i, b in enumerate(base_pixels):
        if b // 4 == rows[0]:
            continue
        sibs = [f for f in base_pixels if f // 4 == b // 4]
        if len(sibs) < 2:
            continue
        out[i] = base_pixels.index(sibs[(sibs.index(b) - 1) % len(sibs)])
    return out


def _window_components(base_pixels, nside, window_size, local_sources, backfilled):
    """Label window slots by sky-contiguity (union-find over canonically
    adjacent, sky-adjacent slot pairs); each backfilled slot is its own label."""
    from heal_swin_nnx.hp.windowing import get_nest_win_idcs
    grid = get_nest_win_idcs(window_size)
    s = grid.shape[0]
    glob = local_to_global(base_pixels, nside, np.asarray(local_sources))
    parent = list(range(window_size))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for gx in range(s):
        for gy in range(s):
            a = int(grid[gx, gy])
            if backfilled[a]:
                continue
            for nx, ny in ((gx + 1, gy), (gx, gy + 1)):
                if nx < s and ny < s and not backfilled[int(grid[nx, ny])]:
                    b = int(grid[nx, ny])
                    if int(glob[b]) in grid_neighbours(nside, int(glob[a])):
                        parent[find(a)] = find(b)
    roots, labels = {}, np.zeros(window_size)
    for slot in range(window_size):
        key = ("bf", slot) if backfilled[slot] else find(slot)
        labels[slot] = roots.setdefault(key, len(roots))
    return labels


def exact_shift_idcs_and_mask(base_pixels, nside, window_size):
    """Seam-exact shift indices + raw region mask (spec section 4, as amended).

    The intended source of destination pixel (x, y, f) is the pixel at
    (x - d, y - d) walked across face seams with orientation transforms,
    d = sqrt(ws)//2, matching the approximate strategy's interior behaviour.
    A single uniform (-d, -d) flow cannot be injective on the whole sphere:
    south-south seam crossings swap x and y (a 90-degree frame rotation, see
    side_matrix rows 8-11), so walked sources collide along those seams and
    near the poles. Assignment is therefore two-phase:

    Phase 1 (in-face): destinations whose source stays inside their own face
    take it unconditionally — an injective translation, identical to the
    approximate strategy's interior assignments.
    Phase 2 (cross-seam, ascending dest order for determinism): the walked
    source is assigned only if it lies in base_pixels and is still unclaimed.
    Pinch corners (walk -> None), out-of-subset sources, and collision losers
    all become holes, backfilled from the never-claimed (lost) pixels in
    ascending order; lost == holes then holds by counting.

    Masking is component-driven for every window: slots are labeled by
    sky-contiguity of their sources (backfilled slots are singleton labels);
    single-component windows carry no mask ('the gather is the rotation'),
    multi-component windows get distinct region labels.
    """
    base_pixels = list(base_pixels)
    face_len = nside * nside
    npix = len(base_pixels) * face_len
    s = int(round(window_size ** 0.5))
    assert s * s == window_size
    d = s // 2
    selected = set(base_pixels)

    x, y, f = pix2xyf(nside, local_to_global(base_pixels, nside, np.arange(npix)))
    src = np.full(npix, -1, dtype=np.int64)
    claimed = np.zeros(npix, dtype=bool)

    # phase 1: in-face translations
    for dest in range(npix):
        if x[dest] >= d and y[dest] >= d:
            g = xyf2pix(nside, int(x[dest]) - d, int(y[dest]) - d, int(f[dest]))
            sl = int(global_to_local(base_pixels, nside, g))
            src[dest] = sl
            claimed[sl] = True

    # phase 2: cross-seam walks; pinch corners, out-of-subset sources and
    # collision losers all become holes
    for dest in range(npix):
        if src[dest] >= 0:
            continue
        r = walk(nside, int(x[dest]) - d, int(y[dest]) - d, int(f[dest]))
        if r is None or r[2] not in selected:
            continue
        sl = int(global_to_local(base_pixels, nside,
                                 xyf2pix(nside, r[0], r[1], r[2])))
        if not claimed[sl]:
            src[dest] = sl
            claimed[sl] = True

    holes = np.where(src < 0)[0]
    lost = np.where(~claimed)[0]
    assert lost.shape[0] == holes.shape[0], (
        "backfill accounting mismatch: %d lost vs %d holes" % (lost.shape[0], holes.shape[0]))
    src[holes] = lost
    assert np.array_equal(np.sort(src), np.arange(npix)), "exact shift is not a permutation"

    mask = np.zeros(npix)
    next_label = 1.0
    for w in range(npix // window_size):
        sl = slice(w * window_size, (w + 1) * window_size)
        backfilled = np.isin(np.arange(sl.start, sl.stop), holes)
        labels = _window_components(base_pixels, nside, window_size, src[sl], backfilled)
        if labels.max() > 0:
            mask[sl] = labels + next_label
            next_label += labels.max() + 1
    return src, mask


def derive_offset_tables(base_pixels):
    """dir1/dir2 base-pixel offset tables for the NEST grid shift, keyed by
    local face position. A face whose source neighbour is outside base_pixels
    maps to itself (offset n-1), producing a fictitious seam that
    derive_mask_faces will mask."""
    base_pixels = list(base_pixels)
    n = len(base_pixels)
    pos = {f: i for i, f in enumerate(base_pixels)}
    off1, off2 = {}, {}
    for i, b in enumerate(base_pixels):
        nb1 = int(neighbours_matrix[b][DIR1_COL[b]])
        nb2 = int(neighbours_matrix[b][DIR2_COL[b]])
        off1[i] = (i - pos.get(nb1, i) - 1) % n
        off2[i] = (i - pos.get(nb2, i) - 1) % n
    return off1, off2
