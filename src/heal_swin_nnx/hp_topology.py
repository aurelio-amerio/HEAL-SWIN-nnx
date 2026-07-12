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
