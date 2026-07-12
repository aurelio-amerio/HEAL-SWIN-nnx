# Full-Sphere HEAL-SWIN Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend HEAL-SWIN to the full sphere (all 12 HEALPix base pixels) and arbitrary base-pixel subsets, with derived topology tables (no hardcoded constants), a geometric mask derivation, and a new exact seam-correct shift strategy.

**Architecture:** A new `hp_topology.py` module holds the HEALPix face-adjacency matrices (copied from Map2Patches with attribution), a healpy-free Morton pixel codec, a seam-crossing coordinate walker, and derivation functions for offset tables, masks, ring donors, and exact shift indices. `hp_shifting.py` consumes it; all computation stays construction-time numpy stored in `Buffer`s; the forward path remains the same `jnp.take` gathers.

**Tech Stack:** numpy (construction), JAX/flax-nnx (runtime), healpy (test-side ground truth only), pytest.

**Spec:** `docs/superpowers/specs/2026-07-12-full-sphere-extension-design.md` — read it before starting any task.

## Global Constraints

- Run tests with `.venv/bin/python -m pytest tests/<file> -v` (pyproject sets `JAX_PLATFORMS=cpu` and `-n 2`; `uv run` may fail if the uv cache is read-only).
- `src/heal_swin_nnx/hp_topology.py` must **never import healpy** — healpy is the independent verifier, used only in tests (`hp_shifting.py` keeps its existing function-local healpy import for the ring roll).
- Construction-time code may favour clarity over speed (per-pixel Python loops are fine). The forward path must remain precomputed-`Buffer` + `jnp.take`/`jnp.roll` only.
- Never regenerate goldens under `tests/goldens/` — the 8-base-pixel goldens are the parity gate. `base_pixels=[0..7]` derived results must be bit-identical to them.
- Follow existing style: module-level functions, `%`-formatting for messages, ~99-column lines, numpy at construction / jnp at runtime.
- Commit messages: conventional prefix (`feat:`/`test:`/`docs:`), ending with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Two flagged iteration points (expected, not failures): the seam walker's along-edge orientation (Task 3) and the parity-mask criterion strictness (Task 6). In both, the named test is the arbiter; the fix locations are documented in the task.

---

### Task 1: `DataSpec.base_pixels`

**Files:**
- Modify: `src/heal_swin_nnx/config.py:12-18`
- Test: `tests/test_dataspec.py` (create)

**Interfaces:**
- Produces: `DataSpec(dim_in, f_in, f_out, base_pix=None, base_pixels=None, class_names=[])`. After `__post_init__`: `base_pixels: List[int]` always set (default `list(range(12))`; legacy `base_pix=k` → `list(range(k))`), `base_pix == len(base_pixels)` always holds. Later tasks consume `data_spec.base_pixels`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dataspec.py`:

```python
import pytest

from heal_swin_nnx import DataSpec


def test_default_is_full_sphere():
    ds = DataSpec(dim_in=12 * 16 ** 2, f_in=1, f_out=1)
    assert ds.base_pixels == list(range(12))
    assert ds.base_pix == 12


def test_legacy_base_pix_resolves_to_prefix():
    ds = DataSpec(dim_in=8 * 16 ** 2, f_in=3, f_out=5, base_pix=8)
    assert ds.base_pixels == list(range(8))
    assert ds.base_pix == 8


def test_explicit_base_pixels():
    ds = DataSpec(dim_in=4 * 16 ** 2, f_in=1, f_out=1, base_pixels=[8, 9, 10, 11])
    assert ds.base_pixels == [8, 9, 10, 11]
    assert ds.base_pix == 4


def test_base_pix_and_base_pixels_must_agree():
    ds = DataSpec(dim_in=4 * 16 ** 2, f_in=1, f_out=1, base_pix=4, base_pixels=[8, 9, 10, 11])
    assert ds.base_pix == 4
    with pytest.raises(ValueError):
        DataSpec(dim_in=4 * 16 ** 2, f_in=1, f_out=1, base_pix=8, base_pixels=[8, 9, 10, 11])


@pytest.mark.parametrize("bad", [[0, 0, 1], [3, 2], [-1, 0], [11, 12]])
def test_invalid_base_pixels_rejected(bad):
    with pytest.raises(ValueError):
        DataSpec(dim_in=len(bad) * 16 ** 2, f_in=1, f_out=1, base_pixels=bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_dataspec.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'base_pixels'` (or attribute errors).

- [ ] **Step 3: Implement**

In `src/heal_swin_nnx/config.py`, replace the `DataSpec` dataclass with:

```python
@dataclass
class DataSpec:
    dim_in: Union[int, Tuple[int, int]]  # int (=npix) for HP, (H, W) for flat
    f_in: int
    f_out: int
    base_pix: Optional[int] = None           # legacy; derived from base_pixels after init
    base_pixels: Optional[List[int]] = None  # HEALPix faces used; None -> full sphere
    class_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.base_pixels is None:
            self.base_pixels = list(range(12 if self.base_pix is None else self.base_pix))
        self.base_pixels = list(self.base_pixels)
        if any(not 0 <= b <= 11 for b in self.base_pixels):
            raise ValueError("base_pixels must be in [0, 11], got %r" % (self.base_pixels,))
        if any(a >= b for a, b in zip(self.base_pixels, self.base_pixels[1:])):
            raise ValueError(
                "base_pixels must be strictly increasing (canonical NEST subset order), "
                "got %r" % (self.base_pixels,))
        if self.base_pix is not None and self.base_pix != len(self.base_pixels):
            raise ValueError("base_pix=%d inconsistent with base_pixels of length %d"
                             % (self.base_pix, len(self.base_pixels)))
        self.base_pix = len(self.base_pixels)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_dataspec.py tests/test_model.py -v`
Expected: PASS (test_model.py exercises the legacy `base_pix=8` path).

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/config.py tests/test_dataspec.py
git commit -m "feat: DataSpec.base_pixels with full-sphere default and legacy base_pix resolution"
```

---

### Task 2: Topology matrices and Morton pixel codec

**Files:**
- Create: `src/heal_swin_nnx/hp_topology.py`
- Test: `tests/test_topology.py` (create)

**Interfaces:**
- Produces (all in `heal_swin_nnx.hp_topology`):
  - `neighbours_matrix, side_matrix, corner_faces_matrix, corner_sides_matrix: np.ndarray` — shape (12, 4), int64.
  - `pix2xyf(nside, pix) -> (x, y, face)` — vectorized, global NEST pixel index → face-local coords; must match `healpy.pix2xyf(nside, pix, nest=True)` exactly.
  - `xyf2pix(nside, x, y, face) -> pix` — inverse.
  - `local_to_global(base_pixels, nside, idx) -> np.ndarray` — model-layout index → global NEST index.
  - `global_to_local(base_pixels, nside, idx) -> np.ndarray` — inverse; `-1` where the face is not selected.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_topology.py`:

```python
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
    # exactly 8 pinch corners, each shared by no diagonal face
    assert (hpt.corner_faces_matrix == -1).sum() == 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_topology.py -v`
Expected: FAIL with `ModuleNotFoundError: heal_swin_nnx.hp_topology`.

- [ ] **Step 3: Implement**

Create `src/heal_swin_nnx/hp_topology.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_topology.py -v`
Expected: PASS. If `test_pix2xyf_matches_healpy` fails with x/y swapped, swap the bit
roles in both codec functions (healpy is definitionally right here — Map2Patches'
matrices were built in healpy's frame).

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/hp_topology.py tests/test_topology.py
git commit -m "feat: hp_topology — face adjacency matrices and Morton pixel codec"
```

---

### Task 3: Seam walker and neighbour oracle

**Files:**
- Modify: `src/heal_swin_nnx/hp_topology.py` (append)
- Test: `tests/test_topology.py` (append)

**Interfaces:**
- Consumes: matrices + codec from Task 2.
- Produces:
  - `walk(nside, x, y, face) -> Optional[Tuple[int, int, int]]` — resolve possibly out-of-range face-local coords to on-sphere `(x, y, face)`; `None` at a pinch corner. Scalar ints.
  - `grid_neighbours(nside, pix) -> set[int]` — the ≤8 sky neighbours of a global NEST pixel.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_topology.py`:

```python
@pytest.mark.parametrize("nside", [2, 4, 8])
def test_grid_neighbours_match_healpy_exhaustively(nside):
    """Every pixel's neighbour set must equal healpy's (edges, corners,
    orientation, and pinch points all validated in one sweep)."""
    for pix in range(12 * nside ** 2):
        ours = hpt.grid_neighbours(nside, pix)
        theirs = set(int(v) for v in hp.get_all_neighbours(nside, pix, nest=True) if v >= 0)
        assert ours == theirs, "pixel %d: %r != %r" % (pix, sorted(ours), sorted(theirs))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_topology.py::test_grid_neighbours_match_healpy_exhaustively -v`
Expected: FAIL with `AttributeError: ... 'grid_neighbours'`.

- [ ] **Step 3: Implement**

Append to `src/heal_swin_nnx/hp_topology.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_topology.py -v`
Expected: PASS. **Flagged iteration point:** if specific seams fail, the failing
(face, column) pairs identify where `_enter` needs a run reversal
(`run -> nside - 1 - run`); add a reversal lookup table keyed by (face-row, side)
derived from the failing cases, re-run until the exhaustive sweep is green.
Do not weaken the test.

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/hp_topology.py tests/test_topology.py
git commit -m "feat: seam walker and neighbour oracle validated against healpy exhaustively"
```

---

### Task 4: Derived offset tables

**Files:**
- Modify: `src/heal_swin_nnx/hp_topology.py` (append)
- Test: `tests/test_topology.py` (append)

**Interfaces:**
- Produces: `derive_offset_tables(base_pixels) -> Tuple[Dict[int, int], Dict[int, int]]` — dir1/dir2 offset dicts keyed by *local* face position, drop-in for the legacy `NEST_GRID_BASE_PIX_OFFSETS_DIR1[8]` / `..._DIR2[8]` values.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_topology.py`:

```python
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
```

(The full-sphere and south-cap values were derived and permutation-validated in the
brainstorming session; see spec §3.1.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_topology.py -k derive_offset -v`
Expected: FAIL with `AttributeError: ... 'derive_offset_tables'`.

- [ ] **Step 3: Implement**

Append to `src/heal_swin_nnx/hp_topology.py`:

```python
# Which neighbour column is the dir1/dir2 shift source for each face; the
# south polar ring is reflected relative to the north, hence the swap.
DIR1_COL = [1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2]
DIR2_COL = [2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1]


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_topology.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/hp_topology.py tests/test_topology.py
git commit -m "feat: derive NestGridShift offset tables from face adjacency"
```

---

### Task 5: Generalize `nest_grid_shift_idcs` and `NestGridShift` to `base_pixels`

**Files:**
- Modify: `src/heal_swin_nnx/hp_shifting.py` (docstring, lines 1-10; tables, lines 63-68; `nest_grid_shift_idcs`, lines 113-143; `NestGridShift`, lines 178-195)
- Test: `tests/test_shifting.py` (modify)

**Interfaces:**
- Consumes: `hp_topology.derive_offset_tables`.
- Produces: `nest_grid_shift_idcs(nside, base_pixels, window_size) -> np.ndarray` (second arg now a sequence of face ids, **not** an int) and `NestGridShift(nside, base_pixels, window_size)`. The `NEST_GRID_BASE_PIX_OFFSETS_*` constants are deleted; `NEST_GRID_MASKED_BASE_PIX` / `NEST_GRID_LEFT_CARRY_OVER_BASE_PIX` remain until Task 6.

- [ ] **Step 1: Update existing tests and add the permutation matrix test**

In `tests/test_shifting.py`, change the two `nest_grid` call sites and replace the
`NotImplementedError` test:

```python
def test_nest_grid_idcs_bit_exact():
    npz, _ = load_case("indices")
    for nside, ws in _grid_combos(npz):
        tag = "ns%d_ws%d" % (nside, ws)
        got = hps.nest_grid_shift_idcs(nside, list(range(8)), ws)
        assert np.array_equal(got, npz["nest_grid/idcs/%s" % tag]), tag
        assert np.array_equal(np.argsort(got), npz["nest_grid/back/%s" % tag]), tag


def test_nest_grid_module_roundtrip_8pix():
    # NestGridShift masks are still legacy-keyed until Task 6; full-sphere and
    # subset module construction is tested there.
    sh = hps.NestGridShift(nside=16, base_pixels=list(range(8)), window_size=4)
    x = jnp.arange(1 * 2048 * 2, dtype=jnp.float32).reshape(1, 2048, 2)
    assert np.array_equal(sh.shift_back(sh.shift(x)), x)


def test_nest_grid_idcs_valid_permutation_matrix():
    for base_pixels in (list(range(12)), list(range(8)), [8, 9, 10, 11], [0, 4, 8]):
        for nside in (8, 16):
            for ws in (4, 16):
                # nest_grid_shift_idcs asserts permutation validity internally
                idcs = hps.nest_grid_shift_idcs(nside, base_pixels, ws)
                assert idcs.shape == (len(base_pixels) * nside ** 2,)
```

Leave `test_nest_grid_masks_bit_exact` failing for now if the mask call breaks —
Task 6 fixes masks; if needed, temporarily change its call to
`hps.nest_grid_mask(nside, list(range(8)), ws)` in this task (the function keeps
working via the legacy constants until Task 6 replaces them).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_shifting.py -v`
Expected: FAIL — `nest_grid_shift_idcs` still takes `base_pix` int / `NestGridShift` rejects the keyword.

- [ ] **Step 3: Implement**

In `src/heal_swin_nnx/hp_shifting.py`:

1. Replace the module docstring lines 5-10 with:

```python
Topology tables (base-pixel adjacency, shift offsets, masks) are derived at
construction time from the HEALPix face-adjacency data in hp_topology, for the
full sphere or any strictly-increasing subset of the 12 base pixels.
```

2. Delete `NEST_GRID_BASE_PIX_OFFSETS_DIR1` and `NEST_GRID_BASE_PIX_OFFSETS_DIR2`
   (keep the two mask constants until Task 6) and add the import:

```python
from heal_swin_nnx import hp_topology
```

3. Change `nest_grid_shift_idcs` signature and table lookup (only the first lines change;
   the loops stay identical):

```python
def nest_grid_shift_idcs(nside, base_pixels, window_size):
    base_pixels = list(base_pixels)
    base_pix = len(base_pixels)
    ws = window_size
    npix = base_pix * nside ** 2
    n_windows = npix // ws
    base_pix_len = (npix // base_pix) // ws
    hws, qws = ws // 2, ws // 4
    off1, off2 = hp_topology.derive_offset_tables(base_pixels)
    ...  # dir1/dir2 loops and the permutation assert are unchanged
```

4. Change `NestGridShift.__init__` to:

```python
class NestGridShift(nnx.Module):
    def __init__(self, nside, base_pixels, window_size):
        base_pixels = list(base_pixels)
        idcs = nest_grid_shift_idcs(nside, base_pixels, window_size)
        self.shift_idcs = Buffer(jnp.asarray(idcs))
        self.back_shift_idcs = Buffer(jnp.asarray(np.argsort(idcs)))
        self.attn_mask = Buffer(jnp.asarray(get_attn_mask_from_mask(
            nest_grid_mask(nside, len(base_pixels), window_size), window_size)))
```

(`nest_grid_mask` still takes the int and legacy constants in this task; Task 6
changes it to `base_pixels` and derived faces.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_shifting.py tests/test_parity_modules.py -v`
Expected: PASS — including bit-exact goldens for `[0..7]`. Note: `NestGridShift` for
subsets other than `[0..7]` now *constructs*, but its mask is still the legacy one
(wrong for other subsets) until Task 6 — acceptable mid-plan state.

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/hp_shifting.py tests/test_shifting.py
git commit -m "feat: NestGridShift on derived offset tables for arbitrary base_pixels"
```

---

### Task 6: Geometric mask derivation for the parity strategy

**Files:**
- Modify: `src/heal_swin_nnx/hp_topology.py` (append `derive_mask_faces`, `_window_slot_grid_ok`)
- Modify: `src/heal_swin_nnx/hp_shifting.py` (`nest_grid_mask`, lines 146-175; delete the two remaining mask constants; `NestGridShift.__init__` mask call)
- Test: `tests/test_topology.py`, `tests/test_shifting.py` (append)

**Interfaces:**
- Consumes: `derive_offset_tables`, `grid_neighbours`, `local_to_global`, `hp_windowing.get_nest_win_idcs`, `hp_shifting.nest_grid_shift_idcs`.
- Produces: `derive_mask_faces(base_pixels, nside, window_size, shift_idcs) -> Tuple[List[int], List[Optional[int]]]` — local-position `masked` list and aligned `carry_over` list (entry may be `None` if no face carries for a masked face — possible only in exotic subsets). `nest_grid_mask(nside, base_pixels, window_size)` now takes the sequence.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_topology.py`:

```python
def test_derive_mask_faces_reproduces_reference_8pix():
    from heal_swin_nnx.hp_shifting import nest_grid_shift_idcs
    base_pixels, nside, ws = list(range(8)), 16, 4
    idcs = nest_grid_shift_idcs(nside, base_pixels, ws)
    masked, carry = hpt.derive_mask_faces(base_pixels, nside, ws, idcs)
    assert masked == [4, 5, 6, 7]
    assert carry == [0, 1, 2, 3]
```

Append to `tests/test_shifting.py`:

```python
def test_nest_grid_module_roundtrip_full_sphere_and_subsets():
    for base_pixels in (list(range(12)), [8, 9, 10, 11], [0, 4, 8]):
        npix = len(base_pixels) * 16 ** 2
        sh = hps.NestGridShift(nside=16, base_pixels=base_pixels, window_size=4)
        x = jnp.arange(1 * npix * 2, dtype=jnp.float32).reshape(1, npix, 2)
        assert np.array_equal(sh.shift_back(sh.shift(x)), x), base_pixels


def _slot_grid(ws):
    from heal_swin_nnx.hp_windowing import get_nest_win_idcs
    return get_nest_win_idcs(ws)


def test_nest_grid_mask_ground_truth_full_sphere_and_south_cap():
    """Unmasked canonically-adjacent slot pairs must be sky-adjacent (spec 6.2)."""
    import healpy as hp
    from heal_swin_nnx import hp_topology as hpt
    for base_pixels in (list(range(12)), [8, 9, 10, 11]):
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_topology.py::test_derive_mask_faces_reproduces_reference_8pix tests/test_shifting.py::test_nest_grid_mask_ground_truth_full_sphere_and_south_cap -v`
Expected: FAIL — `derive_mask_faces` missing; `nest_grid_mask` rejects a list.

- [ ] **Step 3: Implement**

Append to `src/heal_swin_nnx/hp_topology.py`:

```python
def _window_slot_grid_ok(base_pixels, nside, window_size, local_sources):
    """True iff every canonically-adjacent slot pair in this window holds
    sky-adjacent source pixels (the seam-correctness criterion)."""
    from heal_swin_nnx.hp_windowing import get_nest_win_idcs
    grid = get_nest_win_idcs(window_size)
    s = grid.shape[0]
    glob = local_to_global(base_pixels, nside, np.asarray(local_sources))
    for gx in range(s):
        for gy in range(s):
            p = int(glob[grid[gx, gy]])
            for nx, ny in ((gx + 1, gy), (gx, gy + 1)):
                if nx < s and ny < s:
                    if int(glob[grid[nx, ny]]) not in grid_neighbours(nside, p):
                        return False
    return True


def derive_mask_faces(base_pixels, nside, window_size, shift_idcs):
    """Face-level (masked, carry_over) lists for nest_grid_mask, decided by
    geometry: a face is masked iff its first (boundary) window glues content
    that is not sky-adjacent (spec 3.3). carry_over[k] is the face whose dir2
    boundary content comes *from* masked face k (it holds k's carried pixels)."""
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
        takers = [j for j in range(n) if dir2_source[j] == b and j != b]
        carry.append(takers[0] if takers else None)
    return masked, carry
```

In `src/heal_swin_nnx/hp_shifting.py`, delete `NEST_GRID_MASKED_BASE_PIX` and
`NEST_GRID_LEFT_CARRY_OVER_BASE_PIX`, and change `nest_grid_mask`:

```python
def nest_grid_mask(nside, base_pixels, window_size):
    base_pixels = list(base_pixels)
    base_pix = len(base_pixels)
    ws = window_size
    hws, qws = ws // 2, ws // 4
    npix = base_pix * nside ** 2
    base_pix_len = (npix // base_pix) // ws
    masked, carry = hp_topology.derive_mask_faces(
        base_pixels, nside, ws, nest_grid_shift_idcs(nside, base_pixels, ws))
    mask = np.zeros(npix)

    ...  # right_mask_subset / left_mask_subset inner functions unchanged

    for b, co in zip(masked, carry):
        left_mask_subset(b * base_pix_len * ws, base_pix_len * ws, b + 1)
        right_mask_subset(b * base_pix_len * ws, base_pix_len * ws, b + 1 + len(masked))
        if co is not None:
            first_co = co * base_pix_len * ws
            mask[first_co:first_co + qws] = b + 1
    return mask
```

Update `NestGridShift.__init__`'s mask call to
`nest_grid_mask(nside, base_pixels, window_size)`, and
`tests/test_shifting.py::test_nest_grid_masks_bit_exact` to call
`hps.nest_grid_mask(nside, list(range(8)), ws)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_topology.py tests/test_shifting.py -v`
Expected: PASS, including `test_nest_grid_masks_bit_exact` (goldens). **Flagged
iteration point:** if `derive_mask_faces` returns *more* than `[4,5,6,7]` for
`[0..7]` (criterion stricter than the reference), weaken `_window_slot_grid_ok`
to check only pairs whose two sources lie in *different* faces (within-face
Z-hierarchy jumps are the reference's accepted layout). The two named tests are
the acceptance gate: 8-pix regression AND ground-truth mask must both pass.

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/hp_topology.py src/heal_swin_nnx/hp_shifting.py tests/test_topology.py tests/test_shifting.py
git commit -m "feat: derive NestGridShift masks geometrically; delete hardcoded face lists"
```

---

### Task 7: Generalize `RingShift`

**Files:**
- Modify: `src/heal_swin_nnx/hp_topology.py` (append `derive_ring_lost_from`)
- Modify: `src/heal_swin_nnx/hp_shifting.py` (`RING_GET_LOST_FROM` deleted; `ring_shift_idcs_and_mask` lines 202-259; `RingShift` lines 262-273)
- Test: `tests/test_shifting.py` (modify golden test call, append subset tests); `tests/test_topology.py` (append donor test)

**Interfaces:**
- Consumes: `local_to_global`, `global_to_local`.
- Produces: `derive_ring_lost_from(base_pixels) -> Dict[int, int]` (local donor map; faces in the northernmost selected row are pool-filled and absent from the dict); `ring_shift_idcs_and_mask(nside, base_pixels, window_size, shift_size)`; `RingShift(nside, base_pixels, window_size, shift_size)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_topology.py`:

```python
def test_derive_ring_lost_from_reproduces_reference_8pix():
    assert hpt.derive_ring_lost_from(list(range(8))) == {4: 7, 5: 4, 6: 5, 7: 6}
```

In `tests/test_shifting.py`, change the golden call to
`hps.ring_shift_idcs_and_mask(nside, list(range(8)), ws, ws // 2)`, replace
`test_ring_module_and_unsupported_base_pix` with a roundtrip over
`(list(range(12)), list(range(8)), [8, 9, 10, 11], [0, 4, 8])` (same pattern as
`test_nest_grid_module_roundtrip_full_sphere_and_subsets`, constructor
`hps.RingShift(nside=16, base_pixels=base_pixels, window_size=4, shift_size=2)`),
and append:

```python
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
    from heal_swin_nnx import hp_topology as hpt
    sel = hpt.local_to_global(base_pixels, nside, np.arange(len(idcs)))
    out_of_domain = hpt.global_to_local(base_pixels, nside, full[sel]) < 0
    assert np.array_equal(raw > 0, out_of_domain)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_shifting.py -k ring tests/test_topology.py -k ring -v`
Expected: FAIL — signature/NotImplementedError/missing function.

- [ ] **Step 3: Implement**

Append to `src/heal_swin_nnx/hp_topology.py`:

```python
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
```

Replace `ring_shift_idcs_and_mask` and delete `RING_GET_LOST_FROM`:

```python
def ring_shift_idcs_and_mask(nside, base_pixels, window_size, shift_size):
    import healpy as hp  # local import: healpy pulls matplotlib; keep module import light

    base_pixels = list(base_pixels)
    n = len(base_pixels)
    pixel_size = nside ** 2
    npix = n * pixel_size

    # roll on the full 12-face sphere in ring order, then restrict to the subset
    ring_idcs = np.arange(12 * pixel_size)
    shifted_ring_idcs_in_nest = hp.pixelfunc.ring2nest(nside, np.roll(ring_idcs, shift_size))
    full_result = shifted_ring_idcs_in_nest[
        hp.pixelfunc.nest2ring(nside, np.arange(12 * pixel_size))]
    sel = hp_topology.local_to_global(base_pixels, nside, np.arange(npix))
    result = hp_topology.global_to_local(base_pixels, nside, full_result[sel])

    mask = np.zeros(npix)
    for i in range(n):
        sl = slice(i * pixel_size, (i + 1) * pixel_size)
        mask[sl][result[sl] < 0] = i + 1

    lost_pix = [np.setdiff1d(np.arange(i * pixel_size, (i + 1) * pixel_size), result)
                for i in range(n)]
    get_lost_from = hp_topology.derive_ring_lost_from(base_pixels)

    # first pass: donor-fed faces (reference behaviour for its faces 4..7)
    leftover, donated = [], set()
    for i in range(n):
        if i not in get_lost_from:
            continue
        sl = slice(i * pixel_size, (i + 1) * pixel_size)
        result_subset = result[sl]
        holes = np.where(result_subset < 0)[0]
        source_pix = lost_pix[get_lost_from[i]]
        donated.add(get_lost_from[i])
        take = min(holes.shape[0], source_pix.shape[0])
        result_subset[holes[:take]] = source_pix[:take]
        leftover.append(source_pix[take:])
    # pool: donor remainders (reference order), then lost pixels of never-donor faces
    pool = np.concatenate(
        leftover + [lost_pix[i] for i in range(n) if i not in donated]
        or [np.array([], dtype=np.int64)]).astype(np.int64)

    # second pass: remaining holes in face order (reference behaviour for faces 0..3)
    first = 0
    for i in range(n):
        sl = slice(i * pixel_size, (i + 1) * pixel_size)
        result_subset = result[sl]
        holes = np.where(result_subset < 0)[0]
        result_subset[holes] = pool[first:first + holes.shape[0]]
        first += holes.shape[0]

    result = result.astype(np.int64)
    assert np.array_equal(np.sort(result), np.arange(npix)), (
        "shift validation failed for nside=%d, window_size=%d" % (nside, window_size))
    return result, mask.astype(np.int64)
```

`RingShift.__init__` becomes:

```python
class RingShift(nnx.Module):
    def __init__(self, nside, base_pixels, window_size, shift_size):
        idcs, raw_mask = ring_shift_idcs_and_mask(nside, list(base_pixels),
                                                  window_size, shift_size)
        self.shift_idcs = Buffer(jnp.asarray(idcs))
        self.back_shift_idcs = Buffer(jnp.asarray(np.argsort(idcs)))
        self.attn_mask = Buffer(jnp.asarray(get_attn_mask_from_mask(raw_mask, window_size)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_shifting.py tests/test_topology.py -v`
Expected: PASS — `test_ring_idcs_and_masks_bit_exact` (goldens) proves `[0..7]`
bit-exactness of the generalized code path.

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/hp_topology.py src/heal_swin_nnx/hp_shifting.py tests/test_shifting.py tests/test_topology.py
git commit -m "feat: RingShift on full sphere and arbitrary subsets with derived donors"
```

---

### Task 8: Exact geometric shift — `NestGridShiftExact`

**Files:**
- Modify: `src/heal_swin_nnx/hp_topology.py` (append `exact_shift_idcs_and_mask`, `_window_components`)
- Modify: `src/heal_swin_nnx/hp_shifting.py` (append `NestGridShiftExact`)
- Test: `tests/test_shifting.py` (append)

**Interfaces:**
- Consumes: `pix2xyf`, `xyf2pix`, `walk`, `grid_neighbours`, `local_to_global`, `global_to_local`, `get_nest_win_idcs`.
- Produces: `exact_shift_idcs_and_mask(base_pixels, nside, window_size) -> Tuple[np.ndarray, np.ndarray]` (shift indices, raw region mask); `hp_shifting.NestGridShiftExact(nside, base_pixels, window_size)` with the standard `shift`/`shift_back`/`attn_mask` interface.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_shifting.py`:

```python
from heal_swin_nnx import hp_topology as hpt


def test_exact_shift_valid_permutation_and_backfill_accounting():
    # full sphere: only the 4 south-pointing pinch corners are hit by the
    # (-d, -d) walk -> exactly 4 * d^2 backfilled slots; south cap closes on
    # itself around the pole -> zero holes, zero mask.
    for base_pixels, pinch_faces in ((list(range(12)), 4), ([8, 9, 10, 11], 0)):
        nside, ws = 8, 4
        d = int(round(ws ** 0.5)) // 2
        idcs, raw = hpt.exact_shift_idcs_and_mask(base_pixels, nside, ws)
        npix = len(base_pixels) * nside ** 2
        assert np.array_equal(np.sort(idcs), np.arange(npix))
        if pinch_faces == 0:
            assert not raw.any()
        else:
            # d^2 backfilled slots per pinch-hitting face, one affected window each
            n_masked_windows = len(np.unique(np.nonzero(raw)[0] // ws))
            assert n_masked_windows == pinch_faces
            # backfilled slots carry unique labels; count windows with >1 label region
            assert (raw > 0).sum() >= pinch_faces * d * d


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_shifting.py -k exact -v`
Expected: FAIL — missing functions/class.

- [ ] **Step 3: Implement**

Append to `src/heal_swin_nnx/hp_topology.py`:

```python
def _window_components(base_pixels, nside, window_size, local_sources, backfilled):
    """Label window slots by sky-contiguity (union-find over canonically
    adjacent, sky-adjacent slot pairs); each backfilled slot is its own label."""
    from heal_swin_nnx.hp_windowing import get_nest_win_idcs
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
    """Seam-exact shift indices + raw region mask (spec section 4).

    The source of destination pixel (x, y, f) is the pixel at (x - d, y - d)
    walked across face seams with orientation transforms, d = sqrt(ws)//2.
    This matches the approximate strategy's interior behaviour (a destination
    window's slots pull content from the Z-order-previous half window, i.e.
    smaller face coordinates). Destinations whose source falls off a pinch
    corner or outside base_pixels are backfilled from the lost-pixel pool and
    masked; all other windows carry no mask ('the gather is the rotation').
    """
    base_pixels = list(base_pixels)
    face_len = nside * nside
    npix = len(base_pixels) * face_len
    s = int(round(window_size ** 0.5))
    assert s * s == window_size
    d = s // 2
    selected = set(base_pixels)

    src = np.full(npix, -1, dtype=np.int64)
    for dest in range(npix):
        g = int(local_to_global(base_pixels, nside, dest))
        x, y, f = pix2xyf(nside, g)
        r = walk(nside, int(x) - d, int(y) - d, int(f))
        if r is not None and r[2] in selected:
            src[dest] = int(global_to_local(base_pixels, nside,
                                            xyf2pix(nside, r[0], r[1], r[2])))

    holes = np.where(src < 0)[0]
    used = np.zeros(npix, dtype=bool)
    used[src[src >= 0]] = True
    lost = np.where(~used)[0]
    assert lost.shape[0] == holes.shape[0], (
        "backfill accounting mismatch: %d lost vs %d holes" % (lost.shape[0], holes.shape[0]))
    src[holes] = lost
    assert np.array_equal(np.sort(src), np.arange(npix)), "exact shift is not a permutation"

    mask = np.zeros(npix)
    next_label = 1.0
    for w in np.unique(holes // window_size):
        sl = slice(w * window_size, (w + 1) * window_size)
        backfilled = np.isin(np.arange(sl.start, sl.stop), holes)
        labels = _window_components(base_pixels, nside, window_size, src[sl], backfilled)
        mask[sl] = labels + next_label
        next_label += labels.max() + 1
    return src, mask
```

Append to `src/heal_swin_nnx/hp_shifting.py`:

```python
class NestGridShiftExact(nnx.Module):
    """Seam-exact grid shift: connects the true boundary pixels across every
    face seam (mask only at pinch corners and subset borders). See spec §4."""

    def __init__(self, nside, base_pixels, window_size):
        idcs, raw_mask = hp_topology.exact_shift_idcs_and_mask(
            list(base_pixels), nside, window_size)
        self.shift_idcs = Buffer(jnp.asarray(idcs))
        self.back_shift_idcs = Buffer(jnp.asarray(np.argsort(idcs)))
        self.attn_mask = Buffer(jnp.asarray(get_attn_mask_from_mask(raw_mask, window_size)))

    def shift(self, x):
        return jnp.take(x, self.shift_idcs[...], axis=1)

    def shift_back(self, x):
        return jnp.take(x, self.back_shift_idcs[...], axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_shifting.py -v`
Expected: PASS. If `test_exact_matches_approximate_on_face_interiors` fails
*everywhere*, the sign convention is flipped: change the walk target to
`(x + d, y + d)` (and the hole-count expectation moves from the south-pointing
to the north-pointing pinch corners). If it fails only *near* within-face
window boundaries, that is a real bug in the walker or codec — use
superpowers:systematic-debugging, do not weaken the test.

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/hp_topology.py src/heal_swin_nnx/hp_shifting.py tests/test_shifting.py
git commit -m "feat: NestGridShiftExact — seam-exact shift with pinch-corner backfill"
```

---

### Task 9: Seam-geometry test module (headline invariant)

**Files:**
- Test: `tests/test_seam_geometry.py` (create)

**Interfaces:**
- Consumes: `exact_shift_idcs_and_mask`, `local_to_global`, `get_nest_win_idcs`, healpy.

- [ ] **Step 1: Write the tests (they should pass immediately if Task 8 is correct — the point is independent, permanent coverage)**

Create `tests/test_seam_geometry.py`:

```python
"""The central invariant of the full-sphere extension (spec 1.1 and 6.1):
pixels adjacent in a window's canonical 2D layout must be adjacent on the sky.

Unit-offset checks (horizontal, vertical, diagonal) against healpy pin the
window's local isometry; consistency of larger offsets follows by composition
along grid paths, so unit steps are the complete geometric claim."""
import healpy as hp
import numpy as np
import pytest

from heal_swin_nnx import hp_topology as hpt
from heal_swin_nnx.hp_windowing import get_nest_win_idcs

CONFIGS = [
    (list(range(12)), 8, 4),
    (list(range(12)), 8, 16),
    ([8, 9, 10, 11], 8, 4),
]


def _healpy_neighbours(nside, p):
    return set(int(v) for v in hp.get_all_neighbours(nside, p, nest=True) if v >= 0)


@pytest.mark.parametrize("base_pixels,nside,ws", CONFIGS)
def test_exact_shift_seam_correctness(base_pixels, nside, ws):
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
    assert checked_cross_face > 0, "no cross-face pair was exercised — config too small"


@pytest.mark.parametrize("base_pixels,nside,ws", CONFIGS)
def test_unmasked_windows_are_single_component(base_pixels, nside, ws):
    """Every window without mask labels holds one geographically contiguous patch."""
    idcs, raw = hpt.exact_shift_idcs_and_mask(base_pixels, nside, ws)
    for w in range(len(idcs) // ws):
        win_lbl = raw[w * ws:(w + 1) * ws]
        if win_lbl.any():
            continue
        comp = hpt._window_components(base_pixels, nside, ws, idcs[w * ws:(w + 1) * ws],
                                      np.zeros(ws, dtype=bool))
        assert comp.max() == 0, "window %d splits into %d components" % (w, int(comp.max()) + 1)
```

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_seam_geometry.py -v`
Expected: PASS. Any failure here is a real geometry bug in Tasks 3 or 8 — use
superpowers:systematic-debugging; never mark xfail.

- [ ] **Step 3: Commit**

```bash
git add tests/test_seam_geometry.py
git commit -m "test: seam-correctness invariant — window layout adjacency equals sky adjacency"
```

---

### Task 10: Thread `base_pixels` through the model and add the strategy switch

**Files:**
- Modify: `src/heal_swin_nnx/config.py:26` (`shift_strategy` Literal)
- Modify: `src/heal_swin_nnx/swin_hp_transformer.py:117-158` (`SwinTransformerBlock`), `:181-191` (`_make_blocks`), `:195-203` and `:218-226` (`BasicLayer`/`BasicLayer_up`), `:284` and `:330` (call sites)
- Test: `tests/test_model.py` (append)

**Interfaces:**
- Consumes: `DataSpec.base_pixels`, all four shifter classes.
- Produces: model constructors accept the same public API (`config`, `data_spec`); internal block/layer parameter `base_pix: int` is renamed to `base_pixels: Tuple[int, ...]` everywhere.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model.py`:

```python
def tiny_hp_pixels(base_pixels, strategy, nside=16):
    cfg = SwinHPTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4],
                                  drop_path_rate=0.0, shift_strategy=strategy)
    ds = DataSpec(dim_in=len(base_pixels) * nside ** 2, f_in=3, f_out=5,
                  base_pixels=base_pixels)
    return SwinHPTransformerSys(cfg, ds, rngs=nnx.Rngs(0)), ds


@pytest.mark.parametrize("strategy",
                         ["nest_roll", "nest_grid_shift", "nest_grid_shift_exact",
                          "ring_shift"])
@pytest.mark.parametrize("base_pixels", [list(range(12)), [8, 9, 10, 11]])
def test_forward_full_sphere_and_south_cap(base_pixels, strategy):
    model, ds = tiny_hp_pixels(base_pixels, strategy)
    model.eval()
    x = jax.random.normal(jax.random.key(0), (2, ds.dim_in, 3))
    y = model(x)
    assert y.shape == (2, ds.dim_in, 5)
    assert np.isfinite(np.asarray(y)).all()


def test_legacy_8pix_path_still_works():
    model, ds = tiny_hp()  # existing helper, base_pix=8
    model.eval()
    x = jax.random.normal(jax.random.key(0), (1, ds.dim_in, 3))
    assert model(x).shape == (1, ds.dim_in, 5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_model.py -k "full_sphere or legacy" -v`
Expected: FAIL — `'nest_grid_shift_exact'` unknown / shifters receive `base_pix` int.

- [ ] **Step 3: Implement**

1. `config.py`: `shift_strategy: Literal["nest_roll", "nest_grid_shift", "nest_grid_shift_exact", "ring_shift"] = "nest_roll"`.
2. `swin_hp_transformer.py`: rename the `base_pix` parameter to `base_pixels` in
   `SwinTransformerBlock.__init__`, `_make_blocks`, `BasicLayer.__init__`,
   `BasicLayer_up.__init__`; inside the block:

```python
        nside = math.sqrt(input_resolution // len(base_pixels))
        assert nside % 1 == 0, "nside has to be an integer in every layer"
        nside = int(nside)

        if self.shift_size > 0:
            if shift_strategy == "nest_roll":
                self.shifter = hp_shifting.NestRollShift(
                    shift_size=self.shift_size, input_resolution=self.input_resolution,
                    window_size=self.window_size)
            elif shift_strategy == "nest_grid_shift":
                self.shifter = hp_shifting.NestGridShift(
                    nside=nside, base_pixels=base_pixels, window_size=self.window_size)
            elif shift_strategy == "nest_grid_shift_exact":
                self.shifter = hp_shifting.NestGridShiftExact(
                    nside=nside, base_pixels=base_pixels, window_size=self.window_size)
            elif shift_strategy == "ring_shift":
                self.shifter = hp_shifting.RingShift(
                    nside=nside, base_pixels=base_pixels, window_size=self.window_size,
                    shift_size=self.shift_size)
            else:
                raise ValueError("unknown shift_strategy %r" % shift_strategy)
```

3. At the two call sites (`SwinHPEncoder.__init__`, `HPUnetDecoder.__init__`), pass
   `base_pixels=tuple(data_spec.base_pixels)` instead of `base_pix=data_spec.base_pix`.

Note on spec §5 (`weight_transfer.py`): verified during planning — `load_torch_state`
is shape-driven and never reads `base_pix`, so no change is needed there. Reference
checkpoints are 8-pix; loading them into a non-`[0..7]` model fails naturally on the
shape of any resolution-dependent parameter.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS, including the e2e/f64 parity suites (legacy `base_pix=8`
resolves to `[0..7]`, whose derived tables are bit-identical to the reference).

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/config.py src/heal_swin_nnx/swin_hp_transformer.py tests/test_model.py
git commit -m "feat: thread base_pixels through the model; add nest_grid_shift_exact strategy"
```

---

### Task 11: README and final verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-12-full-sphere-healswin-handout.md:4` (status line)

**Interfaces:** none (documentation).

- [ ] **Step 1: Add the README section**

After the existing usage/example section of `README.md`, add:

```markdown
## Full sphere and partial coverage

Models cover the full sphere by default (all 12 HEALPix base pixels). Experiments
that only see part of the sky select the base pixels they cover — e.g. a
ground-based south-pole telescope observing the four southern faces:

```python
from heal_swin_nnx import DataSpec, SwinHPTransformerConfig, SwinHPTransformerSys

nside = 256
data_spec = DataSpec(dim_in=4 * nside**2, f_in=1, f_out=1, base_pixels=[8, 9, 10, 11])
config = SwinHPTransformerConfig(shift_strategy="nest_grid_shift_exact")
```

Inputs are the concatenation of the selected faces' NEST-ordered pixels.
Shift strategies:

- `nest_roll` — 1D roll on the NEST sequence (cheapest, coarsest).
- `nest_grid_shift` — the reference HEAL-SWIN hierarchical grid shift; face-seam
  windows that glue geometrically wrong edges are attention-masked.
- `nest_grid_shift_exact` — seam-exact variant: window content crosses face seams
  with the correct pixels and orientation, so masking remains only at the 8
  pinch points (and at coverage borders for partial-sky models).
- `ring_shift` — shift along HEALPix iso-latitude rings; exact on the full sphere.
```

- [ ] **Step 2: Mark the handout superseded**

Change the handout's `**Status:**` line to:
`**Status:** Superseded by 2026-07-12-full-sphere-extension-design.md (its offset derivation was verified and carried over)`

- [ ] **Step 3: Full-suite verification**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS. Then run the verify skill if available for a forward-path sanity
drive of one full-sphere config.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/superpowers/specs/2026-07-12-full-sphere-healswin-handout.md
git commit -m "docs: full-sphere and partial-coverage usage; supersede handout"
```
