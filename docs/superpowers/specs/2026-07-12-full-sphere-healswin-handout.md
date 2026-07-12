# Full-Sphere HEAL-SWIN: Brainstorming Handout

**Date:** 2026-07-12  
**Status:** Superseded by 2026-07-12-full-sphere-extension-design.md (its offset derivation was verified and carried over)

This document captures all numerical findings and design decisions from the brainstorming
session so work can resume without re-derivation.

---

## What we are building

Phase 2 of the NNX port: extend HEAL-SWIN to work on the **full sphere** (all 12 HEALPix
base pixels) and on **arbitrary user-specified subsets** of base pixels (e.g. `[8,9,10,11]`
for a south-pole telescope). The default is the full sphere.

---

## API decision: `DataSpec`

Replace `dim_in` + `base_pix: int` with explicit geometric parameters:

```python
@dataclass
class DataSpec:
    base_pixels: List[int]   # e.g. list(range(12)) or [8,9,10,11]
    nside: int
    f_in: int
    f_out: int
    class_names: List[str] = field(default_factory=list)

    @property
    def npix(self) -> int:
        return len(self.base_pixels) * self.nside ** 2

    @property
    def base_pix(self) -> int:   # internal alias used by model code
        return len(self.base_pixels)
```

`dim_in` is gone — callers use `data_spec.npix`. Validated: all values in `[0,11]`, no
duplicates.

---

## Shift strategy decisions

| Strategy | Full sphere | Arbitrary subset | Notes |
|---|---|---|---|
| `NestRollShift` | Already works | Already works | No changes needed |
| `RingShift` | Clean no-op backfill | Backfill still needed | Remove NotImplementedError guard, generalise loop ranges |
| `NestGridShift` | **Derivable — see below** | Derivable with masking | Replace hardcoded tables with formula |

---

## NestGridShift: full derivation (validated)

### Key data structure (copy from map2patches, no runtime dependency)

```python
# Copied from Map2Patches (Amerio et al.).
# Row i = [top-left, bottom-left, bottom-right, top-right] neighbour face indices.
_HEALPIX_FACE_NEIGHBOURS = np.array([
    [1, 5, 4, 3], [2, 6, 5, 0], [3, 7, 6, 1], [0, 4, 7, 2],   # north polar 0-3
    [0, 8,11, 3], [1, 9, 8, 0], [2,10, 9, 1], [3,11,10, 2],   # equatorial  4-7
    [5, 9,11, 4], [6,10, 8, 5], [7,11, 9, 6], [4, 8,10, 7],   # south polar 8-11
], dtype=np.int64)
```

### Column mapping (which neighbour is the dir1/dir2 source)

Determined empirically via healpy `get_all_neighbours` on face boundary pixels and
confirmed by permutation validity test:

| Face group | dir1 column | dir2 column | Physical meaning |
|---|---|---|---|
| North polar 0–3 | col 1 (bottom-left) | col 2 (bottom-right) | — |
| Equatorial 4–7 | col 0 (top-left) | col 0 (top-left) | same neighbour both dirs |
| South polar 8–11 | col 2 (bottom-right) | col 1 (bottom-left) | **swapped** vs north polar |

The south-polar swap is a reflection symmetry. Confirmed: face 8's x=0 (left) edge
connects to face 11 (dir1, col2=bottom-right), and y=0 (bottom) edge connects to
face 9 (dir2, col1=bottom-left).

```python
_DIR_COL = {
    "dir1": [1, 1, 1, 1,  0, 0, 0, 0,  2, 2, 2, 2],
    "dir2": [2, 2, 2, 2,  0, 0, 0, 0,  1, 1, 1, 1],
}
```

### Formula (validated: all 8 cases match hardcoded tables exactly)

```python
def _derive_offset_tables(base_pixels: List[int]) -> Tuple[dict, dict]:
    n = len(base_pixels)
    pos = {f: i for i, f in enumerate(base_pixels)}
    off1, off2 = {}, {}
    for i, b in enumerate(base_pixels):
        nb1 = _HEALPIX_FACE_NEIGHBOURS[b][_DIR_COL["dir1"][b]]
        nb2 = _HEALPIX_FACE_NEIGHBOURS[b][_DIR_COL["dir2"][b]]
        # If neighbour is outside the selected subset, pos.get returns i → offset=n-1 → mask
        off1[i] = (i - pos.get(nb1, i) - 1) % n
        off2[i] = (i - pos.get(nb2, i) - 1) % n
    return off1, off2
```

**Derived 12-pixel tables** (validated: produce a valid permutation of all 12·nside² pixels):

```
DIR1: {0:6, 1:6, 2:6, 3:10, 4:3, 5:3, 6:3, 7:3, 8:8, 9:0, 10:0, 11:0}
DIR2: {0:7, 1:7, 2:7, 3:7,  4:3, 5:3, 6:3, 7:3, 8:10, 9:10, 10:10, 11:2}
```

Correct hypothesis for south polar: H2 (col2/col1). Only 2 of 16 combinations produce
valid permutations; H2 confirmed by healpy boundary analysis.

---

## NestGridShift masking

**What it does:** prevents non-sky-adjacent pixels — placed in the same boundary window
by the linearization — from attending to each other. This is the standard Swin cyclic-shift
masking mechanism applied to the 1D NEST sequence.

**Why it's still needed on the full sphere:** The NEST Z-order places the first/last pixels
of each face at the (0,0) and (nside-1,nside-1) *corners* of the face's 2D grid. When the
shift creates a boundary window spanning face A and face B, the specific NEST-order edge
pixels placed together may be at the *wrong geographic edge* of each face — not at the
actual seam where A and B touch on the sky.

Confirmed: face 8's NEST-first pixels are on its y=0 (bottom) edge, which borders face 9.
But dir1 connects face 8 to face 11 via the x=0 (left) edge. So face 8's NEST-first pixels
and face 11's last pixels are placed in the same dir1 boundary window, yet are not
sky-adjacent.

**For the full sphere**, the masking pattern gains a second tier:
- 8-pixel: masked=[4,5,6,7], carry_over=[0,1,2,3]
- 12-pixel: masked=[4,5,6,7, 8,9,10,11], carry_over=[0,1,2,3, 4,5,6,7]

The `nest_grid_mask` function must be generalised to derive `masked` and `carry_over`
lists from the topology tables rather than using hardcoded constants.

**Approach C footnote:** Pixel-level index derivation via `hp.pix2xyf / hp.xyf2pix` (what
we called Approach C) would eliminate this masking artifact entirely — it would connect
*exactly* the right boundary pixels at each face seam. This is a concrete but currently
deferred advantage. A code comment should document this as the fallback if the masking
derivation proves intractable or if the power-of-4 constraint in `_get_offset_dir1/dir2`
becomes a blocker.

---

## RingShift for the full sphere

For `set(base_pixels) == set(range(12))`:
- The ring shift is already a pure permutation of all 12·nside² pixels
- `mask` stays all-zero (no out-of-domain pixels)
- Backfill loops process zero pixels — they are no-ops
- The existing code computes this correctly; only the `NotImplementedError` guard
  and the hardcoded `range(4, base_pix)` / `range(4)` loop bounds need to change

For arbitrary subsets: `GET_LOST_FROM` becomes a construction-time dict derived from
`_HEALPIX_FACE_NEIGHBOURS`, identifying which face "donates" lost pixels to each masked face.

---

## map2patches: usage terms

The `neighbours_matrix` and adjacency data are copied inline into `hp_shifting.py` with
attribution. No runtime dependency on `map2patches`. The package was never formally
published, so copying is fine (Amerio, author of both projects).

---

## What is NOT yet designed

1. **`_derive_mask_pairs(base_pixels)`**: the function that produces the `masked` and
   `carry_over` face lists for `nest_grid_mask`. Structurally analogous to `_derive_offset_tables`
   but the exact logic needs to be worked out and validated numerically.

2. **`GET_LOST_FROM` generalisation** for `RingShift` with arbitrary subsets.

3. **Weight transfer**: `DataSpec` API change propagates to `weight_transfer.py` — needs
   to handle `base_pixels` list instead of `base_pix` int, but should be mechanical.

4. **Test matrix** for the new configurations: `base_pixels=list(range(12))`,
   `base_pixels=[8,9,10,11]`, and the 8-pixel case as regression.

5. **The full spec document** and implementation plan (to be written at start of next session).

---

## Recommended next session flow

1. Start fresh conversation, reference this handout
2. Write the spec doc (`2026-07-12-full-sphere-healswin-design.md` is the existing phase-1 spec;
   add a new `2026-07-12-full-sphere-extension-design.md` or append a new section)
3. Resolve the open items above (especially `_derive_mask_pairs`)
4. Invoke `writing-plans` to create the implementation plan
