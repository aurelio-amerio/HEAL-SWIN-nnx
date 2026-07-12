# Full-Sphere HEAL-SWIN Extension — Design Spec

**Date:** 2026-07-12
**Status:** Approved design, ready for implementation planning
**Supersedes:** `2026-07-12-full-sphere-healswin-handout.md` (working notes; its offset-table
derivation was re-verified numerically in this session and is correct — see §3.1)

## 1. Goal

Phase 2 of the NNX port: make HEAL-SWIN work on the **full sphere** (all 12 HEALPix base
pixels) and on **arbitrary user-specified subsets** of base pixels (e.g. `[8, 9, 10, 11]`
for a ground-based south-pole telescope). Full sphere is the default.

Delivery is staged:

1. Replace the hardcoded 8-base-pixel topology tables with **algorithmically derived**
   tables; verify zero regression against the reference constants and goldens.
2. Add an **exact geometric shift** as a new opt-in strategy (`"nest_grid_shift_exact"`)
   that connects the true boundary pixels across face seams. The approximate strategy is
   kept; it can be removed later if the exact one proves strictly better.

Design principle (user-set): *always derive the right numbers algorithmically, then verify
them empirically against independent ground truth (healpy).*

### 1.1 The central invariant: seam correctness

"Having the right pixels at the seam" is the crucial feature this phase exists to deliver:
**pixels adjacent in a window's canonical 2D layout must be adjacent on the sky.** Without
it, the relative-position bias is wrong at seams and any future window-content operator
(e.g. the parked 2D-conv window variant, `docs/notes/2026-07-12-conv-window-shift-variant.md`)
would see pixels in the wrong order. This invariant gets first-class, explicit test coverage
(§6.1), not incidental coverage.

## 2. API

### 2.1 `DataSpec` (config.py)

Add one field; keep everything else, including flat-Swin compatibility:

```python
@dataclass
class DataSpec:
    dim_in: Union[int, Tuple[int, int]]
    f_in: int
    f_out: int
    base_pix: Optional[int] = None           # legacy, kept
    base_pixels: Optional[List[int]] = None  # NEW: which HEALPix base pixels (faces)
    class_names: List[str] = field(default_factory=list)
```

Resolution rules (`__post_init__`):

- `base_pixels is None and base_pix is None` → full sphere: `base_pixels = list(range(12))`.
- `base_pixels is None and base_pix == k` (legacy callers, incl. all 8-pix fisheye tests)
  → `base_pixels = list(range(k))`. This matches the reference's implicit assumption
  (verified: the fisheye subset is faces 0–7).
- `base_pixels` given → validated: ints in `[0, 11]`, no duplicates, **strictly
  increasing** (model input layout is the concatenation of the selected faces' NEST
  pixels, so order must be canonical). If `base_pix` is also given it must equal
  `len(base_pixels)`.
- After init, `base_pix == len(base_pixels)` always holds, so existing model code reading
  `data_spec.base_pix` keeps working unchanged.
- For the HP model, `dim_in` is validated against `len(base_pixels) * nside**2`.

### 2.2 Config

`SwinHPTransformerConfig.shift_strategy` literal gains `"nest_grid_shift_exact"`.
No other config changes.

### 2.3 Module layout

New file `src/heal_swin_nnx/hp_topology.py`:

- The four topology matrices copied from Map2Patches (Amerio) with attribution, as plain
  numpy: `neighbours_matrix` (12×4, corner-neighbour faces in TL/BL/BR/TR order),
  `side_matrix` (which edge of each neighbour touches — the orientation data),
  `corner_faces_matrix` / `corner_sides_matrix` (corner adjacency; `-1` marks the 8
  pinch points where only 3 faces meet). No runtime dependency on Map2Patches.
- `pix2xyf` / `xyf2pix`: Morton (Z-order) decode/encode of the in-face NEST index,
  ~10 lines of numpy each. healpy is deliberately **not** used at construction time so
  that healpy remains an *independent* ground truth in tests (§6) — deriving and
  verifying with the same library would be circular.
- Derivation functions: `derive_offset_tables(base_pixels)`,
  `derive_mask_faces(base_pixels, nside, window_size)`,
  `derive_ring_lost_from(base_pixels)`, seam coordinate-transform helpers, and the exact
  shift index builder.

`hp_shifting.py` keeps all nnx wrapper classes and consumes `hp_topology`; the hardcoded
`NEST_GRID_*` and `RING_GET_LOST_FROM` constants are deleted. All computation stays
construction-time numpy stored in `Buffer`s; runtime remains pure gathers/rolls.

## 3. NestGridShift generalization (parity strategy)

### 3.1 Offset tables (verified)

```python
DIR1_COL = [1, 1, 1, 1,  0, 0, 0, 0,  2, 2, 2, 2]  # per-face neighbour column for dir1
DIR2_COL = [2, 2, 2, 2,  0, 0, 0, 0,  1, 1, 1, 1]  # south ring reflected vs north

def derive_offset_tables(base_pixels):
    n = len(base_pixels)
    pos = {f: i for i, f in enumerate(base_pixels)}
    off1, off2 = {}, {}
    for i, b in enumerate(base_pixels):
        nb1 = neighbours_matrix[b][DIR1_COL[b]]
        nb2 = neighbours_matrix[b][DIR2_COL[b]]
        off1[i] = (i - pos.get(nb1, i) - 1) % n  # missing neighbour → self → masked seam
        off2[i] = (i - pos.get(nb2, i) - 1) % n
    return off1, off2
```

Verified in this session:

- Reproduces the reference's hardcoded 8-pix tables exactly for `base_pixels=[0..7]`
  (DIR1 `{0:2, 1:2, 2:2, 3:6, 4:3, 5:3, 6:3, 7:3}`, DIR2 all 3).
- Full sphere: DIR1 `{0:6, 1:6, 2:6, 3:10, 4:3, 5:3, 6:3, 7:3, 8:8, 9:0, 10:0, 11:0}`,
  DIR2 `{0:7, 1:7, 2:7, 3:7, 4:3, 5:3, 6:3, 7:3, 8:10, 9:10, 10:10, 11:2}` — valid
  permutations for nside ∈ {16, 32} × ws ∈ {16, 64}.
- South cap `[8, 9, 10, 11]`: valid permutation (both dir neighbours are internal —
  the four south-polar faces close on themselves around the pole).

`nest_grid_shift_idcs` changes signature (`base_pixels` instead of `base_pix`); the
hierarchical traversal (`_get_scale`, `_get_offset_dir1/2`) is untouched.

### 3.2 Why masking exists (limitation, not bug)

Two distinct causes:

1. **NEST-order gluing vs. true geometry** (intrinsic to the 1D machinery, present even
   on the full sphere). The hierarchical shift glues the NEST-order *last* pixels of the
   source face against the NEST-order *first* pixels of the destination face. NEST
   ordering starts at a fixed local corner, so "first pixels" lie on two specific local
   edges — which may not be the edges that physically touch the source face. Example
   (verified): face 8's NEST-first pixels lie on its y=0 edge (physically bordering face
   9), but dir1 connects face 8 to face 11 (true seam: face 8's x=0 edge). The glued
   pixels come from perpendicular edges — a valid permutation, geographic nonsense. The
   pure-1D index arithmetic cannot rotate incoming content, so the reference masks such
   window halves apart. Whether a given seam aligns is per-seam luck, hence *derived*,
   never assumed.
2. **Fictitious seams at ROI borders** (subsets only): a face whose dir1/dir2 neighbour
   is outside `base_pixels` gets wrapped to some in-domain face by the offset formula;
   that seam is fake and always masked.

### 3.3 Mask derivation (`derive_mask_faces`)

Approach: keep the reference's mask *structure* (face-level `masked`/`carry_over` lists
feeding the existing `nest_grid_mask` recursion unchanged), decide *membership*
geometrically at construction time:

1. From the offset tables, enumerate each dir1/dir2 seam (which face's content is glued
   into which face's boundary windows).
2. For each seam, take the actual glued pixel pairs (readable off the shift indices; only
   ~nside edge pixels per seam matter) and check sky-adjacency using the topology matrices
   + Morton `pix2xyf`.
3. Seams gluing non-adjacent pixels put the receiving face in `masked`, paired with the
   face whose first quarter-window receives its carried-over content (`carry_over` — read
   off the shift indices, not guessed).
4. Faces with out-of-subset dir neighbours (cause 2) are always masked.

Acceptance: for `[0..7]` the derived lists must equal the reference constants
(`masked=[4,5,6,7]`, `carry_over=[0,1,2,3]`) — hard regression assert. For the full
sphere, the handout's prediction (`masked=[4..11]`, `carry=[0..7]`) is *checked* against
ground truth, not trusted. If derivation ever disagrees with the reference constants, the
8-pix case pins to the legacy constants and the discrepancy is documented.

### 3.4 RingShift

- Full sphere: drop the `NotImplementedError`; the ring roll is already a pure
  permutation, mask all-zero, backfill loops see zero holes.
- Subsets: the roll is always computed on the full 12-face ring ordering, then restricted.
  Holes (content from unselected faces) are masked and backfilled from the pool of
  selected pixels that rolled out of domain. Donor-face pairing (`GET_LOST_FROM`,
  currently hardcoded `{4:7, 5:4, 6:5, 7:6}`) is derived from the same-latitude
  ring-neighbour relation in `neighbours_matrix`, with the leftover-pool fallback exactly
  as the reference does for its carry faces. 8-pix behaviour must be bit-identical
  (regression golden).

## 4. Exact geometric shift (`NestGridShiftExact`)

Same interface as the other shifters (`shift`/`shift_back`/`attn_mask`, precomputed index
`Buffer`s, runtime `jnp.take`). Construction, per selected pixel:

1. Decode `(x, y, face)` with Morton `pix2xyf`.
2. The shift source is the pixel at `(x + s/2, y + s/2)` in face-local coordinates,
   `s = sqrt(window_size)`. The sign/axis convention is fixed by decoding what the
   approximate `dir1[dir2]` permutation does to one interior pixel, so the two strategies
   are **bit-identical away from face boundaries** (design invariant and test).
3. Off-face coordinates walk into the neighbour via `neighbours_matrix`, applying the
   rotation/reflection dictated by `side_matrix`; corner crossings use
   `corner_faces_matrix`/`corner_sides_matrix`.
4. Encode back with `xyf2pix`.

**The gather is the rotation.** The orientation transform is baked into the precomputed
indices, so each window slot holds the pixel at its *true* relative grid position even
across a rotated seam — the regrouping that "doing it properly" requires happens once at
construction, at zero runtime cost. Consequence: **the canonical relative-position bias
indices are geometrically correct across seams** (the seam transform is an isometry of
the pixel grid). No per-window bias tables are needed; this property is asserted by test
(§6.1), since it is now claimed rather than disclaimed.

Where exactness ends — two unavoidable holes, one shared backfill mechanism:

- **Pinch corners** (full sphere): corner crossings where `corner_faces_matrix == -1` —
  the diagonal neighbour genuinely does not exist (8 points where 3 faces meet). Each
  affected window is missing up to `(s/2)**2` source pixels.
- **ROI borders** (subsets): the source face is not in `base_pixels`.

Orphaned window slots are backfilled from the pool of lost pixels (pixels whose
destination fell into a hole in the other direction; counts must match, asserted) and
masked, RingShift-backfill style. All other windows carry **no mask** — on the full
sphere the attention mask is nonzero only at pinch-corner windows. Mask construction is
geometric labelling on just the affected windows: pixels grouped by sky-contiguity,
backfilled slots get a unique label.

**Efficiency.** All geometry is construction-time; the runtime op is the identical single
`jnp.take` gather the approximate strategy uses — no padding tensor, no scatter, zero
extra runtime cost (this is what distinguishes it from Map2Patches' `PatchPadding`, which
pays a gather/scatter per forward pass). Construction must be **vectorized numpy over all
pixels** (no per-pixel Python loop à la Map2Patches' `get_diag_indices`): the shift is
`s/2 <= nside`, so each pixel crosses at most one face boundary — compute `x+d, y+d` for
all pixels at once, classify into 9 cases (interior / 4 edges / 4 corners), apply each
case's fixed transform as a batched array op. O(npix) once per resolution stage
(~milliseconds at nside=256 full sphere).

Residual approximations (documented, not fixable by indexing):

- Pinch-corner windows (masked).
- Metric distortion: HEALPix face grids are not isometric to the sky (pixels shear near
  seams and poles), so a grid offset is not a constant angular displacement. This is
  inherent to HEAL-SWIN everywhere, including face interiors and the reference; the exact
  shift neither fixes nor worsens it.

## 5. Propagation

- `swin_hp_transformer.py`: thread `base_pixels` to shifter constructors; add the
  `nest_grid_shift_exact` branch; nside computed from
  `input_resolution // len(base_pixels)` as today.
- `weight_transfer.py`: accepts `base_pixels`; torch reference checkpoints are always
  8-pix, so transfer validates `base_pixels == list(range(8))` and otherwise proceeds
  unchanged.
- README: short "full sphere & partial coverage" section with the south-pole example.

## 6. Validation & test matrix

All construction-time numpy — fast, CPU-only, healpy as independent ground truth.

### 6.1 Seam correctness (headline invariant, own module `test_seam_geometry.py`)

For every window straddling a face seam after an exact shift:

1. **Slot-adjacency ↔ sky-adjacency**: reconstruct the window's slot grid via the
   `nest_relative_position_index` coordinate map; for every horizontally/vertically
   adjacent slot pair, assert the two source pixels are mutual neighbours per
   `hp.get_all_neighbours`.
2. **Rel-pos bias consistency**: for each in-window pair, recompute the true grid offset
   by walking the sphere with healpy (`hp.pix2xyf` + seam transforms) and assert it
   equals the canonical offset the bias table uses. Masked (backfilled/pinch) slots
   exempt.

Run for full sphere and south cap.

### 6.2 Full matrix

| Test | Configs |
|---|---|
| Permutation validity (all strategies) | `base_pixels` ∈ {full sphere, `[0..7]`, `[8..11]`, non-contiguous e.g. `[0,4,8]`} × nside ∈ {8, 16} × ws ∈ {16, 64} |
| 8-pix parity regression | derived offsets, masks, RingShift indices bit-equal to reference constants/goldens for `[0..7]` |
| Mask ground truth (parity strategy) | unmasked same-region pairs in boundary windows are sky-adjacent; wrongly-glued content is cross-region |
| Seam correctness (exact strategy) | §6.1, full sphere + south cap |
| Interior equality | exact vs. approximate shift indices identical away from face boundaries |
| Backfill accounting | lost-pixel count == hole count (analytic pinch-corner formula on full sphere; ROI borders on subsets) |
| DataSpec validation | legacy `base_pix` resolution, full-sphere default, rejection of duplicates / out-of-range / unsorted |

### 6.3 Fallback

If the mask derivation (§3.3) proves intractable for some subset topology, the documented
fallback is pixel-level exactness (§4) as the only strategy for that configuration — i.e.
the approximate strategy raises for that subset with a pointer to
`nest_grid_shift_exact`. Hardcoding validated constants for {8-pix, 12-pix, south cap}
(the known-good cases) is the fallback-of-last-resort.

## 7. Out of scope

- The conv-window-shift variant (parked: `docs/notes/2026-07-12-conv-window-shift-variant.md`);
  this phase's seam invariant is a prerequisite it will inherit.
- Per-window bias tables (unnecessary per §4) beyond the consistency test.
- Training-level smoke tests (geometric ground-truth bar chosen instead).
- Data-pipeline work.
