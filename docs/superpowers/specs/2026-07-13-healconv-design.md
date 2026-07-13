# HealConv: a convolutional HEAL-SWIN variant (design)

**Date:** 2026-07-13
**Status:** approved design, ready for implementation planning
**Origin:** `docs/notes/2026-07-12-conv-window-shift-variant.md` (idea capture), refined
through brainstorming.

## 1. Summary and scope

A separate, minimal spherical model that reuses HEAL-SWIN's HEALPix traversal machinery
(window partition + shift strategies) but replaces in-window self-attention with a
depthwise 2D convolution. Rationale (from the note): the windowing carves out
locally-Euclidean √ws×√ws patches where a 2D kernel is well-defined, and the shift
stitches patches across the sphere's topology between blocks — the traversal is the
value; attention is an interchangeable mixer.

Decisions made during brainstorming:

- **Purpose:** separate minimal model (`HealConv`), not a config flag on `HealSwin`.
- **Macro-architecture:** U-Net, mirroring HealSwin (PatchMerging encoder, PatchExpand
  decoder with skip connections). Same I/O contract: channels-last `(B, npix, C)` in
  NEST order over `base_pixels`.
- **One knob:** user picks `kernel_size` k; `window_size = k²`. The depthwise kernel
  spans the whole window (SAME zero-padding), the closest conv analog of attention's
  in-window globality.
- **Block recipe:** depthwise conv (spatial mixing) + shared `Mlp` (channel mixing) —
  the ConvNeXt/MetaFormer/MBConv factorization, structurally identical to
  `HealSwinBlock` with attention swapped out. Channel growth happens only at
  PatchEmbed and stage transitions, exactly as in HealSwin.
- **Shifted-window masking:** zero-in + zero-update-out. Foreign (wrapped-in) pixels
  are zeroed before the conv so they contribute nothing, and their conv update is
  zeroed after, so they pass through the block's residual untouched and get mixed in
  the next unshifted block. Derived from the shift strategies' raw region masks.
- **Code organization:** extract the mixer-agnostic U-Net plumbing from
  `models/healswin.py` into `layers.py`; `healconv.py` never imports from
  `healswin.py`.

Out of scope: torch parity (this is a new architecture; no reference exists),
positional embeddings (a conv carries its own spatial structure — `rel_bias` / RoPE
drop out entirely), attention-specific config (`num_heads`, `qkv_bias`,
`attn_drop_rate`).

## 2. Public API

`models/healconv.py`, exported from `heal_swin_nnx.__init__`:

- `HealConvParams` — pure-data dataclass, JSON-serializable via `dataclasses.asdict`.
- `HealConvEncoder` — patch embed + conv encoder stages + final norm; standalone-usable.
- `HealConvDecoder` — U-Net decoder head producing dense per-pixel outputs.
- `HealConv` — encoder + decoder.

### HealConvParams

```python
@dataclass
class HealConvParams:
    # data / geometry
    nside: int
    in_channels: int
    out_channels: int
    base_pixels: Optional[Sequence[int]] = None   # None -> full sphere (0..11)

    # architecture
    patch_size: int = 4
    kernel_size: int = 4        # THE knob: k×k depthwise kernel; window = k² pixels
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 2, 2)
    mlp_ratio: float = 4.0
    conv_bias: bool = True
    patch_embed_norm: bool = False
    shift_strategy: Literal["nest_roll", "nest_grid_shift",
                            "nest_grid_shift_exact", "ring_shift"] = "nest_grid_shift_exact"

    # regularization / training
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    use_checkpoint: bool = False

    @property
    def window_size(self): return self.kernel_size ** 2
    @property
    def shift_size(self): return self.window_size // 2
    @property
    def npix(self): return len(self.base_pixels) * self.nside ** 2
```

`__post_init__` validation mirrors `HealSwinParams` minus the attention rules:

- `base_pixels` in [0, 11], strictly increasing; `None` → `tuple(range(12))`.
- `shift_strategy` in the four-strategy enum.
- `patch_size` positive multiple of 4; `nside` a power of two; `nside² % patch_size == 0`;
  `(nside²/patch_size) % 4^(n_stages−1) == 0`.
- The `nest_grid_shift` bottleneck constraint (deepest stage must hold a full window),
  identical to HealSwin's.
- **New:** `kernel_size` must be a power of two, ≥ 2 (the window is a nested quadtree
  square, so `window_size = k²` must be a power of four). Consequence of the one-knob
  coupling: k ∈ {2, 4, 8, …}; a 3×3 kernel is not expressible. Error message must state
  this.

Because `window_size` is a derived property, all downstream machinery (shifters,
windowing) consumes `params.window_size` exactly as it does for HealSwin.

## 3. The conv block

`HealConvBlock` keeps `HealSwinBlock`'s skeleton verbatim — same shifter construction
(`window_size = min(params.window_size, input_resolution)`, `shift_size = 0` unless
shifted and resolution allows), same Swin-V2-style post-norm residual wiring — and
swaps only the mixer:

```python
def __call__(self, x):                        # (B, N, C)
    shortcut = x
    x = self.shifter.shift(x)
    w = window_partition(x, ws)               # (B·nW, ws, C), nested order
    if self.validity is not None:             # shifted blocks only
        w = w * validity                      # zero foreign pixels IN
    g = w[:, self.grid_perm, :]               # gather -> (B·nW, k, k, C) Cartesian grid
    g = self.dwconv(g)                        # depthwise k×k, SAME padding, C -> C
    w = g.reshape(-1, ws, C)[:, self.inv_perm, :]   # back to nested order
    if self.validity is not None:
        w = w * validity                      # zero foreign updates OUT
    x = window_reverse(w, ws, N)
    x = self.shifter.shift_back(x)
    x = shortcut + self.drop_path(self.norm1(x))   # foreign px: pure residual pass-through
    return x + self.drop_path(self.norm2(self.mlp(x)))
```

Construction-time state (all numpy at build, stored as `Buffer` — never `nnx.Param` —
per the codebase invariant that the traced forward only does gathers and dense math):

- `grid_perm`: flat (ws,) gather indices derived from `get_nest_win_idcs(ws)` mapping
  nested order → Cartesian (k, k) grid; `inv_perm` its argsort.
- `validity`: `(nW, ws, 1)` float32 0/1, present only when `shift_size > 0`.
  Broadcast over batch at runtime (reshape `(B·nW, ws, C)` ↔ `(B, nW, ws, C)` or tile).
  `NoShift` blocks carry `validity = None` — zero masking cost unshifted.

The mixer is `nnx.Conv(C, C, kernel_size=(k, k), feature_group_count=C,
padding='SAME', use_bias=params.conv_bias)`. SAME zero-padding at window edges is
consistent with zeroed foreign pixels and is the accepted boundary behavior.

### Validity mask semantics

For each shifted window, the shift strategy's **raw region mask** labels every pixel
with the region it was wrapped in from. The window's *dominant region* is the majority
label (ties broken deterministically: lowest label wins). Pixels of the dominant
region are valid (1); all others are foreign (0). Zero-in prevents foreign content
from leaking into dominant-region outputs; zero-out means foreign pixels receive no
update in this block (residual pass-through) and are mixed correctly in the next
unshifted block instead. Dominant-region pixels adjacent to zeroed foreign pixels see
zeros — the same boundary condition as SAME padding at window edges.

## 4. Changes to existing modules

### layers.py (shared building blocks — new home for U-Net plumbing)

Moved verbatim from `models/healswin.py`:

- `PatchMerging`, `PatchExpand`, `FinalPatchExpand` — pure code motion, signatures
  already generic.
- `PatchEmbed` — moved with signature changed from `PatchEmbed(params)` to explicit
  arguments `PatchEmbed(npix, patch_size, in_channels, embed_dim, norm, *, rngs)` so
  the shared component depends on neither params class. Behavior unchanged.

### models/healswin.py

Imports the four components from `layers`; adapts the single `PatchEmbed(...)` call
site. No pytree/state changes: NNX state paths are attribute paths, so existing
checkpoints are unaffected.

### hp/shifting.py (construction-time numpy additions)

- `nest_roll_raw_mask(input_resolution, window_size, shift_size)` — the region-labeling
  logic extracted from `nest_roll_mask`, which becomes a thin wrapper
  (`get_attn_mask_from_mask(nest_roll_raw_mask(...), window_size)`), bit-identical.
  The grid/ring strategies already expose raw masks (`nest_grid_mask`,
  `exact_shift_idcs_and_mask`, `ring_shift_idcs_and_mask`).
- `validity_from_mask(raw_mask, window_size) -> (nW, ws) float32` — windows the raw
  per-pixel mask, majority label per window (deterministic tie-break), 1 for majority
  pixels, 0 otherwise.

Existing shifter classes are untouched; `HealConvBlock` constructs shifters exactly as
`HealSwinBlock` does and derives `validity` from the same raw-mask functions.

### models/healconv.py (new)

- `HealConvParams`, `HealConvBlock` (above).
- `ConvEncoderStage` / `ConvDecoderStage` — same shape as the existing stages (block
  loop with optional `nnx.remat`, then PatchMerging / PatchExpand). Kept local rather
  than genericized over block type: sharing would mean threading block factories
  through for ~30 lines.
- `HealConvEncoder` / `HealConvDecoder` / `HealConv` — mirror the HealSwin trio:
  - Encoder: `PatchEmbed` (in_channels → embed_dim, npix → npix/patch_size) →
    pos-dropout → stages with dim `embed_dim·2^i` and resolution `num_patches/4^i`,
    PatchMerging between stages → final LayerNorm; returns `(tokens, skips)`.
  - Decoder: PatchExpand bottleneck, then per stage: skip concat + linear reduction →
    conv blocks → PatchExpand; final norm, `FinalPatchExpand(patch_size)`, 1×1 output
    conv to `out_channels`.
  - Drop-path schedule: linear 0 → `drop_path_rate` over `sum(depths)`, as in HealSwin
    (small helper duplicated locally).

### heal_swin_nnx/__init__.py

Add `HealConv`, `HealConvEncoder`, `HealConvDecoder`, `HealConvParams` to imports and
`__all__`.

### docs/notes/2026-07-12-conv-window-shift-variant.md

Status line updated to point at this spec.

Dependency direction: `models/healconv.py` → `layers`, `hp/shifting`, `hp/windowing`,
`variables`. It never imports from `models/healswin.py`.

## 5. Error handling

All configuration errors are raised eagerly in `HealConvParams.__post_init__` with
actionable messages (matching the house style: state the rule, the offending value,
and the remedy). Construction keeps HealSwin's per-stage assertion that nside stays an
integer at every layer. No new runtime (traced) error paths.

## 6. Testing

House philosophy: independent ground truth + properties/invariants, no golden values.

**Refactor safety:** the existing suite passing unchanged validates the `layers.py`
move and the `nest_roll_mask` split (wrapper must be bit-identical).

**New tests:**

1. **Grid-perm round-trip** — `grid_perm`/`inv_perm` compose to identity for each
   valid k; `grid_perm` agrees with `get_nest_win_idcs` (itself healpy-verified).
2. **Identity-kernel property** — with the depthwise kernel set to a delta (single
   tap 1 at the zero-spatial-offset position — note k is even, so this is not a
   "center" tap; the test derives the tap index from SAME-padding asymmetry — and
   zero bias), the full shift→window→grid→conv→ungrid→unwindow→unshift sandwich is
   an exact identity in unshifted blocks. One assertion catches any
   permutation/reshape/padding bug.
3. **Cross-region independence (shifted blocks)** — perturbing foreign pixels of a
   window leaves dominant-region outputs unchanged; foreign-pixel outputs equal the
   pure residual path regardless of perturbation. Conv analog of the attention-mask
   tests; verifies zero-in/zero-out directly.
4. **Validity/attn-mask consistency** — for all four shift strategies:
   `validity_from_mask` marks a pixel 0 iff the existing pairwise `attn_mask` assigns
   −100 between it and its window's majority pixels.
5. **Model-level invariants** (mirroring HealSwin's model tests) — forward shape
   `(B, npix, in) → (B, npix, out)` for full sphere and partial sky; jit/eager
   equivalence; batch independence; `use_checkpoint` (remat) equivalence; Buffers
   excluded from `nnx.grad`/optimizer state.
6. **Params validation** (`test_params.py` style) — invalid `kernel_size` (0, 3,
   non-power-of-2), inherited nside/patch/stage/bottleneck rules, `window_size` and
   `shift_size` derivation.
