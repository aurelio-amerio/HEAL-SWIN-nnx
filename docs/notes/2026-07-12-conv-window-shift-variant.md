# Handout: a convolutional variant of HEAL-SWIN (keep the window-shift, drop the attention)

**Status:** idea capture only — not designed, not scheduled, not implemented.
**Date:** 2026-07-12
**One line:** Reuse HEAL-SWIN's HEALPix window-partition + shift machinery to traverse the
sphere, but replace the in-window self-attention with a 2D convolution. A lighter mixer for
simpler problems, keeping the principled spherical traversal.

---

## The core idea

HEAL-SWIN's value is **not** the attention itself — it's the traversal machinery around it.
Attention is just one interchangeable "token mixer" sitting inside that machinery. Swap it for a
2D conv and you get a cheaper model that still moves around the sphere in a principled way.

Per-block pipeline becomes:

```
window-partition (flat → √ws×√ws grid)  →  2D conv  →  flatten back  →  shift  →  repeat
```

instead of `... → window self-attention → ...`.

## Why it actually works (and isn't a hack)

- The sphere is stored as a flat 1D sequence of HEALPix pixels in **nested** order. A contiguous
  chunk of `window_size` pixels (a power of 4) is a spatially-coherent √ws × √ws **square patch**
  — a HEALPix quadtree subtree.
- You **cannot** just slide a plain conv over the nested 1D sequence: it's a space-filling curve
  with discontinuities at quadtree seams and base-pixel boundaries. A naive 1D/2D conv over the
  raw sequence would mix across those seams incorrectly.
- The **windowing** is exactly what carves out locally-Euclidean patches where a 2D kernel is
  meaningful. The **shift** is what stitches those patches across the sphere's topology between
  blocks, growing the receptive field over depth — same role it plays in Swin.

So the windowing+shift is not incidental scaffolding you're borrowing; it's precisely the part
that makes a 2D conv well-defined on the sphere. The instinct is correct.

## Precedent

- **In this very reference repo:** `references/HEAL-SWIN/heal_swin/models_torch/swin_mlp.py`
  already replaces in-window attention with an MLP. Swapping in a conv is the same move.
- **In the literature:** MetaFormer / PoolFormer / ConvNeXt — the token mixer is replaceable
  (pooling, MLP, depthwise conv) while the macro-architecture carries the performance. A
  depthwise conv for spatial mixing + the existing `Mlp` for channel mixing is literally the
  ConvNeXt/MetaFormer recipe.

## Concrete mechanism (with the exact hooks already in the port)

All the pieces needed to reshape a flat window into a real 2D grid already exist:

- `src/heal_swin_nnx/hp_windowing.py:26` — `get_nest_win_idcs(window_size)` returns the
  `(s, s)` grid mapping each Cartesian position → its nested-scheme index inside one window.
  This is the flat↔grid bridge for the conv.
- `src/heal_swin_nnx/hp_windowing.py:12` — `window_partition` / `window_reverse` reshape
  `(B, N, C)` ↔ `(B·nW, ws, C)`.
- `src/heal_swin_nnx/hp_shifting.py` — `NoShift`, `NestRollShift`, `NestGridShift`, `RingShift`;
  each exposes `.shift(x)`, `.shift_back(x)`, `.attn_mask`. Reuse verbatim — the conv variant
  keeps `shift` / `shift_back`, only the mixer between them changes.
- Integration point today: `SwinTransformerBlock.__call__` in
  `src/heal_swin_nnx/swin_hp_transformer.py:160` — the block already does
  `shift → window_partition → attn → window_reverse → shift_back`. A `ConvBlock` would keep that
  exact skeleton and substitute the middle three lines.

Sketch of the mixer body (replacing `WindowAttention.__call__`):

```
# x_windows: (B_, ws, C), nested order
grid = x_windows[:, nest_win_idcs, :]        # (B_, s, s, C) Cartesian via get_nest_win_idcs
grid = depthwise_conv2d(grid)                # (B_, s, s, C)  spatial mixing
x_windows = grid.reshape(B_, ws, C)[:, inv]  # flatten back to nested order (inverse perm)
```

`nest_relative_position_index` / the relative-position-bias table drop out entirely — a conv
carries its own spatial structure.

## The one real wrinkle to resolve later

In **shifted** blocks, a window holds a *mix* of regions that got wrapped in from non-adjacent
parts of the sphere. Attention handles this with an additive `attn_mask` (0 / −100) that forbids
cross-region mixing. A conv blends spatial neighbors directly and **can't be masked the same
way**. Options to weigh when the idea is picked up:

1. **Conv only in non-shifted blocks**, and let the shift act purely as a re-windowing so the
   *next* conv sees the re-tiled grid. Simplest; still gets cross-window flow over depth.
2. **Multiply by a per-window validity mask** derived from `attn_mask` before/after the conv, so
   contributions from foreign regions are zeroed (a masked / partial convolution). More faithful,
   more bookkeeping.
3. **Accept minor boundary leakage** at shifted windows. Cheapest; may be fine for the "simpler
   problem" target.

## Design choices deferred (decide when picked up)

- **Conv type:** depthwise (spatial-only, cheapest, MetaFormer-style) vs depthwise-separable vs
  full 2D conv. Depthwise + existing channel `Mlp` is the natural analog of Swin's structure.
- **Kernel size vs window size:** kernel = window ⇒ full-window mixing (closest to attention's
  in-window globality); kernel < window ⇒ local. Note windows are small (ws=4 ⇒ s=2, ws=16 ⇒
  s=4), so a 3×3 kernel already spans most of a small window.
- **Padding at window edges:** interacts with the wrinkle above.
- **Where it lives:** config flag `mixer='attn'|'conv'` on a shared block (fair A/B, drop-in) vs a
  separate slimmer model that imports only `hp_shifting`/`hp_windowing`. See open question.

## Open scoping question (unanswered — first thing to settle on resume)

What is this variant *for*? The answer steers the whole design:

- **Lightweight drop-in** — config-selectable mixer on the existing `SwinHPTransformer` stack,
  for simpler/smaller tasks where attention is overkill.
- **Research comparison** — clean A/B of conv-mixing vs attention-mixing on the same data; shared
  scaffolding matters for a fair comparison.
- **New minimal model** — fresh stripped-down spherical-conv net reusing only
  `hp_shifting`/`hp_windowing`, not the full transformer stack.
- **Full-sphere / cosmology (phase 2)** — a lighter whole-sphere model as the actual target
  (ties into the base_pix=12 extension; see the full-sphere cosmology goal memory).

## Relationship to other work

- Independent of the float64 golden-parity effort — this is a *new* architecture, so it carries
  no bit-parity constraint with the torch reference. Design freedom.
- If aimed at full-sphere, it inherits the same base_pix=12 topology-table gap that
  `hp_shifting.py` guards with `NotImplementedError` (phase 2).
