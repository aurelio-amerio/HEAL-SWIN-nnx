# HEAL-SWIN → Flax NNX Port — Design

**Date:** 2026-07-12
**Status:** Draft for review

## Goal

Port the HEAL-SWIN model (and the flat SWIN-UNet baseline) from the reference
PyTorch implementation at `references/HEAL-SWIN` to pure JAX / Flax NNX, with
numerical parity verified against golden values generated from the original
code. The training/evaluation pipeline (datasets, PyTorch Lightning, MLflow,
run configs) is explicitly **out of scope** — only the model definitions are
ported.

## Scope

**In scope:**

- `SwinHPTransformerSys` (HEAL-SWIN on the HEALPix sphere) — full option
  surface: all three shift strategies (`nest_roll`, `nest_grid_shift`,
  `ring_shift`), SwinV2 cosine attention, v2 norm placement, flat relative
  position bias, absolute position embedding, gradient checkpointing.
- `SwinTransformerSys` (flat 2D SWIN-UNet baseline) — same fidelity.
- A standalone `SwinHPEncoder` (and flat counterpart): the compression-only
  backbone, usable without any decoder (tokenizer/embedder/property-inference
  use cases).
- A parity harness: golden fixtures (outputs **and** gradients) generated once
  from the reference implementation in a pinned legacy environment, committed
  to the test suite.

**Out of scope:**

- `swin_mlp.py` (dead code in the reference — nothing imports it).
- Datasets, dataloaders, Lightning wrappers, MLflow, run configs, evaluation
  writers, losses, optimizers.
- Alternative heads beyond `UnetDecoder` (pooling/classification heads,
  diffusion-conditioning adapters). The encoder/head seam is designed for
  them, but they are later one-file additions.

## Key decisions (agreed during brainstorming)

1. **Mirror the reference structure closely** in module names, nesting, and
   config fields, so the torch `state_dict` → nnx weight mapping is mechanical
   and parity debugging is a side-by-side diff. An idiomatic API redesign is a
   possible second pass, cheap once goldens are green.
2. **Free streamlining in pass 1** (no behavior change for any buildable
   reference config): drop `flops()`, `dev_mode` prints, `extra_repr`,
   torchscript hooks; deduplicate `Mlp`/`WindowAttention`/block code shared
   between flat and HP models; plain serializable config dataclasses (no live
   class references); drop the never-used `decoder_class` hook and its dead
   branches (`Literal[UnetDecoder]` forbids other values; both `else` branches
   are unreachable in the reference).
3. **Encoder/head split as a first-class seam:** `SwinHPEncoder` is a public
   standalone module returning `(tokens, skips)`; `SwinHPTransformerSys` is a
   thin composition `UnetDecoder(SwinHPEncoder(x))`. The final encoder
   `LayerNorm` (reference `forward_features`'s `self.norm`) belongs to the
   encoder. Instantiating the encoder alone allocates no decoder parameters.
4. **Golden-value parity, forward + gradients:** the torch implementation is
   run once per golden-set generation; fixtures live in the repo. The main
   test suite is pure JAX and never imports torch.
5. **Bottom-up parity (Approach A):** per-module goldens (block-boundary
   intermediates) plus end-to-end, so divergence is localized immediately.
   Integer index machinery is compared bit-exactly, floats by tolerance.
6. **einops is allowed** as a main dependency (common in JAX too).

## Repository layout

```
HEAL-SWIN-nnx/
├── pyproject.toml              # jax, flax, healpy, numpy, einops (+ pytest dev dep)
├── src/heal_swin_nnx/
│   ├── config.py               # SwinHPTransformerConfig, SwinTransformerConfig, DataSpec
│   ├── layers.py               # shared leaves: Mlp, DropPath, WindowAttention,
│   │                           #   PatchMerging, PatchExpand, FinalPatchExpand_X4
│   ├── hp_windowing.py         # window_partition/reverse (jnp), get_nest_win_idcs (numpy)
│   ├── hp_shifting.py          # NoShift/NestRollShift/NestGridShift/RingShift
│   ├── swin_hp_transformer.py  # HP blocks, BasicLayer(_up), SwinHPEncoder,
│   │                           #   UnetDecoder, SwinHPTransformerSys
│   ├── swin_transformer.py     # flat 2D mirror (encoder, decoder, SwinTransformerSys)
│   └── weight_transfer.py      # torch-state-npz → nnx state mapping
├── parity/                     # separate uv project, legacy torch env
│   ├── pyproject.toml          # python >=3.8,<3.9; pinned legacy deps
│   └── generate_goldens.py     # builds reference models, dumps fixtures → npz
└── tests/
    ├── goldens/                # committed .npz fixtures (tiny configs)
    ├── test_parity_modules.py  # per-module / block-boundary golden checks
    ├── test_parity_e2e.py      # end-to-end forward + gradient checks
    └── test_model.py           # JAX-native tests (shapes, jit, vmap, stochastic paths)
```

The reference repo stays untouched under `references/HEAL-SWIN`;
`generate_goldens.py` imports its model files via `sys.path` injection rather
than copying them, so goldens always come from the genuine reference code.

## Parity environment (`parity/`)

The reference pins matter: goldens must come from the exact library versions
the paper used.

- `requires-python = ">=3.8,<3.9"`; uv provisions CPython 3.8 itself.
- Dependencies (only what `models_torch` transitively imports):
  `torch==1.8.0`, `torchvision==0.9.0` (timm requirement), `timm==0.4.12`,
  `einops==0.4.0`, `healpy==1.15.2`, `numpy==1.19.2`.
- **CPU-only torch**, installed per the uv PyTorch guide: a
  `[[tool.uv.index]]` entry for `https://download.pytorch.org/whl/cpu` with
  `torch`/`torchvision` pinned to that index via `[tool.uv.sources]`
  (`1.8.0+cpu` / `0.9.0+cpu` builds).
- The rest of the reference's `setup.py` (mlflow, lightning, astropy,
  chamfer-distance, opencv, …) serves the training pipeline and is not
  installed.
- `.npz` files written under numpy 1.19 load fine in modern numpy.

## Model & module design (nnx side)

All modules are `nnx.Module`s mirroring the torch tree 1:1 in names and
nesting: `PatchEmbed`, `Mlp`, `WindowAttention`, `SwinTransformerBlock`,
`PatchMerging`, `PatchExpand`, `FinalPatchExpand_X4`, `BasicLayer`,
`BasicLayer_up`, `UnetDecoder`, `SwinHPEncoder`, `SwinHPTransformerSys` (plus
flat counterparts).

- **Data layout:** model-internal layout is `(B, N, C)` exactly as in torch,
  so reshapes/rolls/gathers translate literally. The public nnx API is
  channels-last end to end: inputs `(B, N, C_in)`, outputs `(B, N, f_out)`.
  `PatchEmbed` and the output projection use `nnx.Conv` (NWC-native). The
  parity harness transposes torch tensors at the boundary.
- **Parity-critical constants:** `nnx.LayerNorm(epsilon=1e-5)` everywhere
  (torch default; flax default is 1e-6); exact-erf GELU (`approximate=False`;
  torch 1.8 `nn.GELU` is erf-based); qkv reshape order copied verbatim.
- **Buffers:** static precomputes (`relative_position_index`, shift index
  arrays, attention masks) are computed with numpy at construction and stored
  in a custom variable type `class Buffer(nnx.Variable)`. Optimizers filter on
  `nnx.Param` (`wrt=nnx.Param`), so `Buffer`s are never updated or
  differentiated, yet ride through `split`/`merge`/`jit` as pytree state.
- **Stochastic bits:** `nnx.Dropout` plus a small `DropPath` module using
  `nnx.Rngs`; both identity in eval mode. `use_checkpoint` maps to
  `nnx.remat` on the block.
- **Init:** mirrors the reference distributionally (`trunc_normal_(std=0.02)`
  for Linear kernels, zeros for biases, ones/zeros for LayerNorm). Parity
  tests use transferred weights, not init; init parity is distributional only.
- **Config:** plain dataclasses with the reference's field names but
  serializable values (string literals instead of class references;
  `decoder_class` removed). `DataSpec` ported as a minimal dataclass
  (`dim_in`, `f_in`, `f_out`, `base_pix`, `class_names`).

### Encoder/head seam

```python
encoder = SwinHPEncoder(config, data_spec, rngs=rngs)
tokens, skips = encoder(x)   # tokens: (B, N/(patch_size·4^(L-1)), embed_dim·2^(L-1))

model = SwinHPTransformerSys(config, data_spec, rngs=rngs)  # encoder + UnetDecoder
y = model(x)                 # (B, N, f_out) — full input resolution
```

Heads that don't need `skips` ignore them. Future heads (global pooling +
linear for classification/property regression, diffusion-conditioning
adapters over the token pyramid) attach at this seam without touching the
parity-frozen backbone.

## Windowing & shifter port

- `window_partition` / `window_reverse`: pure `jnp.reshape` (1D sequence).
- `get_nest_win_idcs`: stays numpy (recursive quadrant fill), construction
  time only.
- `NoShift` / `NestRollShift`: identity and `jnp.roll(x, ∓shift_size,
  axis=1)`; roll mask built in numpy exactly as the reference does.
- `NestGridShift` / `RingShift`: the full index machinery
  (`_get_offset_dir1/2`, `_get_shifted_idcs_*`, ring↔nest via healpy, mask
  construction, inverse maps via argsort) ported to pure numpy with verbatim
  logic, run once at construction. Shifters store `shift_idcs`,
  `back_shift_idcs`, and the attention mask as `Buffer`s; runtime shift is
  `jnp.take(x, idcs, axis=1)`.
- `get_attn_mask_from_mask`: numpy; result stored as a `(nW, ws, ws)` float
  `Buffer`, added inside attention exactly as in torch.
- The reference's `_validate_shift_result` permutation asserts are kept as
  construction-time checks.
- **Index validation is bit-exact:** the golden generator dumps every index
  array and mask for each (nside, window_size, shifter) combination in the
  test matrix; tests assert integer equality (and exact equality for the
  0/−100 masks). No float tolerance for index logic.

## Parity harness & golden fixtures

### Golden generator (`parity/generate_goldens.py`)

Runs under Python 3.8 / torch 1.8.0. For each case in the config matrix:

- Tiny model sizes to keep fixtures small and committable. HP cases:
  `base_pix=8`, `nside=16` (N=2048 pixels), `patch_size=4`, `window_size=4`,
  `embed_dim=12`, `depths=[2,2]`, `num_heads=[2,4]`, `f_in=3`, `f_out=5`.
  Flat cases: `img_size=(32,64)` at comparable widths. Exact values may be
  adjusted during implementation if a constraint requires it (they are
  recorded in the fixture schema, so tests never hardcode them).
- Matrix covers: both models × {defaults, each of the three shifters,
  cos-attn, v2 norm placement, flat rel-pos-bias, APE}.
- Procedure per case: seed torch → build model in **eval mode** (dropout /
  droppath off; torch and JAX RNGs cannot be matched) → fixed random input →
  forward → scalar loss (`output.sum()`) → backward.
- Saved per case in one `.npz`: full `state_dict`, input, output, input
  gradient, all parameter gradients, per-module intermediate activations at
  block boundaries (captured with forward hooks), the encoder boundary
  `(tokens, skips)`, and all shifter index arrays / masks.

### Weight transfer (`weight_transfer.py`)

- Maps `state_dict` keys → nnx paths mechanically (mirrored tree).
- Transforms: `nn.Linear` weight `(out,in)→(in,out)` transpose; `Conv1d`
  kernel `(out,in,k)→(k,in,out)`; LayerNorm passes through.
- Completeness check: every torch key consumed, every nnx `Param` assigned.

### Tests

- `test_parity_modules.py`: walks block-boundary intermediates against
  goldens; bit-exact index/mask comparison for shifters.
- `test_parity_e2e.py`: final output plus gradients — JAX side recomputes the
  same scalar loss under `jax.grad` for input and parameters; parameter grads
  compared through the same transpose mapping.
- `test_model.py` (JAX-native, no goldens): jit compilation, output shapes
  across configs, vmap batch independence, dropout/droppath statistical
  checks in train mode, assertion that `Buffer`s are excluded from the
  `nnx.Param` filter.
- Tolerances: float32; per-module ≈ `atol=1e-6, rtol=1e-5`; end-to-end ≈
  `1e-4`; integer arrays exact.

## Error handling

- Construction-time asserts mirror the reference (window size power of two,
  `nside` integrality per layer, `patch_size % 4 == 0`, shift permutation
  validity, `NestGridShift` requires `base_pix == 8`).
- Weight transfer fails loudly on unconsumed/unassigned keys or shape
  mismatches.
- Golden loading asserts fixture-schema version so stale fixtures cannot
  silently pass.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| torch 1.8.0+cpu wheels unavailable for cp38 | uv PyTorch CPU index (documented builds exist); worst case: generate goldens in a torch 1.8 docker image once |
| LayerNorm eps / GELU variant mismatch | pinned explicitly in design; per-module goldens localize any residual mismatch |
| healpy version drift changes ring↔nest maps | index arrays are part of the goldens and compared bit-exactly; parity env pins healpy 1.15.2 |
| qkv head-reshape order mismatch | verbatim reshape order + per-module attention goldens |
| Buffers accidentally trained | explicit `Buffer(nnx.Variable)` type + test asserting exclusion from `nnx.Param` filter |

## Second pass (explicitly deferred)

Once goldens are green: optional idiomatic API redesign (renaming, unified
config, head library — pooling/classification, diffusion-conditioning
adapters), performed as behavior-preserving refactors guarded by the golden
suite. Only ongoing cost: keep the weight-mapping table in sync.
