# Config unification and post-parity cleanup

**Date:** 2026-07-12
**Status:** Approved (design), pending implementation plan

## Context

The port has reached feature parity with the torch reference (tagged state, see
§Deletions). The reference-mirroring structures — `DataSpec` +
`SwinHPTransformerConfig` + `SwinTransformerConfig`, legacy knobs, reference
class names — are no longer constraints. This design diverges: one Params
dataclass per model (in the style of GenSBI's `Flux1Params`), legacy parameters
dropped, V2-only attention, idiomatic names, and removal of the parity harness.

## Decisions (agreed in brainstorming)

- Drop the parity harness and `weight_transfer.py`; **keep** the flat 2D Swin
  model as a provided sibling model.
- Drop dead knobs (`qk_scale`, `norm_layer`, string-typed
  `patch_embed_norm_layer`), `ape`, explicit `shift_size`.
- Keep **only the SwinV2 variant**: cosine attention + post-norm placement.
  The V1 code paths (`use_cos_attn`, `use_v2_norm_placement` flags) are removed.
- `rngs` stays **out** of Params: `HealSwin(params, rngs=nnx.Rngs(0))`.
  Params is pure serializable data (`json.dumps(asdict(params))` works).
- Naming: **HealSwin family** for the HP model, **SwinUnet family** for flat.
- Layout: one directory per concern — `models/` and `hp/`.

## A — Params dataclasses

Each model file owns its Params. `DataSpec` is deleted; its fields fold in.

### `HealSwinParams` (in `models/healswin.py`)

```python
@dataclass
class HealSwinParams:
    # data / geometry
    nside: int                                  # input HEALPix resolution
    in_channels: int                            # was f_in
    out_channels: int                           # was f_out
    base_pixels: tuple[int, ...] | None = None  # None -> (0..11), full sphere

    # architecture
    patch_size: int = 4
    window_size: int = 4
    embed_dim: int = 96
    depths: tuple[int, ...] = (2, 2, 2, 2)
    num_heads: tuple[int, ...] = (3, 6, 12, 24)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    use_rel_pos_bias: bool = False              # was rel_pos_bias="flat"
    patch_embed_norm: bool = False              # was patch_embed_norm_layer="layernorm"
    shift_strategy: Literal["nest_roll", "nest_grid_shift",
                            "nest_grid_shift_exact",
                            "ring_shift"] = "nest_grid_shift_exact"

    # regularization / training
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    use_checkpoint: bool = False
```

Derived properties (not fields): `npix = len(base_pixels) * nside**2` (was
`dim_in`) and `shift_size = window_size // 2`.

`__post_init__` normalizes and validates:

- `base_pixels=None` → `(0, ..., 11)`; sequence inputs coerced to tuple.
- base_pixels values in `[0, 11]`, strictly increasing (canonical NEST subset
  order) — same rules as today's `DataSpec`.
- Early architecture validation (today these fail deep inside layer
  construction): `patch_size % 4 == 0`; `nside` a power of two;
  `nside**2 % patch_size == 0`; `(nside**2 // patch_size)` divisible by
  `4**(len(depths) - 1)` so every encoder stage has an integer per-face nside;
  `embed_dim * 2**i % num_heads[i] == 0` for every stage.
- `len(depths) == len(num_heads)`.

**Default change vs. reference:** `shift_strategy` defaults to
`nest_grid_shift_exact` (the seam-exact strategy) instead of `nest_roll`.

### `SwinParams` (in `models/swin.py`)

Same treatment for the flat model:

```python
@dataclass
class SwinParams:
    img_size: tuple[int, int]                   # was dim_in (H, W)
    in_channels: int
    out_channels: int
    patch_size: int | tuple[int, int] = (4, 4)
    window_size: int | tuple[int, int] = (4, 4)
    embed_dim: int = 96
    depths: tuple[int, ...] = (2, 2, 2, 2)
    num_heads: tuple[int, ...] = (3, 6, 12, 24)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    use_rel_pos_bias: bool = True               # flat-model default stays True
    use_masking: bool = True
    patch_embed_norm: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    use_checkpoint: bool = False
```

`__post_init__` coerces int → `(v, v)` tuples for `patch_size`/`window_size`
(as today) and validates divisibility of `img_size` by `patch_size` and stage
resolutions by window/merge factors. Derived: `patches_resolution`,
`shift_size = (window_size[0] // 2, window_size[1] // 2)`.

Dropped from the flat config: `final_upsample` (single legal value),
`norm_layer`, `qk_scale`, `ape`, `use_cos_attn`, `use_v2_norm_placement`,
explicit `shift_size`.

### Removed fields, both models

`dim_in`, `base_pix`, `class_names` (data-pipeline metadata, not model
config — downstream code tracks class names itself), `qk_scale`, `norm_layer`,
`patch_embed_norm_layer` (string), `use_cos_attn`, `use_v2_norm_placement`,
`ape`, `shift_size`.

## B — Model code

- **V2-only attention.** `WindowAttention` (both models) keeps only the
  cosine-attention path: `logit_scale` is always present; the V1
  scaled-dot-product branch, `scale`, and `qk_scale` are deleted. Blocks use
  post-norm (V2) placement unconditionally; `use_v2_norm_placement` branches
  are removed.
- **Params threaded down.** Stage/block constructors take the Params object
  plus only stage-local values (`dim`, `input_resolution`, `drop_path`
  slice, `shifted: bool`), replacing the 17-kwarg constructor chains and
  `_make_blocks` kwarg explosion.
- **Renames:**

  | Old | New |
  |---|---|
  | `SwinHPTransformerSys` | `HealSwin` |
  | `SwinHPEncoder` | `HealSwinEncoder` |
  | `HPUnetDecoder` | `HealSwinDecoder` |
  | `SwinHPTransformerConfig` + `DataSpec` | `HealSwinParams` |
  | `SwinTransformerSys` | `SwinUnet` |
  | `UnetDecoder` | `SwinDecoder` |
  | `SwinTransformerConfig` + `DataSpec` | `SwinParams` |
  | `BasicLayer` | `EncoderStage` |
  | `BasicLayer_up` | `DecoderStage` |
  | `FinalPatchExpand_X4` | `FinalPatchExpand` |
  | `f_in` / `f_out` | `in_channels` / `out_channels` |

- **Construction:** `HealSwin(params, rngs=nnx.Rngs(0))`;
  `HealSwinEncoder` remains standalone-usable (encoder-only / regression uses
  are the next project phase).

## C — Deletions and tests

**Before any code change, tag the current commit `parity-verified`** so the
torch-parity state is one checkout away.

Deleted:

- `parity/` (golden generator, goldens, smoke test)
- `src/heal_swin_nnx/weight_transfer.py`
- `src/heal_swin_nnx/config.py` (contents move into model files)
- `tests/test_parity_modules.py`, `tests/test_parity_e2e.py`,
  `tests/test_parity_f64.py`, `tests/parity_utils.py`

Kept, updated to the new API: `test_model.py`, `test_shifting.py`,
`test_topology.py`, `test_seam_geometry.py`, `test_windowing.py`,
`test_buffers.py`. `test_dataspec.py` becomes `test_params.py` (validation
rules + derived properties for both Params classes).

These behavior-level invariants (seam correctness, shift round-trips, window
layout adjacency, shape checks) are the correctness net now that goldens are
gone. The geometry modules move to `hp/` unchanged, so their tests need only
import updates.

## D — Layout and public API

```
src/heal_swin_nnx/
  __init__.py        # public API
  models/
    __init__.py
    healswin.py      # HealSwinParams, HealSwinEncoder, HealSwinDecoder, HealSwin
    swin.py          # SwinParams, SwinEncoder, SwinDecoder, SwinUnet
  hp/
    __init__.py
    topology.py      # was hp_topology.py (content unchanged)
    shifting.py      # was hp_shifting.py
    windowing.py     # was hp_windowing.py
  layers.py          # shared nnx layers: Mlp, DropPath, Identity, TRUNC_NORMAL,
                     #   LN_EPS, PatchMerging, PatchExpand where identical
  variables.py       # Buffer
```

Layers duplicated identically between the two model files hoist into
`layers.py`; model-specific blocks (e.g. each model's `WindowAttention`,
window partition helpers) stay in their model file or `hp/`.

`__init__.py` exports exactly:
`HealSwin`, `HealSwinEncoder`, `HealSwinDecoder`, `HealSwinParams`,
`SwinUnet`, `SwinEncoder`, `SwinDecoder`, `SwinParams`, `Buffer`.

Docs: README usage snippets and the full-sphere/partial-coverage usage doc
get their examples updated to the new API.

## Out of scope

- `param_dtype` / mixed-precision threading (possible future addition; noted,
  not designed).
- Encoder-only / regression heads (next project phase, separate design).
- Any change to HP geometry algorithms in `hp/` — content moves, behavior
  does not.
