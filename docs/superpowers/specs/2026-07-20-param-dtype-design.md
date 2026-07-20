# Design: `param_dtype` for HealSwin, HealConv, and SwinUnet

**Date:** 2026-07-20
**Status:** Approved

## Goal

Expose a parameter-precision knob on all three models, mirroring the flux1
pattern in GenSBI (`gensbi/models/flux1`): a single `param_dtype` field stored
in the Params dataclass and threaded into every layer constructor, with
computation dtype following parameter storage. Default preserves current
behavior (float32) exactly.

Decisions made during brainstorming:

- **flux1-style `param_dtype` only** — no separate computation `dtype` knob.
- **All three models**: `HealSwinParams`, `HealConvParams`, `SwinParams`.
- **Default `"float32"`** — existing configs, tests, and checkpoints stay
  bit-identical; bfloat16 is opt-in per run.
- **Stored as a canonical string** so `json.dumps(dataclasses.asdict(params))`
  keeps working.

## 1. Config surface

Each Params dataclass gains one field in a new "precision" group:

```python
param_dtype: str = "float32"
```

`__post_init__` accepts anything DTypeLike (`"bfloat16"`, `jnp.bfloat16`,
`np.float32`, ...), canonicalizes with `jnp.dtype(...).name`, and raises
`ValueError` if the result is not a floating dtype:

```python
dt = jnp.dtype(self.param_dtype)
if not jnp.issubdtype(dt, jnp.floating):
    raise ValueError(...)
self.param_dtype = dt.name
```

## 2. Threading

- Shared blocks in `layers.py` — `Mlp`, `PatchMerging`, `PatchExpand`,
  `FinalPatchExpand`, `PatchEmbed` — gain a `param_dtype="float32"` keyword
  forwarded to their `nnx.Linear` / `nnx.LayerNorm` / `nnx.Conv` constructors.
- The model files (`healswin.py`, `healconv.py`, `swin.py`) pass
  `params.param_dtype` to those blocks and to the layers they build directly:
  attention `qkv`/`proj` linears, LayerNorms, skip-concat linears, depthwise
  conv kernels, and the output `nnx.Conv`.
- Manually created `nnx.Param`s are initialized in `param_dtype` as well:
  `logit_scale`, `relative_position_bias_table` (and the rel-bias table in
  `swin.py`).
- No separate computation dtype: nnx layers keep `dtype=None`, so compute
  precision follows the promoted input/parameter dtype.

## 3. Forward-path numerics

The input is cast at the top of each **encoder's** `__call__`, flux1-style:

```python
x = jnp.asarray(x, dtype=self.param_dtype)
```

The encoder is the data entry point both for the full U-Net and for
standalone encoder use, so one cast site covers both; decoders receive
already-cast activations.

so with bf16 parameters the whole forward runs in bf16. Deliberate
exceptions that stay in float32:

- **RoPE**: frequency/angle math and rotation tables remain f32.
  `apply_rope` already computes in f32 and casts back to the input dtype —
  untouched. `rope_freqs` (trainable in the mixed variant) stays f32 because
  it feeds the f32 angle computation.
- **Buffers**: all index permutations and attention masks keep their current
  dtypes (ints and f32). The additive attention mask is cast to `attn.dtype`
  at the use site in `WindowAttention.__call__` so it does not silently
  promote bf16 attention back to f32 (−100.0 and 0.0 are exact in bf16).
- **`logit_scale` clamp**: `jnp.exp(jnp.minimum(...))` runs in the storage
  dtype of `logit_scale`; no special handling.

## 4. Testing

Invariant style, no golden values:

- `test_params.py`, for all three Params classes:
  - canonicalization: `param_dtype=jnp.bfloat16` → stored as `"bfloat16"`,
    and `json.dumps(dataclasses.asdict(params))` succeeds;
  - rejection: `"int32"` and garbage strings raise `ValueError`.
- One propagation test per model: construct with `param_dtype="bfloat16"`,
  assert every leaf of `nnx.state(model, nnx.Param)` is bf16 **except**
  `rope_freqs` (f32 by design), Buffers keep their original dtypes, and the
  forward output is bf16 with finite values.
- Existing tests run unchanged under the f32 default.

## Out of scope

- A separate computation `dtype` knob (float32 master weights with bf16
  compute) — can be added later without breaking this API.
- Loss-scaling / optimizer precision — training-loop concerns, not model
  config.
