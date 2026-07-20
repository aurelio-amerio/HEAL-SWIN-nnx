# Compute-dtype (bf16 mixed precision) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compute `dtype` knob (bf16 default, fp32 master weights) to HealSwin, SwinUnet, and HealConv, with knob-independent fp32 islands and spy-test-locked leak prevention.

**Architecture:** GenSBI-contract port (spec: `docs/superpowers/specs/2026-07-20-compute-dtype-design.md`): `dtype` threaded into every `nnx.Linear`/`nnx.Conv`; LayerNorms, the attention logits→softmax region, RoPE, and `l2_normalize` constructed as fp32 islands that emit the compute dtype via explicit `.astype` casts at non-self-healing exits (residual adds, `PatchExpand`); inputs cast to fp32 at the encoder door; final projections and encoder final norms are emit-fp32 endpoints. **Staged rollout:** every task threads with the `dtype` default temporarily at `"float32"` (existing suite stays green, proving the threading is behavior-neutral); the final task flips the default to `"bfloat16"` and triages.

**Tech Stack:** JAX / Flax NNX, pytest (`uv run pytest tests/ -q`, CPU-pinned via pyproject).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-20-compute-dtype-design.md` — read it before starting any task.
- fp32 islands are **constructed** with `dtype=jnp.float32`, never post-hoc `.astype` of a knob-following layer's output.
- Islands emit the compute dtype (explicit `.astype` at non-self-healing exits) — except the emit-fp32 endpoints: decoder `output` conv, encoder final `norm`, and the `FinalPatchExpand` trailing norm that feeds the output conv.
- fp32 Params (`logit_scale`, rel-bias table, `rope_freqs`) are cast at the use site when the site must stay fp32 (`.astype(jnp.float32)`) so pure-bf16-storage mode (`param_dtype="bfloat16"`) keeps the islands fp32.
- Until the final task, the `dtype` field/kwarg default is `"float32"` everywhere (staged rollout); the final task flips Params fields and shared-block kwargs to `"bfloat16"`.
- Models emit fp32: `HealSwin(x)`, `SwinUnet(x)`, `HealConv(x)`, and every `*Encoder(x)` return float32 regardless of both knobs.
- No PyTorch imports in `src/`; `hp/topology.py` must never import healpy; precomputed state lives in `Buffer`s (none of this plan touches those rules, but don't violate them).
- Every spy test must be RED-verified: temporarily break the thing it guards, see it fail, restore. A spy that never went red proves nothing.
- Tests: `uv run pytest tests/ -q` (runs with `JAX_PLATFORMS=cpu -n 2`). Commit style: conventional (`feat:`/`fix:`/`test:`/`docs:`).

## File Structure

- `src/heal_swin_nnx/layers.py` — `canonical_float_dtype(field_name=...)`, fp32 `l2_normalize`, `dtype` kwarg + fp32 norms on `Mlp`, `PatchMerging`, `PatchExpand`, `FinalPatchExpand`, `PatchEmbed`.
- `src/heal_swin_nnx/models/healswin.py` — `dtype` field on `HealSwinParams`; attention island; block residual casts; door/endpoint dtypes.
- `src/heal_swin_nnx/models/swin.py` — same for `SwinParams` + swin.py's **own** copies of the four patch blocks.
- `src/heal_swin_nnx/models/healconv.py` — same for `HealConvParams` (no attention; `dwconv` is the mixer).
- `tests/test_params.py` — `dtype` validation tests (mirrors the `param_dtype` section).
- `tests/test_precision.py` — **new file**: shared-block dtype tests, per-model battery (master weights / output / grads fp32), spy tests, calibrated drift lock.
- `tests/test_model.py`, `tests/test_healconv.py` — output-dtype assertion updates (bf16 → fp32 under pure-bf16 storage) and, in the flip task, exactness-test pinning.

---

### Task 1: `dtype` field on all three Params dataclasses + field-aware validation

**Files:**
- Modify: `src/heal_swin_nnx/layers.py:26-48` (`canonical_float_dtype`)
- Modify: `src/heal_swin_nnx/models/healswin.py:57-66` (`HealSwinParams`)
- Modify: `src/heal_swin_nnx/models/swin.py:47-56` (`SwinParams`)
- Modify: `src/heal_swin_nnx/models/healconv.py:55-63` (`HealConvParams`)
- Test: `tests/test_params.py` (append after line 261)

**Interfaces:**
- Consumes: existing `canonical_float_dtype(value)` in `layers.py`, `PARAMS_FACTORIES` in `test_params.py:230-236`.
- Produces: `canonical_float_dtype(value, field_name="param_dtype")` (second positional arg used in error messages); `params.dtype: str` on all three dataclasses, default `"float32"` **for now** (flipped in Task 7). Tasks 2-6 rely on `params.dtype` existing and being a canonical dtype-name string.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_params.py` (after `test_param_dtype_rejects_float64_without_x64`, line 261; `json`, `dataclasses`, `jax`, `jnp`, `pytest` are already imported at the top of the file):

```python
# --- compute dtype ----------------------------------------------------------


@pytest.mark.parametrize("mk", PARAMS_FACTORIES)
def test_dtype_canonicalization(mk):
    for spec in ("bfloat16", jnp.bfloat16, jnp.dtype(jnp.bfloat16)):
        p = mk(dtype=spec)
        assert p.dtype == "bfloat16"
        json.dumps(dataclasses.asdict(p))  # must stay serializable


@pytest.mark.parametrize("mk", PARAMS_FACTORIES)
@pytest.mark.parametrize("bad", ["int32", "bool", "not_a_dtype", 7, None])
def test_dtype_rejects_non_floats(mk, bad):
    # \bdtype does NOT match "param_dtype" ('_' is a word char): this verifies
    # the error message names the *compute* knob, i.e. the field_name plumbing.
    with pytest.raises(ValueError, match=r"\bdtype"):
        mk(dtype=bad)


@pytest.mark.parametrize("mk", PARAMS_FACTORIES)
def test_dtype_rejects_float64_without_x64(mk):
    assert not jax.config.jax_enable_x64
    with pytest.raises(ValueError):
        mk(dtype="float64")


@pytest.mark.parametrize("mk", PARAMS_FACTORIES)
def test_precision_knobs_independent(mk):
    p = mk(param_dtype="bfloat16", dtype="float32")
    assert p.param_dtype == "bfloat16" and p.dtype == "float32"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_params.py -q -k "dtype and not param_dtype"`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'dtype'` for every case.

- [ ] **Step 3: Implement**

In `src/heal_swin_nnx/layers.py`, replace `canonical_float_dtype` (lines 26-48) with:

```python
def canonical_float_dtype(value, field_name="param_dtype"):
    """Canonicalize a DTypeLike into its dtype name ("float32", "bfloat16", ...).

    Params dataclasses store dtypes as canonical strings so
    ``json.dumps(dataclasses.asdict(params))`` keeps working; every jnp/nnx
    API accepts the string form. Floating dtypes only. ``field_name`` labels
    error messages (the same helper validates ``param_dtype`` and ``dtype``)."""
    if value is None:
        # numpy resolves dtype(None) to float64 (its legacy default dtype);
        # that's never an intentional dtype choice.
        raise ValueError("%s must be a floating DTypeLike, got %r" % (field_name, value))
    try:
        dt = jnp.dtype(value)
    except TypeError as e:
        raise ValueError("%s must be a floating DTypeLike, got %r"
                         % (field_name, value)) from e
    if not jnp.issubdtype(dt, jnp.floating):
        raise ValueError("%s must be a floating dtype, got %r" % (field_name, value))
    if dt.name == "float64" and not jax.config.jax_enable_x64:
        raise ValueError(
            "%s='float64' requires enabling jax_enable_x64 before "
            "constructing params (jax.config.update('jax_enable_x64', True)); "
            "otherwise jnp array creation silently yields float32, got %r"
            % (field_name, value))
    return dt.name
```

In each of the three dataclasses, extend the precision group. `healswin.py` lines 57-58 become:

```python
    # precision
    param_dtype: str = "float32"     # parameter storage; any DTypeLike, stored as name
    dtype: str = "float32"           # compute/matmul dtype; "float32" is a staging
                                     # default — flipped to "bfloat16" in the final
                                     # task of the compute-dtype plan
```

and in `HealSwinParams.__post_init__` (line 66), replace the canonicalization line with:

```python
        self.param_dtype = canonical_float_dtype(self.param_dtype, "param_dtype")
        self.dtype = canonical_float_dtype(self.dtype, "dtype")
```

Make the identical two edits in `SwinParams` (`swin.py:47-48` field group, `:56` post_init) and `HealConvParams` (`healconv.py:55-56` field group, `:63` post_init).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_params.py -q`
Expected: PASS (all, including the pre-existing `param_dtype` cases — the helper's default `field_name` keeps their messages intact).

- [ ] **Step 5: Full suite (staging invariant: nothing else changed)**

Run: `uv run pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/heal_swin_nnx/layers.py src/heal_swin_nnx/models/healswin.py \
        src/heal_swin_nnx/models/swin.py src/heal_swin_nnx/models/healconv.py \
        tests/test_params.py
git commit -m "feat: dtype (compute) field on all three Params dataclasses

Staging default float32; flipped to bfloat16 at the end of the plan.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: fp32 `l2_normalize` island + `dtype` threading in shared blocks (`layers.py`)

**Files:**
- Modify: `src/heal_swin_nnx/layers.py` (`l2_normalize`:14-23, `Mlp`:73-90, `PatchMerging`:133-147, `PatchExpand`:150-165, `FinalPatchExpand`:168-183, `PatchEmbed`:186-205)
- Test: `tests/test_precision.py` (create)

**Interfaces:**
- Consumes: Task 1's `params.dtype` string convention (blocks take the same string form).
- Produces: every shared block gains keyword `dtype="float32"` (staging default; Task 7 flips to `"bfloat16"`) placed immediately after `param_dtype`:
  - `Mlp(in_features, hidden_features=None, out_features=None, drop=0.0, param_dtype="float32", dtype="float32", *, rngs)`
  - `PatchMerging(dim, dim_scale=2, param_dtype="float32", dtype="float32", *, rngs)`
  - `PatchExpand(dim, dim_scale=2, param_dtype="float32", dtype="float32", *, rngs)` — **emits `dtype`** (exit cast)
  - `FinalPatchExpand(patch_size, dim, param_dtype="float32", dtype="float32", *, rngs)` — **emits fp32** (no exit cast; fp32 tail)
  - `PatchEmbed(npix, patch_size, in_channels, embed_dim, norm=False, param_dtype="float32", dtype="float32", *, rngs)` — emits `dtype`
  - `l2_normalize(x, axis=-1, eps=1e-12)` — unchanged signature, now fp32 math returning `x.dtype`.
  Tasks 3-5 pass `dtype=params.dtype` at every one of these call sites.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_precision.py`:

```python
"""Mixed-precision contract tests: fp32 master weights, bf16 compute knob,
fp32 islands, leak locks (spy tests), and the calibrated drift lock.

Spec: docs/superpowers/specs/2026-07-20-compute-dtype-design.md
"""
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from heal_swin_nnx.layers import (FinalPatchExpand, Mlp, PatchEmbed, PatchExpand,
                                  PatchMerging, l2_normalize)


def test_l2_normalize_is_fp32_island():
    x = jax.random.normal(jax.random.key(0), (2, 3, 8)).astype(jnp.bfloat16)
    y = l2_normalize(x)
    assert y.dtype == jnp.bfloat16                       # emits input dtype
    ref = l2_normalize(x.astype(jnp.float32))
    assert float(jnp.max(jnp.abs(y.astype(jnp.float32) - ref))) < 1e-2
    assert float(jnp.max(jnp.abs(
        jnp.sum(ref * ref, axis=-1) - 1.0))) < 1e-5      # actually normalizes


def test_shared_blocks_emit_compute_dtype():
    rngs = nnx.Rngs(0)
    x = jnp.ones((2, 16, 8), jnp.bfloat16)
    assert Mlp(8, 16, dtype="bfloat16", rngs=rngs)(x).dtype == jnp.bfloat16
    assert PatchMerging(8, dtype="bfloat16", rngs=rngs)(x).dtype == jnp.bfloat16
    assert PatchExpand(8, dtype="bfloat16", rngs=rngs)(x).dtype == jnp.bfloat16
    # FinalPatchExpand's sole consumer is the fp32 output conv: deliberate fp32 tail
    assert FinalPatchExpand(4, 8, dtype="bfloat16", rngs=rngs)(x).dtype == jnp.float32
    xe = jnp.ones((2, 64, 3), jnp.bfloat16)
    pe = PatchEmbed(64, 4, 3, 8, norm=True, dtype="bfloat16", rngs=rngs)
    assert pe(xe).dtype == jnp.bfloat16                  # norm exit cast
    pe2 = PatchEmbed(64, 4, 3, 8, norm=False, dtype="bfloat16", rngs=rngs)
    assert pe2(xe).dtype == jnp.bfloat16                 # conv computes bf16


def test_shared_blocks_master_weights_follow_param_dtype_not_dtype():
    rngs = nnx.Rngs(0)
    m = Mlp(8, 16, param_dtype="float32", dtype="bfloat16", rngs=rngs)
    for path, v in nnx.to_flat_state(nnx.state(m, nnx.Param)):
        assert v[...].dtype == jnp.float32, path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_precision.py -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'dtype'` (blocks), and the `l2_normalize` island test fails on the closeness assertion (bf16-native rsqrt) or passes trivially — either way the block tests are red.

- [ ] **Step 3: Implement in `layers.py`**

Replace `l2_normalize` (lines 14-23):

```python
def l2_normalize(x, axis=-1, eps=1e-12):
    """L2-normalize with a finite gradient at x = 0. fp32 island: math runs in
    float32 regardless of the compute dtype (a bf16 sum-of-squares loses ~5
    mantissa bits and eps=1e-12 is below bf16 resolution), result is cast back
    to the input dtype.

    ``x / max(||x||, eps)`` NaNs in the backward pass for exactly-zero vectors
    (d/dx ||x|| is 0/0 at x = 0, and the clamp doesn't block the NaN), which
    zero-background inputs reach through the zero-initialized biases ahead of
    the first attention block. ``rsqrt(sum(x^2) + eps)`` is smooth at 0 and
    matches the clamped division everywhere else.
    """
    x32 = x.astype(jnp.float32)
    out = x32 * jax.lax.rsqrt(jnp.sum(jnp.square(x32), axis=axis, keepdims=True) + eps)
    return out.astype(x.dtype)
```

Replace `Mlp.__init__` (lines 74-82):

```python
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0,
                 param_dtype="float32", dtype="float32", *, rngs):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nnx.Linear(in_features, hidden_features, kernel_init=TRUNC_NORMAL,
                              dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_features, out_features, kernel_init=TRUNC_NORMAL,
                              dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)
```

Replace `PatchMerging.__init__` (lines 136-141) — norm becomes a knob-independent fp32 island; its output self-heals through `reduction` (a compute-dtype Linear downcasts its inputs), so the module needs no exit cast:

```python
    def __init__(self, dim, dim_scale=2, param_dtype="float32", dtype="float32", *, rngs):
        self.reduction = nnx.Linear(4 * dim, dim_scale * dim, use_bias=False,
                                    kernel_init=TRUNC_NORMAL, dtype=dtype,
                                    param_dtype=param_dtype, rngs=rngs)
        # fp32 island; output feeds `reduction` on the same line (self-healing)
        self.norm = nnx.LayerNorm(4 * dim, epsilon=LN_EPS, dtype=jnp.float32,
                                  param_dtype=param_dtype, rngs=rngs)
```

Replace `PatchExpand.__init__` and `__call__` (lines 153-165) — the trailing norm's consumers are the next stage's residual blocks and the skip concat, neither of which self-heals, so the module emits the compute dtype:

```python
    def __init__(self, dim, dim_scale=2, param_dtype="float32", dtype="float32", *, rngs):
        self.dtype = dtype
        self.expand = (nnx.Linear(dim, dim_scale * dim, use_bias=False,
                                  kernel_init=TRUNC_NORMAL, dtype=dtype,
                                  param_dtype=param_dtype, rngs=rngs)
                       if dim_scale != 1 else Identity())
        self.norm = nnx.LayerNorm(dim * dim_scale // 4, epsilon=LN_EPS,
                                  dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x):
        x = self.expand(x)
        C = x.shape[-1]
        x = rearrange(x, "b n (p c) -> b (n p) c", p=4, c=C // 4)
        # fp32 norm island exits here: consumers (stage residuals, skip concat)
        # do not self-heal, so emit the compute dtype
        return self.norm(x).astype(self.dtype)
```

Replace `FinalPatchExpand.__init__` (lines 171-177) — trailing norm's sole consumer is the emit-fp32 output conv, so **no** exit cast (`__call__` unchanged):

```python
    def __init__(self, patch_size, dim, param_dtype="float32", dtype="float32", *, rngs):
        self.patch_size = patch_size
        self.expand = nnx.Linear(dim, patch_size * dim, use_bias=False,
                                 kernel_init=TRUNC_NORMAL, dtype=dtype,
                                 param_dtype=param_dtype, rngs=rngs)
        # fp32 island with NO exit cast: sole consumer is the fp32 output conv
        # (deliberate fp32 tail — casting here would round right before it)
        self.norm = nnx.LayerNorm(dim, epsilon=LN_EPS, dtype=jnp.float32,
                                  param_dtype=param_dtype, rngs=rngs)
```

Replace `PatchEmbed.__init__` and `__call__` (lines 189-205):

```python
    def __init__(self, npix, patch_size, in_channels, embed_dim, norm=False,
                 param_dtype="float32", dtype="float32", *, rngs):
        self.npix = npix
        self.dtype = dtype
        self.num_patches = npix // patch_size
        self.proj = nnx.Conv(in_channels, embed_dim,
                             kernel_size=(patch_size,), strides=(patch_size,),
                             padding="VALID", dtype=dtype, param_dtype=param_dtype,
                             rngs=rngs)
        self.norm = (nnx.LayerNorm(embed_dim, epsilon=LN_EPS, dtype=jnp.float32,
                                   param_dtype=param_dtype, rngs=rngs) if norm else None)

    def __call__(self, x):  # (B, N, in_channels) channels-last
        assert x.shape[1] == self.npix, (
            "Input map size (%d) doesn't match model (%d)." % (x.shape[1], self.npix))
        x = self.proj(x)
        if self.norm is not None:
            # fp32 norm island exits straight into stage residuals: emit compute dtype
            x = self.norm(x).astype(self.dtype)
        return x
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_precision.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite (staging invariant)**

Run: `uv run pytest tests/ -q`
Expected: PASS — models don't pass `dtype` yet, blocks default `"float32"`, and the `l2_normalize`/norm changes are fp32-in-fp32 no-ops. If `test_jit_matches_eager` or `test_remat_matches_no_remat` drift beyond tolerance, the fp32-island LayerNorms changed lowering — investigate before proceeding (should not happen: `dtype=jnp.float32` on fp32 inputs is the identity configuration).

- [ ] **Step 6: Commit**

```bash
git add src/heal_swin_nnx/layers.py tests/test_precision.py
git commit -m "feat: dtype threading + fp32 islands in shared layer blocks

l2_normalize computes fp32 (emits input dtype); shared-block LayerNorms are
knob-independent fp32 islands; PatchExpand/PatchEmbed exit-cast to compute
dtype (non-self-healing consumers), FinalPatchExpand keeps the fp32 tail.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: HealSwin — attention island, residual casts, door and endpoints

**Files:**
- Modify: `src/heal_swin_nnx/models/healswin.py` (`WindowAttention`:143-212, `HealSwinBlock`:215-265, `EncoderStage`:280, `DecoderStage`:297-298, `HealSwinEncoder`:321-348, `HealSwinDecoder`:354-396)
- Modify: `tests/test_model.py:238-241` (output-dtype assertion)
- Test: `tests/test_precision.py` (append)

**Interfaces:**
- Consumes: Task 1 `params.dtype`; Task 2 shared-block `dtype=` kwargs and fp32 `l2_normalize`.
- Produces: `tests/test_precision.py` gains `make_healswin(**over) -> (HealSwin, HealSwinParams)`, `_smooth_input(key, npix, channels, batch=2) -> (batch, npix, channels) float32`, `_rel_err(a, b) -> float` — Tasks 4-6 reuse `_smooth_input`/`_rel_err` and mirror `make_healswin`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_precision.py`:

```python
# --- HealSwin ---------------------------------------------------------------

from heal_swin_nnx import HealSwin, HealSwinParams
from heal_swin_nnx.models.healswin import HealSwinBlock


def make_healswin(**over):
    kw = dict(nside=16, in_channels=3, out_channels=2, base_pixels=(0, 1, 2, 3),
              embed_dim=16, depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0)
    kw.update(over)
    p = HealSwinParams(**kw)
    return HealSwin(p, rngs=nnx.Rngs(0)), p


def _smooth_input(key, npix, channels, batch=2):
    """Deterministic smooth test signal: a few seeded sinusoids over the pixel
    index (closer to real fields than white noise; error behavior differs)."""
    k1, k2 = jax.random.split(key)
    t = jnp.linspace(0.0, 1.0, npix)[None, :, None, None]        # (1, npix, 1, 1)
    amp = jax.random.normal(k1, (batch, 1, channels, 4))
    phase = jax.random.uniform(k2, (batch, 1, channels, 4), maxval=2 * jnp.pi)
    freqs = 2 * jnp.pi * jnp.arange(1.0, 5.0)                    # (4,)
    return jnp.sum(amp * jnp.sin(freqs * t + phase), axis=-1)    # (batch, npix, C)


def _rel_err(a, b):
    a = jnp.asarray(a, jnp.float32)
    b = jnp.asarray(b, jnp.float32)
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(a)) + 1e-12))


@pytest.mark.parametrize("pos_embed", ["rope_mixed", "rel_bias"])
def test_healswin_master_weights_fp32_under_bf16_compute(pos_embed):
    model, _ = make_healswin(dtype="bfloat16", pos_embed=pos_embed)
    for path, v in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        assert v[...].dtype == jnp.float32, path


def test_healswin_output_and_grads_fp32():
    model, p = make_healswin(dtype="bfloat16")
    model.eval()
    x = _smooth_input(jax.random.key(0), p.npix, 3)
    y = model(x)
    assert y.dtype == jnp.float32 and bool(jnp.isfinite(y).all())
    tokens, _ = model.encoder(x)
    assert tokens.dtype == jnp.float32                 # standalone-encoder endpoint
    grads = nnx.grad(lambda m: jnp.mean(m(x) ** 2))(model)
    for path, g in nnx.to_flat_state(grads):
        joined = "/".join(str(q) for q in path)
        assert g[...].dtype == jnp.float32, joined
        assert bool(jnp.isfinite(g[...]).all()), joined


def _block_entry_spy(monkeypatch, cls):
    seen = []
    orig = cls.__call__

    def spy(self, x, *args, **kw):
        seen.append(x.dtype)
        return orig(self, x, *args, **kw)

    monkeypatch.setattr(cls, "__call__", spy)
    return seen


@pytest.mark.parametrize("patch_embed_norm", [False, True])
def test_healswin_stream_is_bf16_inside_blocks(monkeypatch, patch_embed_norm):
    """Leak lock: the dtype entering EVERY block is the compute dtype. Block i's
    entry is block i-1's residual output, so this sweeps all post-norm residual
    casts, PatchMerging's self-heal, PatchExpand's exit cast, the concat path,
    and (parametrized) the PatchEmbed-norm exit cast."""
    seen = _block_entry_spy(monkeypatch, HealSwinBlock)
    model, p = make_healswin(dtype="bfloat16", patch_embed_norm=patch_embed_norm)
    model.eval()
    model(_smooth_input(jax.random.key(0), p.npix, 3))
    assert len(seen) >= 6            # 4 encoder blocks + decoder-stage blocks
    assert all(dt == jnp.bfloat16 for dt in seen), seen


def test_healswin_softmax_island_is_fp32(monkeypatch):
    seen = []
    orig = jax.nn.softmax

    def spy(x, axis=-1, **kw):
        seen.append(x.dtype)
        return orig(x, axis=axis, **kw)

    monkeypatch.setattr(jax.nn, "softmax", spy)
    model, p = make_healswin(dtype="bfloat16")
    model.eval()
    model(_smooth_input(jax.random.key(0), p.npix, 3))
    assert seen and all(dt == jnp.float32 for dt in seen), seen
```

- [ ] **Step 2: Run tests to verify they fail (natural RED: knob exists but is dead)**

Run: `uv run pytest tests/test_precision.py -q -k healswin`
Expected: `test_healswin_stream_is_bf16_inside_blocks` FAILS (entries are float32 — nothing consumes `dtype` yet). `test_healswin_softmax_island_is_fp32` and the master-weights test PASS already (everything is fp32); they go red only if the implementation regresses — their RED verification is Step 5.

- [ ] **Step 3: Implement in `healswin.py`**

`WindowAttention.__init__`: add `self.dtype = params.dtype` as the first line after `self.pos_embed = params.pos_embed` (line 149), and thread the linears (lines 173-178):

```python
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=params.qkv_bias,
                              kernel_init=TRUNC_NORMAL, dtype=params.dtype,
                              param_dtype=params.param_dtype, rngs=rngs)
        self.attn_drop = nnx.Dropout(params.attn_drop_rate, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=TRUNC_NORMAL, dtype=params.dtype,
                               param_dtype=params.param_dtype, rngs=rngs)
```

`WindowAttention.__call__` (lines 195-209): replace the logits/softmax section with:

```python
        # fp32 logits island: bf16 operands with fp32 accumulation (free on
        # tensor cores), and scale/bias/mask/softmax stay fp32. Cosine logits
        # are bounded (|logit| <= 100), so this is about bf16's 8-bit mantissa
        # resolution near the softmax operating point, not overflow.
        attn = jnp.einsum("bhnd,bhmd->bhnm", q, k,
                          preferred_element_type=jnp.float32)
        logit_scale = jnp.exp(jnp.minimum(
            self.logit_scale[...].astype(jnp.float32), jnp.log(1.0 / 0.01)))
        attn = attn * logit_scale

        if self.pos_embed == "rel_bias":
            bias = self.relative_position_bias_table[...][self.relative_position_index[...]]
            attn = attn + bias.transpose(2, 0, 1)[None].astype(jnp.float32)

        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.reshape(B_ // nW, nW, self.num_heads, N, N)
                    + mask.astype(attn.dtype)[None, :, None])
            attn = attn.reshape(-1, self.num_heads, N, N)
        attn = jax.nn.softmax(attn, axis=-1).astype(self.dtype)  # island exit
        attn = self.attn_drop(attn)
```

(The `.astype(jnp.float32)` on `logit_scale` and `bias` are use-site casts keeping the island fp32 under pure-bf16 *storage* (`param_dtype="bfloat16"`); at the fp32-storage default they are no-ops.)

`HealSwinBlock.__init__` (lines 222-229): add `self.dtype = params.dtype` before `self.norm1`, and make norms fp32 islands / thread the Mlp:

```python
        self.dtype = params.dtype
        self.norm1 = nnx.LayerNorm(dim, epsilon=LN_EPS, dtype=jnp.float32,
                                   param_dtype=params.param_dtype, rngs=rngs)
        self.attn = WindowAttention(params, dim, num_heads, self.window_size, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(dim, epsilon=LN_EPS, dtype=jnp.float32,
                                   param_dtype=params.param_dtype, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * params.mlp_ratio), drop=params.drop_rate,
                       dtype=params.dtype, param_dtype=params.param_dtype, rngs=rngs)
```

`HealSwinBlock.__call__` (lines 264-265):

```python
        # fp32 norm islands exit here: cast BEFORE the residual add — adds do
        # not self-heal, one fp32 summand re-promotes the whole downstream stream
        x = shortcut + self.drop_path(self.norm1(x).astype(self.dtype))
        return x + self.drop_path(self.norm2(self.mlp(x)).astype(self.dtype))
```

`EncoderStage.__init__` (line 280) and `DecoderStage.__init__` (lines 297-298): thread the down/upsamplers:

```python
        self.downsample = (PatchMerging(dim=dim, dtype=params.dtype,
                                        param_dtype=params.param_dtype,
                                        rngs=rngs) if downsample else None)
```

```python
        self.upsample = (PatchExpand(dim=dim, dim_scale=2, dtype=params.dtype,
                                     param_dtype=params.param_dtype, rngs=rngs)
                         if upsample else None)
```

`HealSwinEncoder.__init__` (lines 321-323, 337-338): thread patch embed, pin the final norm fp32 (emit-fp32 endpoint — standalone encoders return it, and inside the U-Net the decoder's first `PatchExpand.expand` linear self-heals):

```python
        self.patch_embed = PatchEmbed(params.npix, params.patch_size, params.in_channels,
                                      params.embed_dim, params.patch_embed_norm,
                                      dtype=params.dtype,
                                      param_dtype=params.param_dtype, rngs=rngs)
```

```python
        self.norm = nnx.LayerNorm(self.num_features, epsilon=LN_EPS, dtype=jnp.float32,
                                  param_dtype=params.param_dtype, rngs=rngs)
```

`HealSwinEncoder.__call__` (line 341): inputs fp32 at the door (the patch-embed conv downcasts):

```python
        x = jnp.asarray(x, dtype=jnp.float32)
```

`HealSwinDecoder.__init__` (lines 363-384): thread `concat_back_dim`, both `PatchExpand` construction sites, and `FinalPatchExpand`; pin `norm_up` fp32 (self-heals into `FinalPatchExpand.expand`) and the output conv fp32 (emit-fp32 endpoint):

```python
            concat_back_dim.append(
                nnx.Linear(2 * dim, dim, kernel_init=TRUNC_NORMAL, dtype=params.dtype,
                           param_dtype=params.param_dtype, rngs=rngs)
                if i_layer > 0 else Identity())
            if i_layer == 0:
                layers_up.append(PatchExpand(dim=dim, dim_scale=2, dtype=params.dtype,
                                             param_dtype=params.param_dtype, rngs=rngs))
```

```python
        self.norm_up = nnx.LayerNorm(params.embed_dim, epsilon=LN_EPS, dtype=jnp.float32,
                                     param_dtype=params.param_dtype, rngs=rngs)
        self.up = FinalPatchExpand(patch_size=params.patch_size, dim=params.embed_dim,
                                   dtype=params.dtype,
                                   param_dtype=params.param_dtype, rngs=rngs)
        # emit-fp32 endpoint: constructed fp32 (never a post-hoc astype)
        self.output = nnx.Conv(params.embed_dim, params.out_channels, kernel_size=(1,),
                               use_bias=False, dtype=jnp.float32,
                               param_dtype=params.param_dtype, rngs=rngs)
```

Update `tests/test_model.py:238-241` (in `test_healswin_param_dtype_propagates` — the output conv is now an fp32 island, so pure-bf16 storage emits fp32):

```python
    x = jax.random.normal(jax.random.key(0), (2, p.npix, 3))
    y = model(x)
    # models emit fp32: the output conv is an fp32 island regardless of knobs
    assert y.dtype == jnp.float32 and y.shape == (2, p.npix, 5)
    assert bool(jnp.isfinite(y).all())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_precision.py tests/test_model.py -q`
Expected: PASS.

- [ ] **Step 5: RED-verify each cast (spy discipline)**

For each of the following, make the temporary edit, run `uv run pytest tests/test_precision.py -q -k "healswin and (stream or softmax)"`, confirm FAIL, then revert:

1. Remove `.astype(self.dtype)` from the `norm1` residual line in `HealSwinBlock.__call__` → stream spy fails.
2. Remove `.astype(self.dtype)` from `PatchExpand.__call__` in `layers.py` → stream spy fails (decoder entries fp32).
3. Remove `.astype(self.dtype)` from the `PatchEmbed` norm exit in `layers.py` → the `patch_embed_norm=True` variant fails.
4. Change the softmax line to `jax.nn.softmax(attn.astype(self.dtype), axis=-1)` → softmax spy fails.

All four must go red. Revert cleanly (`git diff` must show only intended changes).

- [ ] **Step 6: Full suite (staging invariant)**

Run: `uv run pytest tests/ -q`
Expected: PASS — the `dtype` default is still `"float32"`, so default-constructed models are numerically fp32 throughout; only the pure-bf16-storage test assertion changed (Step 3).

- [ ] **Step 7: Commit**

```bash
git add src/heal_swin_nnx/models/healswin.py tests/test_precision.py tests/test_model.py
git commit -m "feat: thread compute dtype through HealSwin (fp32 islands + leak casts)

Attention: fp32-accumulated logits via einsum preferred_element_type, fp32
scale/bias/mask/softmax, exit cast after softmax. Post-norm residuals cast
island outputs to the compute dtype. Door casts to fp32; output conv and
encoder final norm are emit-fp32 endpoints. Spy tests RED-verified.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: SwinUnet — mirror threading (including swin.py's own patch blocks)

**Files:**
- Modify: `src/heal_swin_nnx/models/swin.py` (`WindowAttention`:143-212, `SwinBlock`:215-269, `PatchMerging`:272-291, `PatchExpand`:294-310, `FinalPatchExpand`:313-333, `PatchEmbed`:336-355, `EncoderStage`:370-372, `DecoderStage`:388-390, `SwinEncoder`:423-427, `SwinDecoder`:449-471)
- Modify: `tests/test_model.py:271-274` (output-dtype assertion)
- Test: `tests/test_precision.py` (append)

**Interfaces:**
- Consumes: Task 2 fp32 `l2_normalize`; Task 3's `_smooth_input`, `_rel_err`, `_block_entry_spy` helpers in `tests/test_precision.py`.
- Produces: `make_flat(**over) -> (SwinUnet, SwinParams)` in `tests/test_precision.py` (used by Task 6). swin.py's local patch blocks gain a `dtype="float32"` kwarg after `param_dtype` (same staging default, flipped in Task 7).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_precision.py`:

```python
# --- SwinUnet ---------------------------------------------------------------

from heal_swin_nnx import SwinParams, SwinUnet
from heal_swin_nnx.models.swin import SwinBlock


def make_flat(**over):
    kw = dict(img_size=(32, 64), in_channels=2, out_channels=3, embed_dim=16,
              depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0)
    kw.update(over)
    p = SwinParams(**kw)
    return SwinUnet(p, rngs=nnx.Rngs(0)), p


def _smooth_input_2d(key, img_size, channels, batch=2):
    H, W = img_size
    flat = _smooth_input(key, H * W, channels, batch=batch)
    return flat.reshape(batch, H, W, channels)


def test_flat_master_weights_fp32_under_bf16_compute():
    model, _ = make_flat(dtype="bfloat16")
    for path, v in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        assert v[...].dtype == jnp.float32, path


def test_flat_output_and_grads_fp32():
    model, p = make_flat(dtype="bfloat16")
    model.eval()
    x = _smooth_input_2d(jax.random.key(0), p.img_size, 2)
    y = model(x)
    assert y.dtype == jnp.float32 and bool(jnp.isfinite(y).all())
    tokens, _ = model.encoder(x)
    assert tokens.dtype == jnp.float32
    grads = nnx.grad(lambda m: jnp.mean(m(x) ** 2))(model)
    for path, g in nnx.to_flat_state(grads):
        joined = "/".join(str(q) for q in path)
        assert g[...].dtype == jnp.float32, joined


@pytest.mark.parametrize("patch_embed_norm", [False, True])
def test_flat_stream_is_bf16_inside_blocks(monkeypatch, patch_embed_norm):
    seen = _block_entry_spy(monkeypatch, SwinBlock)
    model, p = make_flat(dtype="bfloat16", patch_embed_norm=patch_embed_norm)
    model.eval()
    model(_smooth_input_2d(jax.random.key(0), p.img_size, 2))
    assert len(seen) >= 6
    assert all(dt == jnp.bfloat16 for dt in seen), seen


def test_flat_softmax_island_is_fp32(monkeypatch):
    seen = []
    orig = jax.nn.softmax

    def spy(x, axis=-1, **kw):
        seen.append(x.dtype)
        return orig(x, axis=axis, **kw)

    monkeypatch.setattr(jax.nn, "softmax", spy)
    model, p = make_flat(dtype="bfloat16")
    model.eval()
    model(_smooth_input_2d(jax.random.key(0), p.img_size, 2))
    assert seen and all(dt == jnp.float32 for dt in seen), seen
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_precision.py -q -k flat`
Expected: `test_flat_stream_is_bf16_inside_blocks` FAILS (entries float32 — knob dead in swin.py).

- [ ] **Step 3: Implement in `swin.py`**

`WindowAttention` (lines 143-212): identical treatment to Task 3's — add `self.dtype = params.dtype` after `self.pos_embed = params.pos_embed`; thread `qkv`/`proj`:

```python
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=params.qkv_bias,
                              kernel_init=TRUNC_NORMAL, dtype=params.dtype,
                              param_dtype=params.param_dtype, rngs=rngs)
        self.attn_drop = nnx.Dropout(params.attn_drop_rate, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=TRUNC_NORMAL, dtype=params.dtype,
                               param_dtype=params.param_dtype, rngs=rngs)
```

and replace the logits/softmax section (lines 194-210) with:

```python
        # fp32 logits island: bf16 operands with fp32 accumulation (free on
        # tensor cores), and scale/bias/mask/softmax stay fp32. Cosine logits
        # are bounded (|logit| <= 100), so this is about bf16's 8-bit mantissa
        # resolution near the softmax operating point, not overflow.
        attn = jnp.einsum("bhnd,bhmd->bhnm", q, k,
                          preferred_element_type=jnp.float32)
        logit_scale = jnp.exp(jnp.minimum(
            self.logit_scale[...].astype(jnp.float32), jnp.log(1.0 / 0.01)))
        attn = attn * logit_scale

        if self.pos_embed == "rel_bias":
            ws_area = self.window_size[0] * self.window_size[1]
            bias = self.relative_position_bias_table[...][
                self.relative_position_index[...].reshape(-1)].reshape(ws_area, ws_area, -1)
            attn = attn + bias.transpose(2, 0, 1)[None].astype(jnp.float32)

        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.reshape(B_ // nW, nW, self.num_heads, N, N)
                    + mask.astype(attn.dtype)[None, :, None])
            attn = attn.reshape(-1, self.num_heads, N, N)
        attn = jax.nn.softmax(attn, axis=-1).astype(self.dtype)  # island exit
        attn = self.attn_drop(attn)
```

`SwinBlock.__init__` (lines 227-234): add `self.dtype = params.dtype` before `self.norm1`; norms fp32; Mlp threaded:

```python
        self.dtype = params.dtype
        self.norm1 = nnx.LayerNorm(dim, epsilon=LN_EPS, dtype=jnp.float32,
                                   param_dtype=params.param_dtype, rngs=rngs)
        self.attn = WindowAttention(params, dim, num_heads, self.window_size, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(dim, epsilon=LN_EPS, dtype=jnp.float32,
                                   param_dtype=params.param_dtype, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * params.mlp_ratio), drop=params.drop_rate,
                       dtype=params.dtype, param_dtype=params.param_dtype, rngs=rngs)
```

`SwinBlock.__call__` (lines 268-269):

```python
        # fp32 norm islands exit here: cast BEFORE the residual add — adds do
        # not self-heal, one fp32 summand re-promotes the whole downstream stream
        x = shortcut + self.drop_path(self.norm1(x).astype(self.dtype))
        return x + self.drop_path(self.norm2(self.mlp(x)).astype(self.dtype))
```

swin.py's **local** patch blocks (these are separate classes from `layers.py`'s — same island/cast logic):

`PatchMerging.__init__` (lines 273-279):

```python
    def __init__(self, input_resolution, dim, param_dtype="float32", dtype="float32",
                 *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.reduction = nnx.Linear(4 * dim, 2 * dim, use_bias=False,
                                    kernel_init=TRUNC_NORMAL, dtype=dtype,
                                    param_dtype=param_dtype, rngs=rngs)
        # fp32 island; output feeds `reduction` on the same line (self-healing)
        self.norm = nnx.LayerNorm(4 * dim, epsilon=LN_EPS, dtype=jnp.float32,
                                  param_dtype=param_dtype, rngs=rngs)
```

`PatchExpand.__init__` and final line of `__call__` (lines 295-310):

```python
    def __init__(self, input_resolution, dim, dim_scale=2, param_dtype="float32",
                 dtype="float32", *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.dtype = dtype
        self.expand = (nnx.Linear(dim, 2 * dim, use_bias=False, kernel_init=TRUNC_NORMAL,
                                  dtype=dtype, param_dtype=param_dtype, rngs=rngs)
                       if dim_scale == 2 else Identity())
        self.norm = nnx.LayerNorm(dim // dim_scale, epsilon=LN_EPS, dtype=jnp.float32,
                                  param_dtype=param_dtype, rngs=rngs)
```

and its `__call__`'s return becomes:

```python
        # fp32 norm island exits here: consumers (stage residuals, skip concat)
        # do not self-heal, so emit the compute dtype
        return self.norm(x.reshape(B, -1, C // 4)).astype(self.dtype)
```

`FinalPatchExpand.__init__` (lines 314-322) — no exit cast (fp32 tail into the output conv), `__call__` unchanged:

```python
    def __init__(self, input_resolution, patch_size, dim, param_dtype="float32",
                 dtype="float32", *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.patch_size = tuple(patch_size)
        self.output_dim = dim
        self.expand = nnx.Linear(dim, self.patch_size[0] * self.patch_size[1] * dim,
                                 use_bias=False, kernel_init=TRUNC_NORMAL, dtype=dtype,
                                 param_dtype=param_dtype, rngs=rngs)
        # fp32 island with NO exit cast: sole consumer is the fp32 output conv
        self.norm = nnx.LayerNorm(dim, epsilon=LN_EPS, dtype=jnp.float32,
                                  param_dtype=param_dtype, rngs=rngs)
```

`PatchEmbed.__init__` and `__call__` (lines 337-355):

```python
    def __init__(self, params, *, rngs):
        self.img_size = params.img_size
        self.dtype = params.dtype
        self.num_patches = params.patches_resolution[0] * params.patches_resolution[1]
        self.proj = nnx.Conv(params.in_channels, params.embed_dim,
                             kernel_size=tuple(params.patch_size),
                             strides=tuple(params.patch_size), padding="VALID",
                             dtype=params.dtype, param_dtype=params.param_dtype,
                             rngs=rngs)
        self.norm = (nnx.LayerNorm(params.embed_dim, epsilon=LN_EPS, dtype=jnp.float32,
                                   param_dtype=params.param_dtype, rngs=rngs)
                     if params.patch_embed_norm else None)

    def __call__(self, x):  # (B, H, W, in_channels) channels-last
        B, H, W, C = x.shape
        assert (H, W) == self.img_size
        x = self.proj(x)                   # (B, Ph, Pw, embed_dim)
        x = x.reshape(B, -1, x.shape[-1])  # row-major flatten
        if self.norm is not None:
            # fp32 norm island exits straight into stage residuals: emit compute dtype
            x = self.norm(x).astype(self.dtype)
        return x
```

`EncoderStage.__init__` (lines 370-372) / `DecoderStage.__init__` (lines 388-390): add `dtype=params.dtype,` next to the existing `param_dtype=params.param_dtype,` in the `PatchMerging` / `PatchExpand` constructions.

`SwinEncoder.__init__` (lines 423-424): final norm fp32 endpoint:

```python
        self.norm = nnx.LayerNorm(self.num_features, epsilon=LN_EPS, dtype=jnp.float32,
                                  param_dtype=params.param_dtype, rngs=rngs)
```

`SwinEncoder.__call__` (line 427): `x = jnp.asarray(x, dtype=jnp.float32)`.

`SwinDecoder.__init__` (lines 449-471): add `dtype=params.dtype,` to the `concat_back_dim` Linear, both `PatchExpand` sites, and `FinalPatchExpand`; pin `norm_up` and the output conv:

```python
        self.norm_up = nnx.LayerNorm(params.embed_dim, epsilon=LN_EPS, dtype=jnp.float32,
                                     param_dtype=params.param_dtype, rngs=rngs)
        self.up = FinalPatchExpand(pr, patch_size=params.patch_size,
                                   dim=params.embed_dim, dtype=params.dtype,
                                   param_dtype=params.param_dtype, rngs=rngs)
        # emit-fp32 endpoint: constructed fp32 (never a post-hoc astype)
        self.output = nnx.Conv(params.embed_dim, params.out_channels, kernel_size=(1, 1),
                               use_bias=False, dtype=jnp.float32,
                               param_dtype=params.param_dtype, rngs=rngs)
```

Update `tests/test_model.py:271-274` (`test_flat_param_dtype_propagates`):

```python
    x = jax.random.normal(jax.random.key(0), (2, *p.img_size, 2))
    y = model(x)
    # models emit fp32: the output conv is an fp32 island regardless of knobs
    assert y.dtype == jnp.float32 and y.shape == (2, p.img_size[0], p.img_size[1], 3)
    assert bool(jnp.isfinite(y).all())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_precision.py tests/test_model.py -q`
Expected: PASS.

- [ ] **Step 5: RED-verify the flat casts**

Same discipline as Task 3 Step 5, in `swin.py`: (1) drop the `norm1` residual cast in `SwinBlock.__call__` → flat stream spy fails; (2) drop the local `PatchExpand` exit cast → flat stream spy fails; (3) softmax pre-cast → flat softmax spy fails. Revert each.

- [ ] **Step 6: Full suite, then commit**

Run: `uv run pytest tests/ -q` — Expected: PASS.

```bash
git add src/heal_swin_nnx/models/swin.py tests/test_precision.py tests/test_model.py
git commit -m "feat: thread compute dtype through SwinUnet (fp32 islands + leak casts)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: HealConv — threading (dwconv mixer, no attention)

**Files:**
- Modify: `src/heal_swin_nnx/models/healconv.py` (`HealConvBlock`:153-163, 221-223, `ConvEncoderStage`:241-242, `ConvDecoderStage`:257-259, `HealConvEncoder`:277-279, 293-297, `HealConvDecoder`:319-340)
- Modify: `tests/test_healconv.py:207-210` (output-dtype assertion)
- Test: `tests/test_precision.py` (append)

**Interfaces:**
- Consumes: Task 2 shared-block `dtype=` kwargs; Task 3 helpers.
- Produces: `make_healconv(**over) -> (HealConv, HealConvParams)` in `tests/test_precision.py` (used by Task 6).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_precision.py`:

```python
# --- HealConv ---------------------------------------------------------------

from heal_swin_nnx.models.healconv import HealConv, HealConvBlock, HealConvParams


def make_healconv(**over):
    kw = dict(nside=16, in_channels=3, out_channels=2, base_pixels=(0, 1, 2, 3),
              embed_dim=16, depths=(2, 2), drop_path_rate=0.0)
    kw.update(over)
    p = HealConvParams(**kw)
    return HealConv(p, rngs=nnx.Rngs(0)), p


def test_healconv_master_weights_fp32_under_bf16_compute():
    model, _ = make_healconv(dtype="bfloat16")
    for path, v in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        assert v[...].dtype == jnp.float32, path


def test_healconv_output_and_grads_fp32():
    model, p = make_healconv(dtype="bfloat16")
    model.eval()
    x = _smooth_input(jax.random.key(0), p.npix, 3)
    y = model(x)
    assert y.dtype == jnp.float32 and bool(jnp.isfinite(y).all())
    tokens, _ = model.encoder(x)
    assert tokens.dtype == jnp.float32
    grads = nnx.grad(lambda m: jnp.mean(m(x) ** 2))(model)
    for path, g in nnx.to_flat_state(grads):
        joined = "/".join(str(q) for q in path)
        assert g[...].dtype == jnp.float32, joined


@pytest.mark.parametrize("patch_embed_norm", [False, True])
def test_healconv_stream_is_bf16_inside_blocks(monkeypatch, patch_embed_norm):
    seen = _block_entry_spy(monkeypatch, HealConvBlock)
    model, p = make_healconv(dtype="bfloat16", patch_embed_norm=patch_embed_norm)
    model.eval()
    model(_smooth_input(jax.random.key(0), p.npix, 3))
    assert len(seen) >= 6
    assert all(dt == jnp.bfloat16 for dt in seen), seen
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_precision.py -q -k healconv`
Expected: stream spy FAILS (float32 entries).

- [ ] **Step 3: Implement in `healconv.py`**

`HealConvBlock.__init__` (lines 153-163): add `self.dtype = params.dtype` before `self.dwconv`; thread the conv and Mlp; norms fp32:

```python
        self.dtype = params.dtype
        self.dwconv = nnx.Conv(dim, dim, kernel_size=(self.grid_size, self.grid_size),
                               feature_group_count=dim, padding="SAME",
                               use_bias=params.conv_bias, kernel_init=TRUNC_NORMAL,
                               dtype=params.dtype, param_dtype=params.param_dtype,
                               rngs=rngs)
        self.norm1 = nnx.LayerNorm(dim, epsilon=LN_EPS, dtype=jnp.float32,
                                   param_dtype=params.param_dtype, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(dim, epsilon=LN_EPS, dtype=jnp.float32,
                                   param_dtype=params.param_dtype, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * params.mlp_ratio), drop=params.drop_rate,
                       dtype=params.dtype, param_dtype=params.param_dtype, rngs=rngs)
```

(`_apply_validity` needs no change: `v = self.validity[...].astype(w.dtype)` already follows the stream, and the 0/1 mask is exact in bf16.)

`HealConvBlock.__call__` (lines 221-223):

```python
    def __call__(self, x):
        # fp32 norm islands exit here: cast BEFORE the residual add — adds do
        # not self-heal, one fp32 summand re-promotes the whole downstream stream
        x = x + self.drop_path(self.norm1(self._mix(x)).astype(self.dtype))
        return x + self.drop_path(self.norm2(self.mlp(x)).astype(self.dtype))
```

`ConvEncoderStage.__init__` (lines 241-242) and `ConvDecoderStage.__init__` (lines 257-259):

```python
        self.downsample = (PatchMerging(dim=dim, dtype=params.dtype,
                                        param_dtype=params.param_dtype,
                                        rngs=rngs) if downsample else None)
```

```python
        self.upsample = (PatchExpand(dim=dim, dim_scale=2, dtype=params.dtype,
                                     param_dtype=params.param_dtype, rngs=rngs)
                         if upsample else None)
```

`HealConvEncoder.__init__` (lines 277-279 and 293-294):

```python
        self.patch_embed = PatchEmbed(params.npix, params.patch_size, params.in_channels,
                                      params.embed_dim, params.patch_embed_norm,
                                      dtype=params.dtype,
                                      param_dtype=params.param_dtype, rngs=rngs)
```

```python
        self.norm = nnx.LayerNorm(self.num_features, epsilon=LN_EPS, dtype=jnp.float32,
                                  param_dtype=params.param_dtype, rngs=rngs)
```

`HealConvEncoder.__call__` (line 297): `x = jnp.asarray(x, dtype=jnp.float32)` (inputs fp32 at the door; the patch-embed conv downcasts).

`HealConvDecoder.__init__` (lines 319-340):

```python
            concat_back_dim.append(
                nnx.Linear(2 * dim, dim, kernel_init=TRUNC_NORMAL, dtype=params.dtype,
                           param_dtype=params.param_dtype, rngs=rngs)
                if i_layer > 0 else Identity())
            if i_layer == 0:
                layers_up.append(PatchExpand(dim=dim, dim_scale=2, dtype=params.dtype,
                                             param_dtype=params.param_dtype, rngs=rngs))
```

```python
        self.norm_up = nnx.LayerNorm(params.embed_dim, epsilon=LN_EPS, dtype=jnp.float32,
                                     param_dtype=params.param_dtype, rngs=rngs)
        self.up = FinalPatchExpand(patch_size=params.patch_size, dim=params.embed_dim,
                                   dtype=params.dtype,
                                   param_dtype=params.param_dtype, rngs=rngs)
        # emit-fp32 endpoint: constructed fp32 (never a post-hoc astype)
        self.output = nnx.Conv(params.embed_dim, params.out_channels, kernel_size=(1,),
                               use_bias=False, dtype=jnp.float32,
                               param_dtype=params.param_dtype, rngs=rngs)
```

Update `tests/test_healconv.py:207-210` (`test_healconv_param_dtype_propagates`):

```python
    x = jax.random.normal(jax.random.key(0), (2, p.npix, 3))
    y = model(x)
    # models emit fp32: the output conv is an fp32 island regardless of knobs
    assert y.dtype == jnp.float32 and y.shape == (2, p.npix, 5)
    assert bool(jnp.isfinite(y).all())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_precision.py tests/test_healconv.py -q`
Expected: PASS.

- [ ] **Step 5: RED-verify the residual casts**

Drop the `norm1` cast in `HealConvBlock.__call__` → `uv run pytest tests/test_precision.py -q -k "healconv and stream"` FAILS. Revert.

- [ ] **Step 6: Full suite, then commit**

Run: `uv run pytest tests/ -q` — Expected: PASS.

```bash
git add src/heal_swin_nnx/models/healconv.py tests/test_precision.py tests/test_healconv.py
git commit -m "feat: thread compute dtype through HealConv (fp32 islands + leak casts)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Calibrated drift lock (measure, then lock two-sided bounds)

**Files:**
- Test: `tests/test_precision.py` (append)

**Interfaces:**
- Consumes: `make_healswin`, `make_flat`, `make_healconv`, `_smooth_input`, `_smooth_input_2d`, `_rel_err` from Tasks 3-5.
- Produces: `DRIFT_BOUND` dict constant in `tests/test_precision.py` holding the measured per-model worst-case relative error.

- [ ] **Step 1: Run the assessment (measure, don't guess)**

From the repo root:

```bash
uv run python - <<'EOF'
import sys; sys.path.insert(0, ".")
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
from tests.test_precision import (make_healswin, make_flat, make_healconv,
                                  _smooth_input, _smooth_input_2d, _rel_err)

CASES = {
    "healswin": (make_healswin, lambda p, s: _smooth_input(jax.random.key(s), p.npix, 3)),
    "swin": (make_flat, lambda p, s: _smooth_input_2d(jax.random.key(s), p.img_size, 2)),
    "healconv": (make_healconv, lambda p, s: _smooth_input(jax.random.key(s), p.npix, 3)),
}
for name, (make, x_of) in CASES.items():
    m32, p = make(dtype="float32"); m32.eval()
    m16, _ = make(dtype="bfloat16"); m16.eval()   # same seed -> same master weights
    errs = sorted(_rel_err(m32(x), m16(x)) for x in (x_of(p, s) for s in range(10)))
    print(f"{name}: median={errs[len(errs)//2]:.4g} max={errs[-1]:.4g}")
EOF
```

Record the printed numbers — they go into `DRIFT_BOUND` below and into the commit message. Sanity expectations: max in the 3e-3 … 6e-2 range (GenSBI measured 2.6% over a 9-block stack). If a model's max exceeds ~0.1, do **not** lock it — that signals a leak or a missing island; debug first (check the spy tests, then bisect stages by comparing `model.encoder(x)` errors).

- [ ] **Step 2: Write the locked test**

Append to `tests/test_precision.py`, filling the three values from Step 1's output (the values shown are the *shape* of the entry — replace them with the measured numbers):

```python
# --- calibrated drift lock --------------------------------------------------

# Measured on CPU, 2026-07-20, seeds 0-9, smooth sinusoid inputs, the tiny
# fixtures above (same-seed fp32 master weights): per-model max relative error
# between dtype="float32" and dtype="bfloat16" forwards. If this test fails
# after a legitimate numerics change, re-run the assessment block from
# docs/superpowers/plans/2026-07-20-compute-dtype.md Task 6 and re-calibrate.
DRIFT_BOUND = {
    "healswin": 0.0,   # <- replace with measured max
    "swin": 0.0,       # <- replace with measured max
    "healconv": 0.0,   # <- replace with measured max
}


@pytest.mark.parametrize("name", ["healswin", "swin", "healconv"])
def test_bf16_drift_within_calibrated_band(name):
    make, x_of = {
        "healswin": (make_healswin,
                     lambda p, s: _smooth_input(jax.random.key(s), p.npix, 3)),
        "swin": (make_flat,
                 lambda p, s: _smooth_input_2d(jax.random.key(s), p.img_size, 2)),
        "healconv": (make_healconv,
                     lambda p, s: _smooth_input(jax.random.key(s), p.npix, 3)),
    }[name]
    m32, p = make(dtype="float32")
    m16, _ = make(dtype="bfloat16")     # same rngs seed -> identical master weights
    m32.eval(); m16.eval()
    errs = [_rel_err(m32(x), m16(x)) for x in (x_of(p, s) for s in range(10))]
    bound = DRIFT_BOUND[name]
    # upper lock: mixed-precision computation quality regressed (dropped cast,
    # removed island, changed accumulation)
    assert max(errs) < bound * 3, (name, errs)
    # lower canary: near-zero drift means bf16 compute silently stopped
    # happening — the knob is dead while every tolerance test still passes
    assert max(errs) > bound / 50, (name, errs)
```

- [ ] **Step 3: Run to verify it passes with the calibrated values**

Run: `uv run pytest tests/test_precision.py -q -k drift`
Expected: PASS (three cases). If a bound was left at `0.0` the test fails loudly — that's intended (unfilled calibration cannot pass).

- [ ] **Step 4: RED-verify both directions**

1. Upper: in `healswin.py`, temporarily hard-code the softmax exit cast `.astype(self.dtype)` → `.astype(jnp.bfloat16)`. This corrupts the **fp32 reference model too** (its softmax output gets rounded to bf16), so the fp32↔bf16 gap grows — run the drift test and confirm the healswin upper bound fails. If it doesn't, the bound is too slack: reduce the `* 3` multiplier to the largest value that fails here and still passes Step 3 (record what you chose). Revert.
   (Note: *removing a leak cast* is NOT an upper-bound probe — a leak makes the bf16 model run partly in fp32, which *shrinks* the drift. Leaks are the spy tests' and the lower canary's territory.)
2. Lower: temporarily construct `m16` with `dtype="float32"` inside the test → the canary must fail (drift collapses to ~0). Revert.

- [ ] **Step 5: Commit**

```bash
git add tests/test_precision.py
git commit -m "test: calibrated two-sided bf16 drift lock for all three models

Measured (CPU, seeds 0-9, smooth inputs): <paste Step 1 output here>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Flip the default to bf16 + existing-suite triage

**Files:**
- Modify: `src/heal_swin_nnx/models/healswin.py`, `swin.py`, `healconv.py` (the `dtype` field default + staging comment)
- Modify: `src/heal_swin_nnx/layers.py` and `src/heal_swin_nnx/models/swin.py` (shared/local block kwarg defaults)
- Modify: `tests/test_params.py`, `tests/test_model.py`, `tests/test_healconv.py` (triage)
- Modify: `CLAUDE.md` (one-line precision note)

**Interfaces:**
- Consumes: everything above.
- Produces: `dtype` default `"bfloat16"` on all three Params dataclasses and all block kwargs — the spec's end state.

- [ ] **Step 1: Write the failing default-assertion test**

Append to the compute-dtype section of `tests/test_params.py`:

```python
@pytest.mark.parametrize("mk", PARAMS_FACTORIES)
def test_precision_defaults(mk):
    p = mk()
    assert p.dtype == "bfloat16"        # bf16 compute by default
    assert p.param_dtype == "float32"   # fp32 master weights by default
```

Run: `uv run pytest tests/test_params.py -q -k precision_defaults`
Expected: FAIL (`dtype == "float32"`).

- [ ] **Step 2: Flip the defaults**

In each of the three Params dataclasses, the precision group becomes:

```python
    # precision
    param_dtype: str = "float32"     # parameter storage; any DTypeLike, stored as name
    dtype: str = "bfloat16"          # compute/matmul dtype; fp32 islands (norms,
                                     # softmax, RoPE, final projections) are
                                     # knob-independent — see the compute-dtype spec
```

In `layers.py`, change every shared-block signature default `dtype="float32"` → `dtype="bfloat16"` (`Mlp`, `PatchMerging`, `PatchExpand`, `FinalPatchExpand`, `PatchEmbed`). Same for swin.py's local `PatchMerging`, `PatchExpand`, `FinalPatchExpand` kwargs. (Models pass `params.dtype` explicitly everywhere; these defaults only affect direct block construction in tests.)

- [ ] **Step 3: Run the full suite and triage**

Run: `uv run pytest tests/ -q`

Triage rules (from the spec — every failure must land in exactly one bucket):

- **Exactness tests** (`np.testing.assert_array_equal` / bitwise comparisons through a forward or `_mix`): pin fp32. Known cases in `tests/test_healconv.py` — the delta-kernel identity tests (`test_identity_kernel_mix_is_identity_unshifted` and any shifted siblings that call `blk._mix` or compare exactly): change their `make_block(...)` / `tiny_conv_params(...)` calls to include `dtype="float32"`, e.g.

  ```python
      blk, p, N = make_block(shifted=False, kernel_size=kernel_size, dtype="float32")
  ```

  with the comment `# exactness test: bf16 compute would round the pass-through`.
- **Structural-equivalence tests** now exercising bf16 (`test_jit_matches_eager`, `test_batch_independence` in `tests/test_model.py`, and any healconv/seam equivalents comparing two runs of the *same* bf16 model): loosen `rtol`/`atol` from `1e-4` to `2e-2` and update the tolerance comment to say bf16 reduction-order drift, e.g.

  ```python
      # tolerance covers bf16-compute reduction-order drift across jit fusion
      # (default dtype is bfloat16); real bugs are >> 2e-2
      np.testing.assert_allclose(..., rtol=2e-2, atol=2e-2)
  ```

  `test_remat_matches_no_remat` replays the identical graph and should stay bit-tight; if it drifts, treat it the same way.
- **Finiteness/shape tests** (`test_forward_full_sphere_and_south_cap`, grads-finite, shapes): must pass unchanged. A failure here is a **genuine bug** — stop and debug, do not pin.
- **Geometry/permutation/topology tests** (`test_topology.py`, `test_shifting.py`, `test_windowing.py`, `test_seam_geometry.py` construction-time checks): dtype-blind, must pass unchanged. A failure is a genuine bug.

Iterate until: `uv run pytest tests/ -q` → all PASS.

- [ ] **Step 4: Document the knob in CLAUDE.md**

In `CLAUDE.md`, in the Architecture section's `models/healswin.py` bullet, after the sentence about `HealSwinParams` validation, add:

```markdown
  Precision: `param_dtype` (storage, fp32 default) and `dtype` (compute, bf16
  default) are independent knobs; norms/softmax/RoPE/final projections are
  knob-independent fp32 islands and models always emit fp32 — see
  `docs/superpowers/specs/2026-07-20-compute-dtype-design.md` before touching
  any dtype or cast.
```

- [ ] **Step 5: Final full-suite run and commit**

Run: `uv run pytest tests/ -q` — Expected: PASS, no skips introduced.

```bash
git add -A
git commit -m "feat!: default compute dtype flips to bfloat16 (fp32 master weights)

Existing exactness tests pin dtype=float32; structural-equivalence
tolerances widened to bf16 scale; drift lock + spy tests guard the contract.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review (run after writing, before handoff)

1. **Spec coverage:** config surface → Task 1; contract table → Tasks 2-5 (islands, door, endpoints), Task 7 (defaults); attention island → Tasks 3-4; leak map → Tasks 2-5 (each cast has a RED-verify step); threading mechanics → Tasks 2-5; testing §6.1 → Task 1, §6.2-6.3 → Tasks 3-5, §6.4 → Task 6, §6.5 → Task 7. fp32-fallback invariant → the "staging invariant" full-suite step in every task (default stays fp32 until Task 7, so the whole suite continuously proves `dtype="float32"` ≡ current behavior).
2. **Type consistency:** blocks take `dtype` as a canonical string (same convention as `param_dtype`); `canonical_float_dtype(value, field_name)`; helper names `make_healswin`/`make_flat`/`make_healconv`, `_smooth_input`, `_smooth_input_2d`, `_rel_err`, `_block_entry_spy` are defined in Task 3 (Task 4 adds `make_flat`/`_smooth_input_2d`, Task 5 adds `make_healconv`) and consumed by Tasks 4-6 with those exact names.
3. **Placeholders:** the only deliberately unfilled values are the three `DRIFT_BOUND` entries — they are measured in Task 6 Step 1 and the test fails loudly if left at 0.0.
