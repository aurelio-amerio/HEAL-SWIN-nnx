# `param_dtype` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a flux1-style `param_dtype` knob to `HealSwinParams`, `HealConvParams`, and `SwinParams`, threaded into every layer so parameters (and therefore computation) run in the chosen precision, with a float32 default that keeps current behavior bit-identical.

**Architecture:** One canonicalizing helper in `layers.py` turns any DTypeLike into its string name (JSON-safe). Each Params dataclass stores that string; model constructors forward it to every `nnx.Linear`/`nnx.LayerNorm`/`nnx.Conv` as `param_dtype` and to manually created `nnx.Param`s. Encoders cast the input to `param_dtype` at entry. Buffers and RoPE angle math stay float32; float buffers are cast to the activation dtype at their use sites so they never silently promote bf16 activations back to f32. Spec: `docs/superpowers/specs/2026-07-20-param-dtype-design.md`.

**Tech Stack:** JAX / Flax NNX, pytest (`uv run pytest`, CPU-forced via pyproject).

## Global Constraints

- Pure JAX/Flax: **no PyTorch imports anywhere in `src/`**.
- Default `param_dtype="float32"` must leave all existing tests passing unchanged — no tolerance retuning.
- `json.dumps(dataclasses.asdict(params))` must keep working for all three Params classes.
- Exceptions that stay float32 (from the spec): RoPE frequencies/tables (`rope_freqs`, `rope_table`, `rope_coords`) and all Buffers (index permutations, attention/validity masks).
- Tests are invariant-style (dtype propagation, buffer-dtype preservation, finite outputs) — no golden values.
- Run tests with `uv run pytest tests/<file> -q` (pytest is preconfigured: `JAX_PLATFORMS=cpu`, `-n 2`).
- All paths below are relative to the repo root `/lustre/ific.uv.es/ml/ific088/github/HEAL-SWIN-nnx`.

---

### Task 1: `param_dtype` field on the three Params dataclasses

**Files:**
- Modify: `src/heal_swin_nnx/layers.py` (add `canonical_float_dtype` helper, after `l2_normalize`, ~line 24)
- Modify: `src/heal_swin_nnx/models/healswin.py` (`HealSwinParams`, lines 24–128; import at lines 15–18)
- Modify: `src/heal_swin_nnx/models/healconv.py` (`HealConvParams`, lines 26–113; import at lines 19–22)
- Modify: `src/heal_swin_nnx/models/swin.py` (`SwinParams`, lines 22–75; import at lines 11–13)
- Test: `tests/test_params.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `canonical_float_dtype(value) -> str` in `heal_swin_nnx.layers` (raises `ValueError` for non-float or unparseable dtypes); field `param_dtype: str = "float32"` on `HealSwinParams`, `HealConvParams`, `SwinParams`, canonicalized in `__post_init__`. Later tasks read `params.param_dtype` (a string like `"bfloat16"` — every jnp/nnx API accepts it).

- [ ] **Step 1: Sanity-check the dtype-name round trip**

Run:
```bash
uv run python -c "import jax.numpy as jnp; print(jnp.dtype('bfloat16').name, jnp.issubdtype(jnp.dtype('bfloat16'), jnp.floating))"
```
Expected: `bfloat16 True`

- [ ] **Step 2: Write the failing tests**

Append at the **end** of `tests/test_params.py` — the file imports `SwinParams` at line 123 and `HealConvParams` at line 162 (section-style mid-file imports), so all three names are in scope there. Add `import jax.numpy as jnp` next to the existing imports at the top:

```python
# --- param_dtype ------------------------------------------------------------

PARAMS_FACTORIES = [
    lambda **over: HealSwinParams(nside=16, in_channels=1, out_channels=1, **over),
    lambda **over: HealConvParams(nside=16, in_channels=1, out_channels=1,
                                  depths=(2, 2), **over),
    lambda **over: SwinParams(img_size=(32, 64), in_channels=1, out_channels=1,
                              embed_dim=16, depths=(2, 2), num_heads=(2, 4), **over),
]


@pytest.mark.parametrize("mk", PARAMS_FACTORIES)
def test_param_dtype_defaults_and_canonicalization(mk):
    assert mk().param_dtype == "float32"
    for spec in ("bfloat16", jnp.bfloat16, jnp.dtype(jnp.bfloat16)):
        p = mk(param_dtype=spec)
        assert p.param_dtype == "bfloat16"
        json.dumps(dataclasses.asdict(p))  # must stay serializable


@pytest.mark.parametrize("mk", PARAMS_FACTORIES)
@pytest.mark.parametrize("bad", ["int32", "bool", "not_a_dtype", 7])
def test_param_dtype_rejects_non_floats(mk, bad):
    with pytest.raises(ValueError):
        mk(param_dtype=bad)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_params.py -q`
Expected: the new tests FAIL with `TypeError: ... got an unexpected keyword argument 'param_dtype'` (default-value test fails with `AttributeError: ... has no attribute 'param_dtype'`); all pre-existing tests PASS.

- [ ] **Step 4: Add the helper to `layers.py`**

Insert after `l2_normalize` (after line 23):

```python
def canonical_float_dtype(value):
    """Canonicalize a DTypeLike into its dtype name ("float32", "bfloat16", ...).

    Params dataclasses store dtypes as canonical strings so
    ``json.dumps(dataclasses.asdict(params))`` keeps working; every jnp/nnx
    API accepts the string form. Floating dtypes only."""
    try:
        dt = jnp.dtype(value)
    except TypeError as e:
        raise ValueError("param_dtype must be a floating DTypeLike, got %r"
                         % (value,)) from e
    if not jnp.issubdtype(dt, jnp.floating):
        raise ValueError("param_dtype must be a floating dtype, got %r" % (value,))
    return dt.name
```

- [ ] **Step 5: Add the field to all three dataclasses**

In `src/heal_swin_nnx/models/healswin.py`, extend the `heal_swin_nnx.layers` import list (lines 15–18) with `canonical_float_dtype`, then in `HealSwinParams` add after `use_checkpoint: bool = False` (line 55):

```python
    # precision
    param_dtype: str = "float32"     # any DTypeLike accepted; stored as the dtype name
```

and in `__post_init__`, right after the tuple coercions (after line 62 `self.num_heads = tuple(self.num_heads)`):

```python
        self.param_dtype = canonical_float_dtype(self.param_dtype)
```

In `src/heal_swin_nnx/models/healconv.py`, extend the `heal_swin_nnx.layers` import list (lines 19–22) with `canonical_float_dtype`, add the same two-line field block after `use_checkpoint: bool = False` (line 54), and the same `__post_init__` line right after `self.depths = tuple(self.depths)` (line 59).

In `src/heal_swin_nnx/models/swin.py`, extend the `heal_swin_nnx.layers` import list (lines 11–13) with `canonical_float_dtype`, add the same field block after `use_checkpoint: bool = False` (line 45), and the same `__post_init__` line right after `self.num_heads = tuple(self.num_heads)` (line 52).

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_params.py -q`
Expected: PASS (all tests, old and new).

- [ ] **Step 7: Commit**

```bash
git add src/heal_swin_nnx/layers.py src/heal_swin_nnx/models/healswin.py \
        src/heal_swin_nnx/models/healconv.py src/heal_swin_nnx/models/swin.py \
        tests/test_params.py
git commit -m "feat: param_dtype config field on all three Params dataclasses"
```

---

### Task 2: thread `param_dtype` through the shared blocks in `layers.py`

**Files:**
- Modify: `src/heal_swin_nnx/layers.py` (`Mlp` line 48, `PatchMerging` line 105, `PatchExpand` line 120, `FinalPatchExpand` line 136, `PatchEmbed` line 152)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: nothing from Task 1 (the kwarg is independent of the dataclasses).
- Produces: every shared block gains a keyword `param_dtype="float32"` in this exact position:
  - `Mlp(in_features, hidden_features=None, out_features=None, drop=0.0, param_dtype="float32", *, rngs)`
  - `PatchMerging(dim, dim_scale=2, param_dtype="float32", *, rngs)`
  - `PatchExpand(dim, dim_scale=2, param_dtype="float32", *, rngs)`
  - `FinalPatchExpand(patch_size, dim, param_dtype="float32", *, rngs)`
  - `PatchEmbed(npix, patch_size, in_channels, embed_dim, norm=False, param_dtype="float32", *, rngs)`

  Tasks 3–4 call these with `param_dtype=params.param_dtype`. `__call__` methods are unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model.py`, and extend its layers import (line 8) to
`from heal_swin_nnx.layers import (DropPath, FinalPatchExpand, Identity, Mlp, PatchEmbed, PatchExpand, PatchMerging)`:

```python
def test_shared_layers_accept_param_dtype():
    rngs = nnx.Rngs(0)
    mods = [Mlp(8, 32, param_dtype="bfloat16", rngs=rngs),
            PatchMerging(8, param_dtype="bfloat16", rngs=rngs),
            PatchExpand(8, param_dtype="bfloat16", rngs=rngs),
            FinalPatchExpand(4, 8, param_dtype="bfloat16", rngs=rngs),
            PatchEmbed(64, 4, 3, 8, norm=True, param_dtype="bfloat16", rngs=rngs)]
    for m in mods:
        flat = list(nnx.to_flat_state(nnx.state(m, nnx.Param)))
        assert len(flat) > 0
        for path, v in flat:
            # v[...] (not .value — deprecated in this flax version) reads the array
            assert v[...].dtype == jnp.bfloat16, (type(m).__name__, path)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_model.py::test_shared_layers_accept_param_dtype -q`
Expected: FAIL with `TypeError: ... got an unexpected keyword argument 'param_dtype'`

- [ ] **Step 3: Add the kwarg to the five constructors**

Replace each `__init__` in `src/heal_swin_nnx/layers.py` (bodies of `__call__` untouched):

```python
class Mlp(nnx.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0,
                 param_dtype="float32", *, rngs):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nnx.Linear(in_features, hidden_features, kernel_init=TRUNC_NORMAL,
                              param_dtype=param_dtype, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_features, out_features, kernel_init=TRUNC_NORMAL,
                              param_dtype=param_dtype, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)
```

```python
class PatchMerging(nnx.Module):
    """Merge 4 nested pixels into 1: (B, N, C) -> (B, N/4, dim_scale*C)."""

    def __init__(self, dim, dim_scale=2, param_dtype="float32", *, rngs):
        self.reduction = nnx.Linear(4 * dim, dim_scale * dim, use_bias=False,
                                    kernel_init=TRUNC_NORMAL, param_dtype=param_dtype,
                                    rngs=rngs)
        self.norm = nnx.LayerNorm(4 * dim, epsilon=LN_EPS, param_dtype=param_dtype,
                                  rngs=rngs)
```

```python
class PatchExpand(nnx.Module):
    """Expand 1 pixel into 4 nested pixels: (B, N, C) -> (B, 4N, C*dim_scale/4)."""

    def __init__(self, dim, dim_scale=2, param_dtype="float32", *, rngs):
        self.expand = (nnx.Linear(dim, dim_scale * dim, use_bias=False,
                                  kernel_init=TRUNC_NORMAL, param_dtype=param_dtype,
                                  rngs=rngs)
                       if dim_scale != 1 else Identity())
        self.norm = nnx.LayerNorm(dim * dim_scale // 4, epsilon=LN_EPS,
                                  param_dtype=param_dtype, rngs=rngs)
```

```python
class FinalPatchExpand(nnx.Module):
    """Undo the patch embedding's downsampling: (B, N, C) -> (B, N*patch_size, C)."""

    def __init__(self, patch_size, dim, param_dtype="float32", *, rngs):
        self.patch_size = patch_size
        self.expand = nnx.Linear(dim, patch_size * dim, use_bias=False,
                                 kernel_init=TRUNC_NORMAL, param_dtype=param_dtype,
                                 rngs=rngs)
        self.norm = nnx.LayerNorm(dim, epsilon=LN_EPS, param_dtype=param_dtype,
                                  rngs=rngs)
```

```python
class PatchEmbed(nnx.Module):
    """Non-overlapping 1D patch embedding over the nested pixel sequence."""

    def __init__(self, npix, patch_size, in_channels, embed_dim, norm=False,
                 param_dtype="float32", *, rngs):
        self.npix = npix
        self.num_patches = npix // patch_size
        self.proj = nnx.Conv(in_channels, embed_dim,
                             kernel_size=(patch_size,), strides=(patch_size,),
                             padding="VALID", param_dtype=param_dtype, rngs=rngs)
        self.norm = (nnx.LayerNorm(embed_dim, epsilon=LN_EPS, param_dtype=param_dtype,
                                   rngs=rngs) if norm else None)
```

- [ ] **Step 4: Run the test to verify it passes, plus the whole file for regressions**

Run: `uv run pytest tests/test_model.py -q`
Expected: PASS (the f32 default makes every existing test bit-identical).

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/layers.py tests/test_model.py
git commit -m "feat: thread param_dtype through shared layer blocks"
```

---

### Task 3: HealSwin threading, input cast, and mask cast

**Files:**
- Modify: `src/heal_swin_nnx/models/healswin.py` (`WindowAttention` 139–202, `HealSwinBlock` 205–216, `EncoderStage` 261–267, `DecoderStage` 277–283, `HealSwinEncoder` 297–329, `HealSwinDecoder` 332–362)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `params.param_dtype: str` (Task 1); `param_dtype=` kwarg on `Mlp`, `PatchMerging`, `PatchExpand`, `FinalPatchExpand`, `PatchEmbed` (Task 2).
- Produces: `HealSwin`/`HealSwinEncoder`/`HealSwinDecoder` fully honor `params.param_dtype`. No signature changes — construction still `HealSwin(params, rngs=rngs)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model.py`, extending its first import line to
`from heal_swin_nnx import Buffer, HealSwin, HealSwinEncoder, HealSwinParams, SwinParams, SwinUnet`:

```python
def _param_dtypes(model):
    return {"/".join(str(q) for q in path): v[...].dtype
            for path, v in nnx.to_flat_state(nnx.state(model, nnx.Param))}


def _buffer_dtypes(model):
    return {"/".join(str(q) for q in path): v[...].dtype
            for path, v in nnx.to_flat_state(nnx.state(model, Buffer))}


@pytest.mark.parametrize("pos_embed", ["rope_mixed", "rope_axial", "rel_bias"])
def test_healswin_param_dtype_propagates(pos_embed):
    model, p = tiny_hp(param_dtype="bfloat16", pos_embed=pos_embed)
    model.eval()
    for path, dtype in _param_dtypes(model).items():
        # rope_freqs feeds the f32 RoPE angle computation and stays f32 by design
        expected = jnp.float32 if "rope_freqs" in path else jnp.bfloat16
        assert dtype == expected, path

    ref, _ = tiny_hp(pos_embed=pos_embed)          # buffers ignore param_dtype
    assert _buffer_dtypes(model) == _buffer_dtypes(ref)

    x = jax.random.normal(jax.random.key(0), (2, p.npix, 3))
    y = model(x)
    assert y.dtype == jnp.bfloat16 and y.shape == (2, p.npix, 5)
    assert bool(jnp.isfinite(y).all())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_model.py::test_healswin_param_dtype_propagates -q`
Expected: FAIL with an `AssertionError` on a param path whose dtype is `float32` instead of `bfloat16` (Params accepts the field already, but nothing reads it yet).

- [ ] **Step 3: Thread `param_dtype` through `healswin.py`**

In `WindowAttention.__init__` (lines 143–170) make these replacements:

`logit_scale` (line 147):
```python
        self.logit_scale = nnx.Param(
            jnp.full((num_heads, 1, 1), jnp.log(10.0), dtype=params.param_dtype))
```

`relative_position_bias_table` (lines 152–153):
```python
            self.relative_position_bias_table = nnx.Param(
                TRUNC_NORMAL(rngs.params(), ((2 * s - 1) ** 2, num_heads),
                             params.param_dtype))
```

Directly above the `rope_mixed` branch's `self.rope_freqs = ...` (line 159), add the comment line
`# rope_freqs stays f32: it feeds the f32 angle computation (see apply_rope)` — the `rope_freqs`, `rope_coords`, and `rope_table` lines themselves are **unchanged**.

`qkv` / `proj` (lines 166–169):
```python
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=params.qkv_bias,
                              kernel_init=TRUNC_NORMAL, param_dtype=params.param_dtype,
                              rngs=rngs)
        self.attn_drop = nnx.Dropout(params.attn_drop_rate, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=TRUNC_NORMAL,
                               param_dtype=params.param_dtype, rngs=rngs)
```

In `WindowAttention.__call__`, cast the f32 mask buffer to the activation dtype so it can't promote bf16 attention to f32 (lines 194–197):
```python
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.reshape(B_ // nW, nW, self.num_heads, N, N)
                    + mask.astype(attn.dtype)[None, :, None])
            attn = attn.reshape(-1, self.num_heads, N, N)
```

In `HealSwinBlock.__init__` (lines 212–216):
```python
        self.norm1 = nnx.LayerNorm(dim, epsilon=LN_EPS, param_dtype=params.param_dtype,
                                   rngs=rngs)
        self.attn = WindowAttention(params, dim, num_heads, self.window_size, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(dim, epsilon=LN_EPS, param_dtype=params.param_dtype,
                                   rngs=rngs)
        self.mlp = Mlp(dim, int(dim * params.mlp_ratio), drop=params.drop_rate,
                       param_dtype=params.param_dtype, rngs=rngs)
```

In `EncoderStage.__init__` (line 267):
```python
        self.downsample = (PatchMerging(dim=dim, param_dtype=params.param_dtype,
                                        rngs=rngs) if downsample else None)
```

In `DecoderStage.__init__` (line 283):
```python
        self.upsample = (PatchExpand(dim=dim, dim_scale=2,
                                     param_dtype=params.param_dtype, rngs=rngs)
                         if upsample else None)
```

In `HealSwinEncoder.__init__` (lines 305–306 and 320):
```python
        self.patch_embed = PatchEmbed(params.npix, params.patch_size, params.in_channels,
                                      params.embed_dim, params.patch_embed_norm,
                                      param_dtype=params.param_dtype, rngs=rngs)
```
```python
        self.norm = nnx.LayerNorm(self.num_features, epsilon=LN_EPS,
                                  param_dtype=params.param_dtype, rngs=rngs)
```

In `HealSwinEncoder.__call__` (line 322), cast the input at entry — the encoder is the data entry point for both the U-Net and standalone use:
```python
    def __call__(self, x):
        x = jnp.asarray(x, dtype=self.params.param_dtype)
        x = self.patch_embed(x)
```

In `HealSwinDecoder.__init__` (lines 344–348 and 358–362):
```python
            concat_back_dim.append(
                nnx.Linear(2 * dim, dim, kernel_init=TRUNC_NORMAL,
                           param_dtype=params.param_dtype, rngs=rngs)
                if i_layer > 0 else Identity())
            if i_layer == 0:
                layers_up.append(PatchExpand(dim=dim, dim_scale=2,
                                             param_dtype=params.param_dtype, rngs=rngs))
```
```python
        self.norm_up = nnx.LayerNorm(params.embed_dim, epsilon=LN_EPS,
                                     param_dtype=params.param_dtype, rngs=rngs)
        self.up = FinalPatchExpand(patch_size=params.patch_size, dim=params.embed_dim,
                                   param_dtype=params.param_dtype, rngs=rngs)
        self.output = nnx.Conv(params.embed_dim, params.out_channels, kernel_size=(1,),
                               use_bias=False, param_dtype=params.param_dtype, rngs=rngs)
```

- [ ] **Step 4: Run the whole model test file**

Run: `uv run pytest tests/test_model.py -q`
Expected: PASS — the new propagation test and every pre-existing test (f32 default unchanged; `test_jit_matches_eager` tolerances untouched).

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/models/healswin.py tests/test_model.py
git commit -m "feat: thread param_dtype through HealSwin (input cast at encoder entry)"
```

---

### Task 4: HealConv threading, input cast, and validity-mask cast

**Files:**
- Modify: `src/heal_swin_nnx/models/healconv.py` (`HealConvBlock` 126–196, `ConvEncoderStage` 229–234, `ConvDecoderStage` 244–249, `HealConvEncoder` 259–291, `HealConvDecoder` 294–324)
- Test: `tests/test_healconv.py`

**Interfaces:**
- Consumes: `params.param_dtype: str` (Task 1); `param_dtype=` kwarg on shared blocks (Task 2).
- Produces: `HealConv`/`HealConvEncoder`/`HealConvDecoder` fully honor `params.param_dtype`. No signature changes.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_healconv.py`, extending its import (line 7) to
`from heal_swin_nnx.models.healconv import HealConv, HealConvBlock, HealConvParams` and adding
`from heal_swin_nnx.variables import Buffer` below it:

```python
def test_healconv_param_dtype_propagates():
    p = tiny_conv_params(param_dtype="bfloat16")
    model = HealConv(p, rngs=nnx.Rngs(0))
    model.eval()
    for path, v in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        assert v[...].dtype == jnp.bfloat16, path

    ref = HealConv(tiny_conv_params(), rngs=nnx.Rngs(0))  # buffers ignore param_dtype
    bufs = {"/".join(str(q) for q in path): v[...].dtype
            for path, v in nnx.to_flat_state(nnx.state(model, Buffer))}
    ref_bufs = {"/".join(str(q) for q in path): v[...].dtype
                for path, v in nnx.to_flat_state(nnx.state(ref, Buffer))}
    assert bufs == ref_bufs

    x = jax.random.normal(jax.random.key(0), (2, p.npix, 3))
    y = model(x)
    assert y.dtype == jnp.bfloat16 and y.shape == (2, p.npix, 5)
    assert bool(jnp.isfinite(y).all())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_healconv.py::test_healconv_param_dtype_propagates -q`
Expected: FAIL with an `AssertionError` on a param path whose dtype is `float32`.

- [ ] **Step 3: Thread `param_dtype` through `healconv.py`**

In `HealConvBlock.__init__` (lines 149–156):
```python
        self.dwconv = nnx.Conv(dim, dim, kernel_size=(self.grid_size, self.grid_size),
                               feature_group_count=dim, padding="SAME",
                               use_bias=params.conv_bias, kernel_init=TRUNC_NORMAL,
                               param_dtype=params.param_dtype, rngs=rngs)
        self.norm1 = nnx.LayerNorm(dim, epsilon=LN_EPS, param_dtype=params.param_dtype,
                                   rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(dim, epsilon=LN_EPS, param_dtype=params.param_dtype,
                                   rngs=rngs)
        self.mlp = Mlp(dim, int(dim * params.mlp_ratio), drop=params.drop_rate,
                       param_dtype=params.param_dtype, rngs=rngs)
```

In `HealConvBlock._apply_validity` (lines 193–196), cast the f32 validity buffer to the activation dtype (0.0/1.0 are exact in bf16) so the multiply can't promote bf16 activations to f32:
```python
    def _apply_validity(self, w):
        B_, ws, C = w.shape
        v = self.validity[...].astype(w.dtype)       # (nW, ws, 1)
        return (w.reshape(-1, self.num_windows, ws, C) * v[None]).reshape(B_, ws, C)
```

In `ConvEncoderStage.__init__` (line 234):
```python
        self.downsample = (PatchMerging(dim=dim, param_dtype=params.param_dtype,
                                        rngs=rngs) if downsample else None)
```

In `ConvDecoderStage.__init__` (line 249):
```python
        self.upsample = (PatchExpand(dim=dim, dim_scale=2,
                                     param_dtype=params.param_dtype, rngs=rngs)
                         if upsample else None)
```

In `HealConvEncoder.__init__` (lines 267–268 and 282):
```python
        self.patch_embed = PatchEmbed(params.npix, params.patch_size, params.in_channels,
                                      params.embed_dim, params.patch_embed_norm,
                                      param_dtype=params.param_dtype, rngs=rngs)
```
```python
        self.norm = nnx.LayerNorm(self.num_features, epsilon=LN_EPS,
                                  param_dtype=params.param_dtype, rngs=rngs)
```

In `HealConvEncoder.__call__` (line 284), cast the input at entry:
```python
    def __call__(self, x):
        x = jnp.asarray(x, dtype=self.params.param_dtype)
        x = self.patch_embed(x)
```

In `HealConvDecoder.__init__` (lines 306–310 and 320–324):
```python
            concat_back_dim.append(
                nnx.Linear(2 * dim, dim, kernel_init=TRUNC_NORMAL,
                           param_dtype=params.param_dtype, rngs=rngs)
                if i_layer > 0 else Identity())
            if i_layer == 0:
                layers_up.append(PatchExpand(dim=dim, dim_scale=2,
                                             param_dtype=params.param_dtype, rngs=rngs))
```
```python
        self.norm_up = nnx.LayerNorm(params.embed_dim, epsilon=LN_EPS,
                                     param_dtype=params.param_dtype, rngs=rngs)
        self.up = FinalPatchExpand(patch_size=params.patch_size, dim=params.embed_dim,
                                   param_dtype=params.param_dtype, rngs=rngs)
        self.output = nnx.Conv(params.embed_dim, params.out_channels, kernel_size=(1,),
                               use_bias=False, param_dtype=params.param_dtype, rngs=rngs)
```

- [ ] **Step 4: Run the whole HealConv test file**

Run: `uv run pytest tests/test_healconv.py -q`
Expected: PASS — including the delta-kernel identity tests, which set an f32 kernel on the default-f32 models and are unaffected.

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/models/healconv.py tests/test_healconv.py
git commit -m "feat: thread param_dtype through HealConv (validity mask cast at use site)"
```

---

### Task 5: SwinUnet threading and full-suite verification

**Files:**
- Modify: `src/heal_swin_nnx/models/swin.py` (`WindowAttention` 139–203, `SwinBlock` 206–228, local `PatchMerging` 260–265, local `PatchExpand` 280–285, local `FinalPatchExpand` 297–304, local `PatchEmbed` 318–326, `SwinEncoder` 382–410, `SwinDecoder` 413–443)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `params.param_dtype: str` (Task 1); `Mlp(..., param_dtype=...)` from `layers.py` (Task 2). Note `swin.py` has its **own local** `PatchMerging`/`PatchExpand`/`FinalPatchExpand`/`PatchEmbed` classes — this task adds the kwarg to those local classes; the shared ones in `layers.py` were done in Task 2.
- Produces: `SwinUnet`/`SwinEncoder`/`SwinDecoder` fully honor `params.param_dtype`. Local-class signatures become:
  - `PatchMerging(input_resolution, dim, param_dtype="float32", *, rngs)`
  - `PatchExpand(input_resolution, dim, dim_scale=2, param_dtype="float32", *, rngs)`
  - `FinalPatchExpand(input_resolution, patch_size, dim, param_dtype="float32", *, rngs)`
  - `PatchEmbed(params, *, rngs)` (unchanged — reads `params.param_dtype` internally)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model.py` (reuses `_param_dtypes`/`_buffer_dtypes` from Task 3 — if executing this task independently, those helpers are defined in Task 3's Step 1 and already committed):

```python
@pytest.mark.parametrize("pos_embed", ["rope_mixed", "rope_axial", "rel_bias"])
def test_flat_param_dtype_propagates(pos_embed):
    model, p = tiny_flat(param_dtype="bfloat16", pos_embed=pos_embed)
    model.eval()
    for path, dtype in _param_dtypes(model).items():
        expected = jnp.float32 if "rope_freqs" in path else jnp.bfloat16
        assert dtype == expected, path

    ref, _ = tiny_flat(pos_embed=pos_embed)        # buffers ignore param_dtype
    assert _buffer_dtypes(model) == _buffer_dtypes(ref)

    x = jax.random.normal(jax.random.key(0), (2, *p.img_size, 2))
    y = model(x)
    assert y.dtype == jnp.bfloat16 and y.shape == (2, p.img_size[0], p.img_size[1], 3)
    assert bool(jnp.isfinite(y).all())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_model.py::test_flat_param_dtype_propagates -q`
Expected: FAIL with an `AssertionError` on a param path whose dtype is `float32`.

- [ ] **Step 3: Thread `param_dtype` through `swin.py`**

In `WindowAttention.__init__` (lines 143–170), make the same four replacements as HealSwin's attention:

`logit_scale` (line 148):
```python
        self.logit_scale = nnx.Param(
            jnp.full((num_heads, 1, 1), jnp.log(10.0), dtype=params.param_dtype))
```

`relative_position_bias_table` (lines 152–153):
```python
            self.relative_position_bias_table = nnx.Param(
                TRUNC_NORMAL(rngs.params(), (n_rel, num_heads), params.param_dtype))
```

Directly above `self.rope_freqs = ...` (line 159), add the comment line
`# rope_freqs stays f32: it feeds the f32 angle computation (see apply_rope)` — the rope lines themselves are unchanged.

`qkv` / `proj` (lines 166–169):
```python
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=params.qkv_bias,
                              kernel_init=TRUNC_NORMAL, param_dtype=params.param_dtype,
                              rngs=rngs)
        self.attn_drop = nnx.Dropout(params.attn_drop_rate, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=TRUNC_NORMAL,
                               param_dtype=params.param_dtype, rngs=rngs)
```

In `WindowAttention.__call__`, cast the mask (lines 196–199):
```python
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.reshape(B_ // nW, nW, self.num_heads, N, N)
                    + mask.astype(attn.dtype)[None, :, None])
            attn = attn.reshape(-1, self.num_heads, N, N)
```

In `SwinBlock.__init__` (lines 218–222):
```python
        self.norm1 = nnx.LayerNorm(dim, epsilon=LN_EPS, param_dtype=params.param_dtype,
                                   rngs=rngs)
        self.attn = WindowAttention(params, dim, num_heads, self.window_size, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(dim, epsilon=LN_EPS, param_dtype=params.param_dtype,
                                   rngs=rngs)
        self.mlp = Mlp(dim, int(dim * params.mlp_ratio), drop=params.drop_rate,
                       param_dtype=params.param_dtype, rngs=rngs)
```

Local `PatchMerging.__init__` (lines 261–265):
```python
    def __init__(self, input_resolution, dim, param_dtype="float32", *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.reduction = nnx.Linear(4 * dim, 2 * dim, use_bias=False,
                                    kernel_init=TRUNC_NORMAL, param_dtype=param_dtype,
                                    rngs=rngs)
        self.norm = nnx.LayerNorm(4 * dim, epsilon=LN_EPS, param_dtype=param_dtype,
                                  rngs=rngs)
```

Local `PatchExpand.__init__` (lines 281–285):
```python
    def __init__(self, input_resolution, dim, dim_scale=2, param_dtype="float32", *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.expand = (nnx.Linear(dim, 2 * dim, use_bias=False, kernel_init=TRUNC_NORMAL,
                                  param_dtype=param_dtype, rngs=rngs)
                       if dim_scale == 2 else Identity())
        self.norm = nnx.LayerNorm(dim // dim_scale, epsilon=LN_EPS,
                                  param_dtype=param_dtype, rngs=rngs)
```

Local `FinalPatchExpand.__init__` (lines 298–304):
```python
    def __init__(self, input_resolution, patch_size, dim, param_dtype="float32", *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.patch_size = tuple(patch_size)
        self.output_dim = dim
        self.expand = nnx.Linear(dim, self.patch_size[0] * self.patch_size[1] * dim,
                                 use_bias=False, kernel_init=TRUNC_NORMAL,
                                 param_dtype=param_dtype, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, epsilon=LN_EPS, param_dtype=param_dtype,
                                  rngs=rngs)
```

Local `PatchEmbed.__init__` (takes `params` — lines 322–326):
```python
        self.proj = nnx.Conv(params.in_channels, params.embed_dim,
                             kernel_size=tuple(params.patch_size),
                             strides=tuple(params.patch_size), padding="VALID",
                             param_dtype=params.param_dtype, rngs=rngs)
        self.norm = (nnx.LayerNorm(params.embed_dim, epsilon=LN_EPS,
                                   param_dtype=params.param_dtype, rngs=rngs)
                     if params.patch_embed_norm else None)
```

`EncoderStage.__init__` (line 350):
```python
        self.downsample = (PatchMerging(input_resolution, dim=dim,
                                        param_dtype=params.param_dtype, rngs=rngs)
                           if downsample else None)
```

`DecoderStage.__init__` (line 367):
```python
        self.upsample = (PatchExpand(input_resolution, dim=dim, dim_scale=2,
                                     param_dtype=params.param_dtype, rngs=rngs)
                         if upsample else None)
```

`SwinEncoder.__init__` final norm (line 401):
```python
        self.norm = nnx.LayerNorm(self.num_features, epsilon=LN_EPS,
                                  param_dtype=params.param_dtype, rngs=rngs)
```

`SwinEncoder.__call__` (line 403), cast the input at entry:
```python
    def __call__(self, x):
        x = jnp.asarray(x, dtype=self.params.param_dtype)
        x = self.patch_embed(x)
```

`SwinDecoder.__init__` (lines 425–429 and 439–443):
```python
            concat_back_dim.append(
                nnx.Linear(2 * dim, dim, kernel_init=TRUNC_NORMAL,
                           param_dtype=params.param_dtype, rngs=rngs)
                if i_layer > 0 else Identity())
            if i_layer == 0:
                layers_up.append(PatchExpand(res, dim=dim, dim_scale=2,
                                             param_dtype=params.param_dtype, rngs=rngs))
```
```python
        self.norm_up = nnx.LayerNorm(params.embed_dim, epsilon=LN_EPS,
                                     param_dtype=params.param_dtype, rngs=rngs)
        self.up = FinalPatchExpand(pr, patch_size=params.patch_size,
                                   dim=params.embed_dim,
                                   param_dtype=params.param_dtype, rngs=rngs)
        self.output = nnx.Conv(params.embed_dim, params.out_channels, kernel_size=(1, 1),
                               use_bias=False, param_dtype=params.param_dtype, rngs=rngs)
```

- [ ] **Step 4: Run the new test, then the full suite**

Run: `uv run pytest tests/test_model.py -q`
Expected: PASS.

Run: `uv run pytest tests/ -q`
Expected: ALL tests PASS (full regression sweep across geometry, shifting, RoPE, buffers, and both model families).

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/models/swin.py tests/test_model.py
git commit -m "feat: thread param_dtype through SwinUnet (input cast at encoder entry)"
```
