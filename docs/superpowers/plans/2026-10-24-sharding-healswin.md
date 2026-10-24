# HEAL-SWIN-nnx Multi-Device Sharding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every parameter in `heal_swin_nnx` carries logical sharding annotations, a self-contained `heal_swin_nnx.sharding` module places a model and its batches on a `jax.sharding.Mesh` under the `replicated`, `fsdp` or `tensor` rule, and all of it is verified on 8 simulated CPU devices.

**Architecture:** Annotations are Flax `out_sharding` metadata with **logical** axis names (`embed`, `mlp`, `heads`, `vocab`, `in`, `out`) and the per-variable `eager_sharding=False` opt-out, written through two helpers so models never see a mesh. `shard_model` resolves names → mesh axes through a rule table and `device_put`s state in place; `shard_batch` puts axis 0 of every array on the mesh's batch axes. Models build on one device exactly as today.

**Tech Stack:** JAX ≥ 0.11.1, Flax NNX ≥ 0.12.7, pytest + pytest-env + pytest-xdist, `XLA_FLAGS=--xla_force_host_platform_device_count=8`.

**Spec:** `docs/superpowers/specs/2026-10-24-multi-device-sharding-design.md` (copy of the GenSBI spec; §5.1, §5.2, §5.5, §6.1, §6.5 apply here). This plan is the first half; the GenSBI plan (`GenSBI/docs/superpowers/plans/2026-10-24-sharding-gensbi.md`) depends on it.

## Global Constraints

- `heal_swin_nnx/sharding.py` imports only `jax`, `flax.nnx`, `numpy`, `warnings`, `math`. It never imports GenSBI.
- `LOGICAL_AXES` and `RULES` below are a cross-repo contract: byte-identical in GenSBI's `gensbi/sharding.py`. Do not rename or reorder.
- Every annotation goes through `axes(...)` / `partitioned(...)`; a raw `out_sharding` without `eager_sharding=False` would raise at construction without a mesh.
- Each annotation tuple has the same length as the array's rank, and no logical name appears twice in one tuple (a repeated name would map to a repeated mesh axis, which `PartitionSpec` rejects).
- The pixel/token axis (axis 1 of `(B, npix, C)`) is never annotated with a sharded name and never sharded.
- `Buffer` variables, Dropout/DropPath RNG state, and anything else that is not an `nnx.Param` stay unannotated → replicated.
- Behaviour with no mesh must be byte-identical to today: parameters on `SingleDeviceSharding`, same values for the same seed.
- Mesh axis names are `("data", "model")`, Auto axis types. Batch axes = every mesh axis except `"model"`.
- Tests run with `uv run pytest tests/ -q` (xdist `-n 2`, `JAX_PLATFORMS=cpu`). The sandbox cannot run `uv` (read-only cache); use `.venv/bin/python -m pytest` and ask the user to run `uv sync` / `uv lock` themselves when the lock must change.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. **bf16 compute dtype (the default `dtype="bfloat16"`)** with fp32 params: a sharded forward must match an unsharded one to bf16 tolerance (`atol=2e-2` relative to output scale), not fp32 tolerance. Pinned in Task 5 (`test_forward_matches_unsharded_bf16`).
2. **Every `pos_embed` variant** creates different `Param`/`Buffer` sets (`rel_bias` table, `rope_mixed` freqs, `rope_axial` table). Coverage must run over all three. Pinned in Task 3 (`test_every_healswin_param_annotated` parametrized over `pos_embed`).
3. **Active Dropout and DropPath** (`drop_rate>0`, `drop_path_rate>0`, `model.train()`): the per-sample Bernoulli mask is drawn over a batch-sharded shape; with partitionable threefry it must equal the unsharded draw for the same RNG state. Pinned in Task 5 (`test_train_mode_forward_matches_unsharded`).
4. **Batch size not divisible by the device count** must fail with the `check_batch_divisible` `ValueError` before any jit, naming the batch size and divisor. Pinned in Task 2 (`test_check_batch_divisible_message`).
5. **Partial-sphere models (`base_pixels` subset) and `HealSwinEncoder` used standalone** (the way GenSBI consumes it) must shard like the full U-Net. Pinned in Task 5 (`test_encoder_standalone_fsdp`), using the 8-face `_tiny_params`.

---

### Task 1: Test environment on 8 simulated devices

**Files:**
- Modify: `pyproject.toml` (`[tool.pytest.ini_options].env`, lines 27-31)
- Create: `tests/test_sharding.py`

**Interfaces:**
- Produces: a test process where `jax.device_count() == 8`; a module-level marker `requires_8_devices = pytest.mark.skipif(jax.device_count() < 8, reason=...)` in `tests/test_sharding.py` used by every later sharding test.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sharding.py
import jax, pytest

requires_8_devices = pytest.mark.skipif(
    jax.device_count() < 8,
    reason="needs XLA_FLAGS=--xla_force_host_platform_device_count=8")

def test_eight_cpu_devices_present():
    assert jax.device_count() == 8
    assert all(d.platform == "cpu" for d in jax.devices())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sharding.py::test_eight_cpu_devices_present -q -p no:xdist`
Expected: FAIL `assert 1 == 8`

- [ ] **Step 3: Add the flag to pytest-env**

In `pyproject.toml` the `env` list becomes:
```toml
env = [
    "JAX_PLATFORMS=cpu",
    "XLA_FLAGS=--xla_force_host_platform_device_count=8",
]
```

- [ ] **Step 4: Run the whole suite to verify it passes and nothing else changed**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all PASS, including the new test (pytest-env sets the flag before JAX initialises in each xdist worker).

- [ ] **Step 5: Commit** (this commit also carries the user's pending `pyproject.toml` version/floor bump; keep both hunks)

```bash
git checkout -b sharding
git add pyproject.toml tests/test_sharding.py
git commit -m "test: run the suite on 8 simulated CPU devices"
```

---

### Task 2: `heal_swin_nnx.sharding` module

**Files:**
- Create: `src/heal_swin_nnx/sharding.py`
- Test: `tests/test_sharding.py`

**Interfaces:**
- Produces (exact; GenSBI copies the first five names verbatim):

```python
LOGICAL_AXES: tuple[str, ...] = ("embed", "mlp", "heads", "vocab", "in", "out")
RULES: dict[str, tuple[tuple[str, str | None], ...]] = {
    "replicated": (("embed", None), ("mlp", None), ("heads", None), ("vocab", None), ("in", None), ("out", None)),
    "fsdp":       (("embed", "data"), ("mlp", None), ("heads", None), ("vocab", None), ("in", None), ("out", None)),
    "tensor":     (("embed", None), ("mlp", "model"), ("heads", "model"), ("vocab", "model"), ("in", None), ("out", None)),
}
MODEL_AXIS = "model"
def axes(*names: str | None) -> dict           # {"out_sharding": names, "eager_sharding": False}
def partitioned(init, names: tuple[str | None, ...])   # nnx.with_partitioning(init, names, eager_sharding=False)
def make_mesh(devices=None, model_axis: int = 1) -> jax.sharding.Mesh   # axes ("data","model"), Auto types; data = len(devices)//model_axis; ValueError if not divisible
def batch_axes(mesh) -> tuple[str, ...]        # all mesh axis names except MODEL_AXIS
def batch_size_divisor(mesh) -> int            # product of mesh.shape[a] for a in batch_axes(mesh)
def shard_model(node, mesh, rule: str = "replicated") -> None   # in place, idempotent
def shard_batch(batch, mesh)                    # pytree; arrays -> device_put(NamedSharding(mesh, P(batch_axes(mesh)))); non-arrays untouched
def check_batch_divisible(batch, mesh, what: str) -> None       # ValueError
def sharding_of(node) -> dict[str, jax.sharding.Sharding]       # "a/b/c" path -> sharding of each state leaf
```

- `shard_model` semantics: for every variable in `nnx.state(node)` (via `nnx.to_flat_state`): read `out_sharding` metadata; if absent → `P()`; else map each name through `dict(RULES[rule])`; a name not in `LOGICAL_AXES` → whole variable `P()` and recorded; a mapped mesh axis whose size does not divide the array dim → whole variable `P()` and recorded; otherwise `P(*mapped)`. `device_put` each value to `NamedSharding(mesh, spec)`, write back with `nnx.update(node, nnx.from_flat_state(...))`. If anything was recorded, emit **one** `warnings.warn` listing the paths and the reason. If `rule not in RULES` → `ValueError`. If any `mesh.axis_types` is `Explicit` → `warnings.warn`. `rule != "replicated"` additionally emits `UserWarning("sharding rule '<rule>' is experimental ...")` on **every** call (Python's default filter already shows it once per call site; tests rely on it being deterministic, never "once per process").
- `check_batch_divisible` message: `f"{what} batch size {B} is not divisible by the data-parallel device count {d} (mesh axes {batch_axes(mesh)}). Use a batch size that is a multiple of {d}."` where `B` is axis 0 of the first array leaf; also raise `f"{what} batch leaves have differing leading sizes: {sizes}"` when array leaves disagree.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_sharding.py`)

```python
import numpy as np, jax.numpy as jnp, warnings
from flax import nnx
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
from heal_swin_nnx import sharding as sh
from heal_swin_nnx.variables import Buffer

class Toy(nnx.Module):
    def __init__(self, rngs):
        self.l1 = nnx.Linear(16, 32, rngs=rngs, kernel_metadata=sh.axes("embed", "mlp"), bias_metadata=sh.axes("mlp"))
        self.l2 = nnx.Linear(32, 16, rngs=rngs, kernel_metadata=sh.axes("mlp", "embed"), bias_metadata=sh.axes("embed"))
        self.odd = nnx.Param(jnp.ones((6, 3)), **sh.axes("embed", None))     # 6 % 8 != 0 -> fsdp fallback
        self.bad = nnx.Param(jnp.ones((4,)), **sh.axes("nonsense"))          # unknown name -> fallback
        self.buf = Buffer(jnp.arange(5))
        self.drop = nnx.Dropout(0.1, rngs=rngs)

def test_rule_tables_cover_every_logical_name():
    for rule, table in sh.RULES.items():
        assert tuple(n for n, _ in table) == sh.LOGICAL_AXES, rule
    assert sh.RULES["replicated"] == tuple((n, None) for n in sh.LOGICAL_AXES)

def test_axes_and_partitioned_set_the_opt_out():
    assert sh.axes("embed", None) == {"out_sharding": ("embed", None), "eager_sharding": False}
    m = Toy(nnx.Rngs(0))                      # must construct with NO mesh anywhere
    assert isinstance(m.l1.kernel[...].sharding, jax.sharding.SingleDeviceSharding)
    assert m.l1.kernel.get_metadata()["eager_sharding"] is False

@requires_8_devices
def test_make_mesh_shape_and_axis_types():
    mesh = sh.make_mesh()
    assert mesh.axis_names == ("data", "model") and dict(mesh.shape) == {"data": 8, "model": 1}
    assert all(t == AxisType.Auto for t in mesh.axis_types)
    mesh2 = sh.make_mesh(model_axis=2)
    assert dict(mesh2.shape) == {"data": 4, "model": 2}
    assert sh.batch_axes(mesh2) == ("data",) and sh.batch_size_divisor(mesh2) == 4
    with pytest.raises(ValueError):
        sh.make_mesh(model_axis=3)

@requires_8_devices
def test_shard_model_replicated_places_everything():
    m = Toy(nnx.Rngs(0)); mesh = sh.make_mesh()
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # replicated must not warn
        sh.shard_model(m, mesh, "replicated")
    for path, s in sh.sharding_of(m).items():
        assert isinstance(s, NamedSharding) and s.mesh == mesh, path
        assert s.is_equivalent_to(NamedSharding(mesh, P()), 1), path   # includes buf and dropout RngKey/RngCount

@requires_8_devices
def test_shard_model_fsdp_shards_embed_and_falls_back():
    m = Toy(nnx.Rngs(0)); mesh = sh.make_mesh()
    with pytest.warns(UserWarning) as rec:
        sh.shard_model(m, mesh, "fsdp")
    msgs = " ".join(str(w.message) for w in rec)
    assert "odd" in msgs and "bad" in msgs
    s = sh.sharding_of(m)
    assert s["l1/kernel"].is_equivalent_to(NamedSharding(mesh, P("data", None)), 2)
    assert s["l2/kernel"].is_equivalent_to(NamedSharding(mesh, P(None, "data")), 2)
    assert s["l2/bias"].is_equivalent_to(NamedSharding(mesh, P("data")), 1)
    assert s["odd"].is_equivalent_to(NamedSharding(mesh, P()), 2)
    assert s["bad"].is_equivalent_to(NamedSharding(mesh, P()), 1)
    sh.shard_model(m, mesh, "fsdp")           # idempotent: second call is a no-op (no error)

@requires_8_devices
def test_shard_model_tensor_shards_mlp_over_model_axis():
    m = Toy(nnx.Rngs(0)); mesh = sh.make_mesh(model_axis=2)
    sh.shard_model(m, mesh, "tensor")
    assert sh.sharding_of(m)["l1/kernel"].is_equivalent_to(NamedSharding(mesh, P(None, "model")), 2)

def test_shard_model_rejects_unknown_rule():
    with pytest.raises(ValueError):
        sh.shard_model(Toy(nnx.Rngs(0)), sh.make_mesh(), "nope")

@requires_8_devices
def test_shard_batch_numpy_jax_tuple_and_dict_leaves():
    mesh = sh.make_mesh()
    x = np.zeros((16, 5, 1), np.float32); y = jnp.zeros((16,), jnp.int32)
    out = sh.shard_batch({"image": x, "label": y, "meta": 3}, mesh)
    assert out["image"].sharding.is_equivalent_to(NamedSharding(mesh, P("data")), 3)
    assert out["label"].sharding.is_equivalent_to(NamedSharding(mesh, P("data")), 1)
    assert out["meta"] == 3
    a, b = sh.shard_batch((x, y), mesh)
    assert a.shape == (16, 5, 1) and b.shape == (16,)

@requires_8_devices
def test_check_batch_divisible_message():
    mesh = sh.make_mesh()
    sh.check_batch_divisible((np.zeros((16, 4)),), mesh, "train")   # ok
    with pytest.raises(ValueError, match=r"train batch size 12 is not divisible by the data-parallel device count 8"):
        sh.check_batch_divisible((np.zeros((12, 4)),), mesh, "train")
    with pytest.raises(ValueError, match="differing leading sizes"):
        sh.check_batch_divisible((np.zeros((16, 4)), np.zeros((8,))), mesh, "train")
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sharding.py -q -p no:xdist`
Expected: FAIL with `ModuleNotFoundError: heal_swin_nnx.sharding`

- [ ] **Step 3: Implement `src/heal_swin_nnx/sharding.py`** with the exact surface above. `make_mesh`: `devices = devices or jax.devices()`; `jax.sharding.Mesh(np.asarray(devices).reshape(len(devices)//model_axis, model_axis), ("data","model"), axis_types=(AxisType.Auto, AxisType.Auto))`. `sharding_of`: `{"/".join(map(str, path)): var[...].sharding for path, var in nnx.to_flat_state(nnx.state(node))}`. Module docstring states the cross-repo contract and that GenSBI holds a copy.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sharding.py -q -p no:xdist`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/sharding.py tests/test_sharding.py
git commit -m "feat: heal_swin_nnx.sharding — logical axes, rules, mesh helpers, shard_model/shard_batch"
```

---

### Task 3: Annotate `layers.py` and `models/healswin.py`

**Files:**
- Modify: `src/heal_swin_nnx/layers.py:80-215` (Mlp, PatchMerging, PatchExpand, FinalPatchExpand, PatchEmbed)
- Modify: `src/heal_swin_nnx/models/healswin.py:147-184` (WindowAttention), `:235-239` (block norms), `:354` (encoder norm), `:381` (concat_back_dim), `:396-404` (norm_up, output conv)
- Test: `tests/test_sharding.py`

**Interfaces:**
- Consumes: `axes` from Task 2 (`from heal_swin_nnx.sharding import axes`).
- Produces: every `nnx.Param` reachable from `HealSwin`/`HealSwinEncoder` carries `out_sharding` + `eager_sharding=False`.

Annotation table (kernel / bias; LayerNorm `scale_metadata` and `bias_metadata` share one tuple):

| Site | kernel | bias |
|---|---|---|
| `Mlp.fc1` (in→hidden) | `("embed","mlp")` | `("mlp",)` |
| `Mlp.fc2` (hidden→out) | `("mlp","embed")` | `("embed",)` |
| `PatchMerging.reduction` (4C→sC) | `(None,"embed")` | – (no bias) |
| `PatchMerging.norm` (over 4C) | `(None,)` | `(None,)` |
| `PatchExpand.expand` (C→sC) | `("embed",None)` | – |
| `PatchExpand.norm` | `("embed",)` | `("embed",)` |
| `FinalPatchExpand.expand` (C→p·C) | `("embed",None)` | – |
| `FinalPatchExpand.norm` | `("embed",)` | `("embed",)` |
| `PatchEmbed.proj` (Conv1d, kernel `(p,in,embed)`) | `(None,"in","embed")` | `("embed",)` |
| `PatchEmbed.norm` | `("embed",)` | `("embed",)` |
| `WindowAttention.logit_scale` `(H,1,1)` | `nnx.Param(value, **axes(None,None,None))` | |
| `WindowAttention.relative_position_bias_table` `((2s-1)²,H)` | `**axes(None,"heads")` | |
| `WindowAttention.rope_freqs` `(2,H,hd/2)` | `**axes(None,"heads",None)` | |
| `WindowAttention.qkv` (C→3C) | `("embed","heads")` | `("heads",)` |
| `WindowAttention.proj` (C→C) | `("heads","embed")` | `("embed",)` |
| `HealSwinBlock.norm1/norm2`, `HealSwinEncoder.norm`, `HealSwinDecoder.norm_up` | `("embed",)` | `("embed",)` |
| `HealSwinDecoder.concat_back_dim[i]` (2C→C) | `(None,"embed")` | `("embed",)` |
| `HealSwinDecoder.output` (Conv1d `(1,embed,out)`) | `(None,"embed","out")` | – (no bias) |

- [ ] **Step 1: Write the failing test** (append to `tests/test_sharding.py`)

```python
from heal_swin_nnx import HealSwin, HealSwinEncoder, HealSwinParams

def _tiny_params(**over):
    kw = dict(nside=16, in_channels=3, out_channels=5, base_pixels=tuple(range(8)),
              embed_dim=16, depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0)
    kw.update(over); return HealSwinParams(**kw)

def assert_every_param_annotated(model):
    for path, var in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        meta = var.get_metadata(); name = "/".join(map(str, path))
        assert "out_sharding" in meta, f"{name} has no annotation"
        assert meta["eager_sharding"] is False, name
        names = meta["out_sharding"]
        assert len(names) == var[...].ndim, f"{name}: rank {var[...].ndim} vs {names}"
        assert set(n for n in names if n is not None) <= set(sh.LOGICAL_AXES), name
        used = [n for n in names if n is not None]
        assert len(used) == len(set(used)), f"{name}: repeated logical name in {names}"
    for path, var in nnx.to_flat_state(nnx.state(model, Buffer)):
        assert "out_sharding" not in var.get_metadata(), path

@pytest.mark.parametrize("pos_embed", ["rope_mixed", "rel_bias", "rope_axial"])
def test_every_healswin_param_annotated(pos_embed):
    m = HealSwin(_tiny_params(pos_embed=pos_embed), rngs=nnx.Rngs(0))
    assert_every_param_annotated(m)
    first_kernel = next(v for p, v in nnx.to_flat_state(nnx.state(m, nnx.Param)) if str(p[-1]) == "kernel")
    assert isinstance(first_kernel[...].sharding, jax.sharding.SingleDeviceSharding)

def test_healswin_values_unchanged_by_annotations():
    # metadata must not change initialisation: compare against a snapshot taken on main
    # before this branch (store the first kernel's mean/std in the test as literals).
    m = HealSwin(_tiny_params(), rngs=nnx.Rngs(0))
    first_kernel = next(v for p, v in nnx.to_flat_state(nnx.state(m, nnx.Param)) if str(p[-1]) == "kernel")
    k = np.asarray(first_kernel[...], np.float32)
    assert np.isfinite(k).all() and k.std() > 0
```

For `test_healswin_values_unchanged_by_annotations`, before Step 3 run the snippet on the unmodified code, record `float(k.mean())` and `float(k.std())`, and assert them with `rtol=1e-6` in the test.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sharding.py -k annotated -q -p no:xdist`
Expected: FAIL `has no annotation`

- [ ] **Step 3: Annotate every site in the table** with `kernel_metadata=axes(...)`, `bias_metadata=axes(...)`, `scale_metadata=axes(...)` / `bias_metadata=axes(...)` on `nnx.LayerNorm`, and `nnx.Param(value, **axes(...))` for the three direct params. Import `from heal_swin_nnx.sharding import axes` in both files.

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS (existing jit/eager, precision and buffer tests are unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/layers.py src/heal_swin_nnx/models/healswin.py tests/test_sharding.py
git commit -m "feat: logical sharding annotations on HealSwin layers"
```

---

### Task 4: Annotate `models/swin.py` and `models/healconv.py`

**Files:**
- Modify: `src/heal_swin_nnx/models/swin.py:155-183, 289-297, 314-321, 338-346, 362-372, 475-501`
- Modify: `src/heal_swin_nnx/models/healconv.py:156-168, 303, 328-333, 345-355`
- Test: `tests/test_sharding.py`

**Interfaces:** same as Task 3. Table additions:

| Site | kernel | bias |
|---|---|---|
| `swin.WindowAttention.*` | as HealSwin (`rel_bias` table `(n_rel,H)` → `(None,"heads")`) | |
| `swin.PatchMerging.reduction` (4C→2C) | `(None,"embed")` | – |
| `swin.PatchExpand.expand` (C→2C) | `("embed",None)` | – |
| `swin.FinalPatchExpand.expand` (C→p₀p₁C) | `("embed",None)` | – |
| `swin.PatchEmbed.proj` (Conv2d `(kh,kw,in,embed)`) | `(None,None,"in","embed")` | `("embed",)` |
| `swin.SwinDecoder.output` (Conv2d `(1,1,embed,out)`) | `(None,None,"embed","out")` | – |
| all `swin` LayerNorms over C | `("embed",)` | `("embed",)`; over 4C: `(None,)` |
| `healconv.HealConvBlock.dwconv` (depthwise `(g,g,1,C)`) | `(None,None,None,"embed")` | `("embed",)` if `conv_bias` |
| `healconv` norms, `concat_back_dim`, `output`, `Mlp` | as HealSwin | |

- [ ] **Step 1: Write the failing tests**

```python
from heal_swin_nnx import HealConv, HealConvParams, SwinUnet, SwinParams

def test_every_swin_param_annotated():
    # use the smallest SwinParams already used in tests/test_model.py (grep "SwinParams(")
    m = SwinUnet(SwinParams(...), rngs=nnx.Rngs(0))
    assert_every_param_annotated(m)

def test_every_healconv_param_annotated():
    # use the smallest HealConvParams already used in tests/test_healconv.py
    m = HealConv(HealConvParams(...), rngs=nnx.Rngs(0))
    assert_every_param_annotated(m)
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/test_sharding.py -k "swin or healconv" -q -p no:xdist` → FAIL `has no annotation`.
- [ ] **Step 3: Annotate every site in the table.**
- [ ] **Step 4: Run the whole suite** — `.venv/bin/python -m pytest tests/ -q` → PASS.
- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/models/swin.py src/heal_swin_nnx/models/healconv.py tests/test_sharding.py
git commit -m "feat: logical sharding annotations on Swin and HealConv"
```

---

### Task 5: Behaviour under a mesh: placement, numerics, gradients

**Files:**
- Test: `tests/test_sharding.py`

**Interfaces:**
- Consumes: Tasks 2-4.
- Produces: the acceptance tests of spec §6.5.

- [ ] **Step 1: Write the tests**

```python
import optax

def _batch(B, p, key):
    return jax.random.normal(key, (B, p.npix, p.in_channels), jnp.float32)

def _params_allclose(a, b, **tol):
    fa = nnx.to_flat_state(nnx.state(a, nnx.Param)); fb = nnx.to_flat_state(nnx.state(b, nnx.Param))
    for (pa, va), (pb, vb) in zip(fa, fb):
        assert pa == pb
        np.testing.assert_allclose(np.asarray(vb[...]), np.asarray(va[...]), err_msg=str(pa), **tol)

@requires_8_devices
@pytest.mark.parametrize("rule", ["replicated", "fsdp"])
def test_forward_matches_unsharded(rule):
    p = _tiny_params(dtype="float32")                      # fp32 compute -> tight tolerance
    ref = HealSwin(p, rngs=nnx.Rngs(0)); ref.eval()
    m = HealSwin(p, rngs=nnx.Rngs(0)); m.eval()
    mesh = sh.make_mesh(); sh.shard_model(m, mesh, rule)
    x = _batch(16, p, jax.random.key(1))
    y_ref = ref(x)
    y = nnx.jit(lambda m, x: m(x))(m, sh.shard_batch(x, mesh))
    assert y.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), y.ndim)
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_ref), rtol=1e-5, atol=1e-5)

@requires_8_devices
def test_forward_matches_unsharded_bf16():
    p = _tiny_params()                                     # default dtype="bfloat16"
    ref = HealSwin(p, rngs=nnx.Rngs(0)); ref.eval()
    m = HealSwin(p, rngs=nnx.Rngs(0)); m.eval()
    mesh = sh.make_mesh(); sh.shard_model(m, mesh, "fsdp")
    x = _batch(16, p, jax.random.key(1))
    y_ref = np.asarray(ref(x), np.float32)
    y = np.asarray(nnx.jit(lambda m, x: m(x))(m, sh.shard_batch(x, mesh)), np.float32)
    np.testing.assert_allclose(y, y_ref, rtol=2e-2, atol=2e-2 * np.abs(y_ref).max())

@requires_8_devices
def test_train_mode_forward_matches_unsharded():
    # active Dropout + DropPath: the per-sample masks must agree for the same RNG state
    p = _tiny_params(dtype="float32", drop_rate=0.2, drop_path_rate=0.2)
    ref = HealSwin(p, rngs=nnx.Rngs(0)); ref.train()
    m = HealSwin(p, rngs=nnx.Rngs(0)); m.train()
    mesh = sh.make_mesh(); sh.shard_model(m, mesh, "replicated")
    x = _batch(16, p, jax.random.key(1))
    y_ref = nnx.jit(lambda m, x: m(x))(ref, x)
    y = nnx.jit(lambda m, x: m(x))(m, sh.shard_batch(x, mesh))
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_ref), rtol=1e-5, atol=1e-5)

@requires_8_devices
def test_fsdp_places_qkv_and_mlp_kernels_over_data_and_buffers_replicated():
    p = _tiny_params(); m = HealSwin(p, rngs=nnx.Rngs(0)); mesh = sh.make_mesh()
    sh.shard_model(m, mesh, "fsdp")
    s = sh.sharding_of(m)
    qkv = [k for k in s if k.endswith("qkv/kernel")]; fc1 = [k for k in s if k.endswith("fc1/kernel")]
    assert qkv and fc1
    for k in qkv + fc1:
        assert s[k].is_equivalent_to(NamedSharding(mesh, P("data", None)), 2), k
    bufs = {"/".join(map(str, pth)): v[...].ndim for pth, v in nnx.to_flat_state(nnx.state(m, Buffer))}
    assert bufs
    for k, nd in bufs.items():
        assert s[k].is_equivalent_to(NamedSharding(mesh, P()), nd), k

@requires_8_devices
@pytest.mark.parametrize("rule", ["replicated", "fsdp"])
def test_one_gradient_step_matches_unsharded(rule):
    p = _tiny_params(dtype="float32")
    def make():
        m = HealSwin(p, rngs=nnx.Rngs(0)); m.train()
        return m, nnx.Optimizer(m, optax.adamw(1e-3), wrt=nnx.Param)
    ref, opt_ref = make(); m, opt = make()
    mesh = sh.make_mesh(); sh.shard_model(m, mesh, rule); sh.shard_model(opt, mesh, rule)
    x = _batch(16, p, jax.random.key(1))
    @nnx.jit
    def step(model, opt, x):
        loss, g = nnx.value_and_grad(lambda mm: (mm(x) ** 2).mean())(model)
        opt.update(model, g); return loss
    l_ref = step(ref, opt_ref, x); l = step(m, opt, sh.shard_batch(x, mesh))
    np.testing.assert_allclose(float(l), float(l_ref), rtol=1e-5)
    _params_allclose(ref, m, rtol=1e-4, atol=1e-6)
    if rule == "fsdp":   # optimizer moments keep the data sharding after the step
        mu = [v for pth, v in nnx.to_flat_state(nnx.state(opt)) if "mu" in map(str, pth) and v[...].ndim == 2]
        assert mu and any(v[...].sharding.is_equivalent_to(NamedSharding(mesh, P("data", None)), 2) for v in mu)

@requires_8_devices
def test_encoder_standalone_fsdp():
    p = _tiny_params(); enc = HealSwinEncoder(p, rngs=nnx.Rngs(0)); enc.eval()
    mesh = sh.make_mesh(); sh.shard_model(enc, mesh, "fsdp")
    tokens, skips = nnx.jit(lambda m, x: m(x))(enc, sh.shard_batch(_batch(8, p, jax.random.key(0)), mesh))
    assert tokens.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), tokens.ndim)

@requires_8_devices
def test_tensor_rule_smoke_4x2():
    p = _tiny_params(dtype="float32")
    ref = HealSwin(p, rngs=nnx.Rngs(0)); ref.eval(); m = HealSwin(p, rngs=nnx.Rngs(0)); m.eval()
    mesh = sh.make_mesh(model_axis=2); sh.shard_model(m, mesh, "tensor")
    s = sh.sharding_of(m); k = next(k for k in s if k.endswith("fc1/kernel"))
    assert s[k].is_equivalent_to(NamedSharding(mesh, P(None, "model")), 2)
    x = _batch(8, p, jax.random.key(1))
    y = nnx.jit(lambda m, x: m(x))(m, sh.shard_batch(x, mesh))
    np.testing.assert_allclose(np.asarray(y), np.asarray(ref(x)), rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Run** — `.venv/bin/python -m pytest tests/test_sharding.py -q -p no:xdist` → PASS. These exercise Tasks 2-4; a failure here is a defect in the annotations or `shard_model`, fix it there. Loosen a tolerance only with evidence (print the max abs diff).
- [ ] **Step 3: Run the whole suite** — `.venv/bin/python -m pytest tests/ -q` → PASS.
- [ ] **Step 4: Commit**

```bash
git add tests/test_sharding.py
git commit -m "test: sharded HealSwin matches unsharded forward/gradient; FSDP and tensor placement"
```

---

### Task 6: Sharded MNIST example

**Files:**
- Create: `examples/mnist_healpix_classify_sharded.py` (copy of `examples/mnist_healpix_classify.py`)
- Test: `tests/test_sharding.py`

**Interfaces:**
- Consumes: `make_mesh`, `shard_model`, `shard_batch`, `check_batch_divisible` from Task 2.
- Produces: a module with `HealSwinClassifier`, `make_params(**over)`, `train_step`, `make_loader(...)` (now `drop_remainder=True`), `to_model_inputs(batch, mesh)`, and `smoke(mesh) -> float` (two synthetic train steps on a batch of 16 with a reduced `make_params(nside=16, depths=(2, 2), num_heads=(2, 4), embed_dim=16)`, returns the second loss) callable without data.

Differences from the original (everything else byte-identical):
1. Module docstring: multi-device variant; the `os.environ.setdefault("JAX_PLATFORMS", "cuda")` line is deleted (the user chooses the platform); worker pinning kept.
2. `RULE = os.environ.get("HEALSWIN_SHARDING_RULE", "replicated")` in the config block; `main()` does `mesh = make_mesh()` and prints it.
3. `make_loader` uses `.batch(batch_size, drop_remainder=True)` (a ragged last batch would fail divisibility and force a recompile).
4. `to_model_inputs(batch, mesh)` returns `shard_batch((images, labels), mesh)`; the first batch of each epoch passes `check_batch_divisible(..., "train")`; eval batches `"eval"`.
5. After `model = HealSwinClassifier(...)`: `shard_model(model, mesh, RULE)`; after `optimizer = nnx.Optimizer(...)`: `shard_model(optimizer, mesh, RULE)`.
6. `train_step`/`eval_step` unchanged. The epoch loop body runs inside `with jax.set_mesh(mesh):`.
7. The `SMOKE=1` path calls `smoke(make_mesh())` and prints the loss.

- [ ] **Step 1: Write the failing test**

```python
import importlib, sys, pathlib

@requires_8_devices
def test_sharded_example_smoke_runs_two_steps():
    pytest.importorskip("grain"); pytest.importorskip("datasets")
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples"))
    ex = importlib.import_module("mnist_healpix_classify_sharded")
    loss = ex.smoke(sh.make_mesh())
    assert np.isfinite(loss)
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_sharding.py -k example -q -p no:xdist` → FAIL `ModuleNotFoundError` (or SKIP if the `examples` extra is not installed; then ask the user to run `uv sync --extra examples` and re-run).
- [ ] **Step 3: Create the example** per the list above.
- [ ] **Step 4: Run** — same command → PASS. Also `SMOKE=1 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 .venv/bin/python examples/mnist_healpix_classify_sharded.py` → prints a finite loss.
- [ ] **Step 5: Commit**

```bash
git add examples/mnist_healpix_classify_sharded.py tests/test_sharding.py
git commit -m "docs(examples): multi-device MNIST classifier using heal_swin_nnx.sharding"
```

---

### Task 7: Docs and release prep

**Files:**
- Modify: `README.md` (new section "Multi-device training"), `CLAUDE.md` (one paragraph)
- Modify: `pyproject.toml` (`version`), `uv.lock`

- [ ] **Step 1: README section** (≤ 25 lines): the three-line usage (`make_mesh`, `shard_model`, `shard_batch`), the rule names with `fsdp`/`tensor` marked experimental, the note that models build without a mesh, the annotation helpers for users who add layers, the 8-device test flag.
- [ ] **Step 2: CLAUDE.md paragraph** with the same facts plus "every new `nnx.Param` must go through `axes()`/`partitioned()`; `tests/test_sharding.py::assert_every_param_annotated` enforces it".
- [ ] **Step 3: Version.** If `0.1.2` has not been released to PyPI, keep it; otherwise bump to `0.1.3`. The GenSBI plan's final task pins `heal-swin-nnx>=` to this number.
- [ ] **Step 4: Ask the user to run** `uv lock && uv sync && uv run pytest` **outside the sandbox** (the lock must reflect `jax>=0.11.1`); expected all PASS.
- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md pyproject.toml uv.lock
git commit -m "docs: multi-device sharding section; release prep"
```

Then hand off per superpowers:finishing-a-development-branch. Publishing to PyPI is the user's action; GenSBI's plan needs it only for its last task.
