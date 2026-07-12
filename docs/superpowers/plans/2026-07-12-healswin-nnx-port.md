# HEAL-SWIN → Flax NNX Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `SwinHPTransformerSys` (HEAL-SWIN) and `SwinTransformerSys` (flat SWIN-UNet) from `references/HEAL-SWIN` to pure JAX/Flax NNX with golden-value numerical parity (forward + gradients) against the original PyTorch implementation.

**Architecture:** Mirrored module tree (torch state_dict keys map mechanically to nnx paths) with an explicit encoder/decoder seam (`SwinHPEncoder` + `HPUnetDecoder`, `SwinEncoder` + `UnetDecoder`). All sphere-index machinery is computed in numpy/healpy at construction time and stored in a custom `Buffer(nnx.Variable)` type; runtime is pure JAX. Goldens are generated ONCE from the reference code in a pinned legacy torch env (`parity/`) and committed; the main test suite never imports torch.

**Tech Stack:** JAX, Flax NNX, numpy, healpy, einops (main env); Python 3.8 + torch 1.8.0+cpu + timm 0.4.12 + healpy 1.15.2 (parity env, via uv).

**Spec:** `docs/superpowers/specs/2026-07-12-healswin-nnx-port-design.md` — read it first.

## Global Constraints

- Reference code at `references/HEAL-SWIN` is READ-ONLY. Never modify it.
- Parity env pins (exact): `requires-python = ">=3.8,<3.9"`, `torch==1.8.0+cpu`, `torchvision==0.9.0+cpu`, `timm==0.4.12`, `einops==0.4.0`, `healpy==1.15.2`, `numpy==1.19.2`, `scipy==1.6.0`, `matplotlib==3.3.4`, `astropy==5.1`. Torch/torchvision come from the uv index `https://download.pytorch.org/whl/cpu`.
- `uv sync` in `parity/` may fail inside the sandbox (network). The user has pre-approved disabling the sandbox FOR THAT COMMAND ONLY.
- Every `nnx.LayerNorm` uses `epsilon=1e-5` (torch default; flax default 1e-6 is WRONG here).
- GELU is exact-erf: `jax.nn.gelu(x, approximate=False)`.
- Public API is channels-last: HP `(B, N, C)`, flat `(B, H, W, C)`. Torch goldens are transposed at test boundaries only.
- Non-trainable constants (index arrays, attention masks, relative_position_index) live in `Buffer(nnx.Variable)`, never `nnx.Param`.
- Integer index arrays and 0/−100 masks are compared BIT-EXACTLY against goldens (`np.array_equal`), floats with `np.testing.assert_allclose` (leaf/module: `rtol=1e-5, atol=1e-6`; end-to-end forward: `rtol=1e-4, atol=1e-4`; end-to-end grads: `rtol=1e-3, atol=1e-4` — loosen only with a comment explaining why).
- 8-base-pixel topology constants must be module-level data keyed by `base_pix` (dicts), with `NotImplementedError` raised for unsupported `base_pix` (full-sphere extension lands later; see spec).
- Parity tests run models in eval mode (`model.eval()`); goldens are generated with torch `model.eval()` and `drop_path_rate=0.0`.
- Bug-for-bug fidelity: port reference bugs verbatim with a `# bug-for-bug with reference (<file>:<line>)` comment (known case: flat block roll uses `shift_size[0]` twice).
- Commit after every task (at minimum). Run `uv run pytest tests/ -x -q` before every commit from Task 6 onward.

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` (modify) | main env: add einops, pytest, build system |
| `src/heal_swin_nnx/__init__.py` | public exports |
| `src/heal_swin_nnx/variables.py` | `Buffer` variable type |
| `src/heal_swin_nnx/config.py` | `DataSpec`, `SwinHPTransformerConfig`, `SwinTransformerConfig` |
| `src/heal_swin_nnx/layers.py` | `Identity`, `DropPath`, `Mlp`, `TRUNC_NORMAL` init |
| `src/heal_swin_nnx/hp_windowing.py` | 1D window partition/reverse (jnp), `get_nest_win_idcs`, `nest_relative_position_index` (numpy) |
| `src/heal_swin_nnx/hp_shifting.py` | numpy index functions + `NoShift`/`NestRollShift`/`NestGridShift`/`RingShift` nnx wrappers + topology tables |
| `src/heal_swin_nnx/swin_hp_transformer.py` | HP `WindowAttention`, `SwinTransformerBlock`, `PatchMerging`, `PatchExpand`, `FinalPatchExpand_X4`, `BasicLayer`, `BasicLayer_up`, `PatchEmbed`, `SwinHPEncoder`, `HPUnetDecoder`, `SwinHPTransformerSys` |
| `src/heal_swin_nnx/swin_transformer.py` | flat 2D counterparts + `SwinEncoder`, `UnetDecoder`, `SwinTransformerSys` |
| `src/heal_swin_nnx/weight_transfer.py` | torch-npz → nnx state loading |
| `parity/pyproject.toml`, `parity/.python-version` | legacy env |
| `parity/generate_goldens.py` | fixture generator (torch side) |
| `parity/README.md` | how to regenerate goldens |
| `tests/goldens/*.npz` | committed fixtures |
| `tests/parity_utils.py` | golden loading + comparison helpers |
| `tests/test_buffers.py`, `tests/test_windowing.py`, `tests/test_shifting.py`, `tests/test_parity_modules.py`, `tests/test_parity_e2e.py`, `tests/test_model.py` | test suites |

Reference sources (read-only): `references/HEAL-SWIN/heal_swin/models_torch/{swin_hp_transformer.py,hp_windowing.py,hp_shifting.py,swin_transformer.py}`.

---

### Task 1: Package scaffold, `Buffer` variable, configs

**Files:**
- Modify: `pyproject.toml`
- Delete: `main.py` (uv init leftover)
- Create: `src/heal_swin_nnx/__init__.py`, `src/heal_swin_nnx/variables.py`, `src/heal_swin_nnx/config.py`
- Test: `tests/test_buffers.py`, `tests/__init__.py` (empty)

**Interfaces:**
- Produces: `Buffer(nnx.Variable)`; `DataSpec(dim_in, f_in, f_out, base_pix, class_names)`; `SwinHPTransformerConfig` (fields below); `SwinTransformerConfig` (fields below, normalizes int→tuple in `__post_init__`). All later tasks import these from `heal_swin_nnx.variables` / `heal_swin_nnx.config`.

- [ ] **Step 1: Update pyproject and install deps**

Add to `pyproject.toml` (keep existing `[project]` fields and dependencies):

```toml
[build-system]
requires = ["uv_build>=0.9,<1"]
build-backend = "uv_build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Run: `uv add einops && uv add --dev pytest && rm main.py && touch tests/__init__.py && uv sync`
Expected: resolves; `uv run python -c "import flax.nnx, einops"` prints nothing.

- [ ] **Step 2: Write the failing test**

`tests/test_buffers.py`:

```python
import jax.numpy as jnp
from flax import nnx

from heal_swin_nnx.variables import Buffer
from heal_swin_nnx.config import DataSpec, SwinHPTransformerConfig, SwinTransformerConfig


class Toy(nnx.Module):
    def __init__(self):
        self.w = nnx.Param(jnp.ones((3,)))
        self.idx = Buffer(jnp.arange(3))

    def __call__(self):
        return (self.w * self.idx).sum()


def test_buffer_excluded_from_params():
    m = Toy()
    params = nnx.state(m, nnx.Param)
    flat = dict(params.flat_state())
    assert ("w",) in flat
    assert not any("idx" in path for path in flat)


def test_buffer_not_differentiated():
    m = Toy()
    grads = nnx.grad(lambda m: m())(m)  # default wrt=nnx.Param
    flat = dict(grads.flat_state())
    assert ("w",) in flat
    assert not any("idx" in path for path in flat)


def test_configs_construct_with_reference_defaults():
    hp = SwinHPTransformerConfig()
    assert hp.patch_size == 4 and hp.window_size == 4 and hp.shift_size == 2
    assert hp.shift_strategy == "nest_roll" and hp.rel_pos_bias is None
    assert hp.depths == [2, 2, 2, 2] and hp.num_heads == [3, 6, 12, 24]
    flat = SwinTransformerConfig()
    assert flat.patch_size == (4, 4) and flat.window_size == (4, 4)
    assert flat.shift_size == (2, 2)  # -1 sentinel resolved to window//2
    flat2 = SwinTransformerConfig(window_size=8, shift_size=3)
    assert flat2.window_size == (8, 8) and flat2.shift_size == (3, 3)
    ds = DataSpec(dim_in=2048, f_in=3, f_out=5, base_pix=8, class_names=["a"] * 5)
    assert ds.dim_in == 2048
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_buffers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'heal_swin_nnx'` (or missing attribute).

- [ ] **Step 4: Implement**

`src/heal_swin_nnx/variables.py`:

```python
from flax import nnx


class Buffer(nnx.Variable):
    """Non-trainable constant state: index permutations, attention masks,
    relative position indices. Excluded from ``nnx.Param`` filters, so
    optimizers and ``nnx.grad`` never touch it, but it travels through
    ``nnx.split``/``nnx.merge``/``nnx.jit`` as regular pytree state."""
```

`src/heal_swin_nnx/config.py`:

```python
"""Serializable mirrors of the reference model configs.

Differences from the reference (all agreed in the spec):
- ``norm_layer``/``patch_embed_norm_layer`` are string literals, not classes.
- ``decoder_class`` removed (dead extension hook, only ``UnetDecoder`` existed).
- ``patch_norm`` and ``dev_mode`` removed (unused / debug scaffolding).
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union


@dataclass
class DataSpec:
    dim_in: Union[int, Tuple[int, int]]  # int (=npix) for HP, (H, W) for flat
    f_in: int
    f_out: int
    base_pix: Optional[int] = None
    class_names: List[str] = field(default_factory=list)


@dataclass
class SwinHPTransformerConfig:
    patch_size: int = 4
    window_size: int = 4
    shift_size: int = 2
    shift_strategy: Literal["nest_roll", "nest_grid_shift", "ring_shift"] = "nest_roll"
    rel_pos_bias: Optional[Literal["flat"]] = None
    embed_dim: int = 96
    patch_embed_norm_layer: Optional[Literal["layernorm"]] = None
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_heads: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    use_cos_attn: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: Literal["layernorm"] = "layernorm"
    use_v2_norm_placement: bool = False
    ape: bool = False
    use_checkpoint: bool = False


@dataclass
class SwinTransformerConfig:
    patch_size: Union[int, Tuple[int, int]] = (4, 4)
    window_size: Union[int, Tuple[int, int]] = (4, 4)
    shift_size: Union[int, Tuple[int, int]] = -1  # -1 -> window_size // 2
    embed_dim: int = 96
    patch_embed_norm_layer: Optional[Literal["layernorm"]] = None
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_heads: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    use_cos_attn: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: Literal["layernorm"] = "layernorm"
    use_v2_norm_placement: bool = False
    ape: bool = False
    use_checkpoint: bool = False
    final_upsample: Literal["expand_first"] = "expand_first"
    use_masking: bool = True
    use_rel_pos_bias: bool = True

    def __post_init__(self):
        if isinstance(self.patch_size, int):
            self.patch_size = (self.patch_size, self.patch_size)
        self.patch_size = tuple(self.patch_size)
        if isinstance(self.window_size, int):
            self.window_size = (self.window_size, self.window_size)
        self.window_size = tuple(self.window_size)
        if self.shift_size == -1:
            self.shift_size = (self.window_size[0] // 2, self.window_size[1] // 2)
        elif isinstance(self.shift_size, int):
            self.shift_size = (self.shift_size, self.shift_size)
        self.shift_size = tuple(self.shift_size)
```

`src/heal_swin_nnx/__init__.py`: leave empty for now (exports land in Task 18).

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_buffers.py -v` — Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: package scaffold, Buffer variable type, config dataclasses"
```

---

### Task 2: Parity environment

**Files:**
- Create: `parity/pyproject.toml`, `parity/.python-version`, `parity/README.md`, `parity/smoke_test.py`

**Interfaces:**
- Produces: a working `uv run` env in `parity/` that can import the reference models. Tasks 3–5 run `generate_goldens.py` inside it.

- [ ] **Step 1: Write env files**

`parity/.python-version`:

```
3.8
```

`parity/pyproject.toml`:

```toml
[project]
name = "heal-swin-parity"
version = "0.1.0"
description = "Legacy torch env for generating HEAL-SWIN golden parity fixtures"
requires-python = ">=3.8,<3.9"
dependencies = [
    "torch==1.8.0+cpu",
    "torchvision==0.9.0+cpu",
    "timm==0.4.12",
    "einops==0.4.0",
    "healpy==1.15.2",
    "numpy==1.19.2",
    "scipy==1.6.0",
    "matplotlib==3.3.4",
    "astropy==5.1",
]

[tool.uv]
package = false

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

`parity/README.md`:

```markdown
# Parity environment

Pinned legacy environment (Python 3.8, torch 1.8.0+cpu, timm 0.4.12, healpy
1.15.2) matching `references/HEAL-SWIN/setup.py`, used ONLY to generate the
golden fixtures in `tests/goldens/`. The main test suite never needs this env.

Regenerate all goldens:

    cd parity
    uv sync
    uv run python generate_goldens.py

Fixtures are written to `../tests/goldens/`. Commit them.
```

`parity/smoke_test.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references", "HEAL-SWIN"))

import healpy
import numpy
import timm
import torch

from heal_swin.data.segmentation.data_spec import DataSpec
from heal_swin.models_torch import hp_shifting, hp_windowing  # noqa: F401
from heal_swin.models_torch.swin_hp_transformer import SwinHPTransformerConfig, SwinHPTransformerSys
from heal_swin.models_torch.swin_transformer import SwinTransformerConfig, SwinTransformerSys  # noqa: F401

assert torch.__version__.startswith("1.8.0"), torch.__version__
assert numpy.__version__ == "1.19.2", numpy.__version__
assert healpy.__version__ == "1.15.2", healpy.__version__
assert timm.__version__ == "0.4.12", timm.__version__

cfg = SwinHPTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4], drop_path_rate=0.0)
ds = DataSpec(dim_in=2048, f_in=3, f_out=5, base_pix=8, class_names=["c%d" % i for i in range(5)])
model = SwinHPTransformerSys(cfg, ds).eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 2048))
assert out.shape == (1, 5, 2048), out.shape
print("parity env OK")
```

- [ ] **Step 2: Sync env**

Run: `cd parity && uv sync`
Expected: installs Python 3.8 + pinned wheels. If it fails with network/permission errors under sandbox, retry with sandbox disabled (pre-approved for this command). If `torch==1.8.0+cpu` cannot resolve, try `torch==1.8.0` / `torchvision==0.9.0` (the CPU index only hosts +cpu builds, so the local tag may be implied) before escalating.

- [ ] **Step 3: Run smoke test**

Run: `cd parity && uv run python smoke_test.py`
Expected: prints `parity env OK`.

- [ ] **Step 4: Commit**

```bash
git add parity && git commit -m "feat: pinned legacy parity environment (py3.8, torch 1.8.0+cpu)"
```

---

### Task 3: Goldens A — index and mask fixtures

**Files:**
- Create: `parity/generate_goldens.py` (first section), `tests/goldens/` (output: `indices.npz`)

**Interfaces:**
- Produces: `tests/goldens/indices.npz` with keys `nest_win_idcs/ws{4,16}`, and per `(nside, ws)` combo: `nest_roll/mask/ns{n}_ws{w}`, `nest_grid/{idcs,back,attn_mask,mask_raw}/ns{n}_ws{w}`, `ring/{idcs,back,attn_mask,mask_raw}/ns{n}_ws{w}`; plus `hp_rel_pos_index/ws{4,16}` (relative_position_index from a reference HP `WindowAttention` with `rel_pos_bias="flat"`). Also the shared helpers `save_case`/`to_np`/`load_meta` used by Tasks 4–5.
- Produces (test side): `tests/parity_utils.py` with `load_case(name) -> (npz, meta)`.

- [ ] **Step 1: Write the generator skeleton + index section**

`parity/generate_goldens.py`:

```python
#!/usr/bin/env python3
"""Generate golden parity fixtures from the reference HEAL-SWIN implementation.

Run inside the parity environment:

    cd parity && uv run python generate_goldens.py [--only indices|leaves|models]

Everything is deterministic (fixed seeds). Output: ../tests/goldens/*.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "references", "HEAL-SWIN"))

from heal_swin.data.segmentation.data_spec import DataSpec
from heal_swin.models_torch import hp_shifting, hp_windowing
from heal_swin.models_torch import swin_hp_transformer as hp
from heal_swin.models_torch import swin_transformer as flat

OUT_DIR = os.path.join(HERE, "..", "tests", "goldens")
SCHEMA_VERSION = 1


def to_np(t):
    return t.detach().cpu().numpy()


def save_case(name, arrays, meta):
    arrays = dict(arrays)
    arrays["schema_version"] = np.array(SCHEMA_VERSION)
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    path = os.path.join(OUT_DIR, name + ".npz")
    np.savez_compressed(path, **arrays)
    print("wrote %s (%d arrays)" % (path, len(arrays)))


def gen_indices():
    arrays = {}
    for ws in (4, 16):
        arrays["nest_win_idcs/ws%d" % ws] = to_np(hp_windowing.get_nest_win_idcs(ws))
        wa = hp.WindowAttention(dim=4, window_size=ws, num_heads=2, rel_pos_bias="flat")
        arrays["hp_rel_pos_index/ws%d" % ws] = to_np(wa.relative_position_index)
    for nside in (4, 8, 16):
        npix = 8 * nside ** 2
        for ws in (4, 16):
            tag = "ns%d_ws%d" % (nside, ws)
            if npix // ws < 8:
                continue
            nr = hp_shifting.NestRollShift(shift_size=ws // 2, input_resolution=npix, window_size=ws)
            arrays["nest_roll/mask/%s" % tag] = to_np(nr.get_mask())
            if (npix // 8) // ws < 4:
                continue  # too small for grid/ring window traversal
            ng = hp_shifting.NestGridShift(nside=nside, base_pix=8, window_size=ws)
            arrays["nest_grid/idcs/%s" % tag] = to_np(ng.shift_idcs)
            arrays["nest_grid/back/%s" % tag] = to_np(ng.back_shift_idcs)
            arrays["nest_grid/attn_mask/%s" % tag] = to_np(ng.get_mask())
            arrays["nest_grid/mask_raw/%s" % tag] = to_np(ng.get_mask(get_attn_mask=False))
            rs = hp_shifting.RingShift(nside=nside, base_pix=8, window_size=ws, shift_size=ws // 2)
            arrays["ring/idcs/%s" % tag] = to_np(rs.shift_idcs)
            arrays["ring/back/%s" % tag] = to_np(rs.back_shift_idcs)
            arrays["ring/attn_mask/%s" % tag] = to_np(rs.get_mask())
            arrays["ring/mask_raw/%s" % tag] = to_np(rs.get_mask(get_attn_mask=False))
    save_case("indices", arrays, {"base_pix": 8, "nsides": [4, 8, 16], "window_sizes": [4, 16],
                                  "shift_size": "ws//2"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["indices", "leaves", "models"], default=None)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.set_grad_enabled(True)
    if args.only in (None, "indices"):
        gen_indices()
    if args.only in (None, "leaves"):
        gen_leaves()   # Task 4
    if args.only in (None, "models"):
        gen_models()   # Task 5


def gen_leaves():
    pass  # filled in Task 4


def gen_models():
    pass  # filled in Task 5


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd parity && uv run python generate_goldens.py --only indices`
Expected: `wrote .../tests/goldens/indices.npz`. If `hp.WindowAttention(...)` at ws=16 fails an assert, check the reference constructor signature — `rel_pos_bias="flat"` is required for the index buffer to exist.

- [ ] **Step 3: Write the golden-loading helper (main env)**

`tests/parity_utils.py`:

```python
import json
import os

import numpy as np

GOLDENS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")


def load_case(name):
    npz = np.load(os.path.join(GOLDENS_DIR, name + ".npz"))
    assert int(npz["schema_version"]) == 1, "golden schema mismatch — regenerate goldens"
    meta = json.loads(bytes(npz["meta_json"].tobytes()).decode("utf-8"))
    return npz, meta


def state_dict_of(npz):
    return {k[len("sd/"):]: npz[k] for k in npz.files if k.startswith("sd/")}


def grads_of(npz):
    return {k[len("grad/"):]: npz[k] for k in npz.files if k.startswith("grad/")}
```

Sanity check: `uv run python -c "from tests.parity_utils import load_case; npz,meta = load_case('indices'); print(len(npz.files), meta)"`
Expected: prints array count and meta dict.

- [ ] **Step 4: Commit**

```bash
git add parity/generate_goldens.py tests/goldens/indices.npz tests/parity_utils.py
git commit -m "feat: golden index/mask fixtures from reference shifters"
```

---

### Task 4: Goldens B — leaf-module fixtures

**Files:**
- Modify: `parity/generate_goldens.py` (fill `gen_leaves`)
- Create: `tests/goldens/leaf_*.npz`

**Interfaces:**
- Produces one npz per leaf case, each with keys: `input`, `output`, `input_grad`, `sd/<torch key>`, `grad/<param key>`, and meta describing constructor kwargs. Case names (Tasks 11–13 and 16 consume them): `leaf_mlp`, `leaf_hp_attn`, `leaf_hp_attn_relbias`, `leaf_hp_attn_cos`, `leaf_hp_patch_merging`, `leaf_hp_patch_expand`, `leaf_hp_final_expand`, `leaf_hp_patch_embed`, `leaf_hp_block_noshift`, `leaf_hp_block_nestroll`, `leaf_hp_block_grid`, `leaf_hp_block_ring`, `leaf_flat_attn`, `leaf_flat_attn_norelbias`, `leaf_flat_attn_cos`, `leaf_flat_patch_merging`, `leaf_flat_patch_expand`, `leaf_flat_final_expand`, `leaf_flat_patch_embed`, `leaf_flat_block_noshift`, `leaf_flat_block_shift`, `leaf_flat_block_nomask`.

- [ ] **Step 1: Fill in `gen_leaves`**

Replace the `gen_leaves` stub with:

```python
def run_leaf(name, module, x, meta, call=None):
    module.eval()
    x = x.clone().detach().requires_grad_(True)
    y = call(module, x) if call is not None else module(x)
    y.sum().backward()
    arrays = {"input": to_np(x), "output": to_np(y), "input_grad": to_np(x.grad)}
    for k, v in module.state_dict().items():
        arrays["sd/%s" % k] = to_np(v)
    for k, p in module.named_parameters():
        if p.grad is not None:
            arrays["grad/%s" % k] = to_np(p.grad)
    save_case(name, arrays, meta)


def randn(*shape):
    g = torch.Generator().manual_seed(1234)
    return torch.randn(*shape, generator=g)


def gen_leaves():
    HP_DS = DataSpec(dim_in=2048, f_in=3, f_out=5, base_pix=8,
                     class_names=["c%d" % i for i in range(5)])
    FLAT_DS = DataSpec(dim_in=(32, 64), f_in=3, f_out=5, base_pix=None,
                       class_names=["c%d" % i for i in range(5)])

    torch.manual_seed(0)
    run_leaf("leaf_mlp", hp.Mlp(12, 48), randn(4, 32, 12), {"in_features": 12, "hidden_features": 48})

    for name, kw in [("leaf_hp_attn", {}),
                     ("leaf_hp_attn_relbias", {"rel_pos_bias": "flat"}),
                     ("leaf_hp_attn_cos", {"use_cos_attn": True})]:
        torch.manual_seed(0)
        m = hp.WindowAttention(dim=12, window_size=4, num_heads=2, **kw)
        if name == "leaf_hp_attn_relbias":
            with torch.no_grad():  # table is zeros at init; make the bias path non-trivial
                m.relative_position_bias_table.normal_(0, 0.02)
        run_leaf(name, m, randn(16, 4, 12), {"dim": 12, "window_size": 4, "num_heads": 2, **kw})

    torch.manual_seed(0)
    run_leaf("leaf_hp_patch_merging", hp.PatchMerging(dim=12), randn(2, 128, 12), {"dim": 12})
    torch.manual_seed(0)
    run_leaf("leaf_hp_patch_expand", hp.PatchExpand(dim=24), randn(2, 32, 24), {"dim": 24})
    torch.manual_seed(0)
    run_leaf("leaf_hp_final_expand", hp.FinalPatchExpand_X4(patch_size=4, dim=12),
             randn(2, 512, 12), {"patch_size": 4, "dim": 12})

    torch.manual_seed(0)
    hp_cfg = hp.SwinHPTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4],
                                        drop_path_rate=0.0)
    run_leaf("leaf_hp_patch_embed", hp.PatchEmbed(hp_cfg, HP_DS), randn(2, 3, 2048),
             {"patch_size": 4, "embed_dim": 12, "f_in": 3, "dim_in": 2048})

    for name, kw in [("leaf_hp_block_noshift", {"shift_size": 0}),
                     ("leaf_hp_block_nestroll", {"shift_size": 2, "shift_strategy": "nest_roll"}),
                     ("leaf_hp_block_grid", {"shift_size": 2, "shift_strategy": "nest_grid_shift"}),
                     ("leaf_hp_block_ring", {"shift_size": 2, "shift_strategy": "ring_shift"})]:
        torch.manual_seed(0)
        m = hp.SwinTransformerBlock(dim=12, input_resolution=512, base_pix=8, num_heads=2,
                                    window_size=4, **kw)
        run_leaf(name, m, randn(2, 512, 12),
                 {"dim": 12, "input_resolution": 512, "base_pix": 8, "num_heads": 2,
                  "window_size": 4, **kw})

    for name, kw in [("leaf_flat_attn", {}),
                     ("leaf_flat_attn_norelbias", {"use_rel_pos_bias": False}),
                     ("leaf_flat_attn_cos", {"use_cos_attn": True})]:
        torch.manual_seed(0)
        m = flat.WindowAttention(dim=12, window_size=(4, 4), num_heads=2, **kw)
        run_leaf(name, m, randn(8, 16, 12), {"dim": 12, "window_size": [4, 4], "num_heads": 2, **kw})

    torch.manual_seed(0)
    run_leaf("leaf_flat_patch_merging", flat.PatchMerging(input_resolution=(8, 16), dim=12),
             randn(2, 128, 12), {"input_resolution": [8, 16], "dim": 12})
    torch.manual_seed(0)
    run_leaf("leaf_flat_patch_expand", flat.PatchExpand(input_resolution=(4, 8), dim=24),
             randn(2, 32, 24), {"input_resolution": [4, 8], "dim": 24})
    torch.manual_seed(0)
    run_leaf("leaf_flat_final_expand",
             flat.FinalPatchExpand_X4(input_resolution=(8, 16), patch_size=(4, 4), dim=12),
             randn(2, 128, 12), {"input_resolution": [8, 16], "patch_size": [4, 4], "dim": 12})

    torch.manual_seed(0)
    flat_cfg = flat.SwinTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4],
                                          drop_path_rate=0.0)
    flat.SwinTransformerSys(flat_cfg, FLAT_DS)  # normalizes cfg tuple fields in-place
    torch.manual_seed(0)
    run_leaf("leaf_flat_patch_embed", flat.PatchEmbed(flat_cfg, FLAT_DS), randn(2, 3, 32, 64),
             {"patch_size": [4, 4], "embed_dim": 12, "f_in": 3, "dim_in": [32, 64]})

    for name, kw in [("leaf_flat_block_noshift", {"shift_size": [0, 0]}),
                     ("leaf_flat_block_shift", {"shift_size": [2, 2]}),
                     ("leaf_flat_block_nomask", {"shift_size": [2, 2], "use_masking": False})]:
        torch.manual_seed(0)
        m = flat.SwinTransformerBlock(dim=12, input_resolution=(8, 16), num_heads=2,
                                      window_size=[4, 4], **kw)
        run_leaf(name, m, randn(2, 128, 12),
                 {"dim": 12, "input_resolution": [8, 16], "num_heads": 2,
                  "window_size": [4, 4], **kw})
```

- [ ] **Step 2: Run and inspect**

Run: `cd parity && uv run python generate_goldens.py --only leaves`
Expected: 22 `wrote ...leaf_*.npz` lines. Then `ls -la ../tests/goldens/ && du -sh ../tests/goldens/` — total should be well under 20 MB.

- [ ] **Step 3: Commit**

```bash
git add parity/generate_goldens.py tests/goldens/leaf_*.npz
git commit -m "feat: leaf-module golden fixtures (forward + gradients)"
```

---

### Task 5: Goldens C — full-model fixtures

**Files:**
- Modify: `parity/generate_goldens.py` (fill `gen_models`)
- Create: `tests/goldens/hp_*.npz`, `tests/goldens/flat_*.npz`

**Interfaces:**
- Produces one npz per model case with keys: `input`, `output`, `input_grad`, `sd/*`, `grad/*`, `int/<hook>` (block-boundary intermediates). HP hooks: `patch_embed`, `enc_layer_0`, `enc_layer_1`, `enc_norm`, `dec_layer_up_0`, `dec_layer_up_1`, `dec_norm_up`, `dec_up`. Flat hooks: same names. Case names: `hp_base`, `hp_grid`, `hp_ring`, `hp_cos_v2`, `hp_relbias`, `hp_ape`, `flat_base`, `flat_cos_v2`, `flat_norelbias`, `flat_nomask`, `flat_ape`. Meta records the config-override dict + data spec. Tasks 14–17 consume these.

- [ ] **Step 1: Fill in `gen_models`**

```python
HP_CASES = {
    "hp_base": {},
    "hp_grid": {"shift_strategy": "nest_grid_shift"},
    "hp_ring": {"shift_strategy": "ring_shift"},
    "hp_cos_v2": {"use_cos_attn": True, "use_v2_norm_placement": True},
    "hp_relbias": {"rel_pos_bias": "flat"},
    "hp_ape": {"ape": True},
}
FLAT_CASES = {
    "flat_base": {},
    "flat_cos_v2": {"use_cos_attn": True, "use_v2_norm_placement": True},
    "flat_norelbias": {"use_rel_pos_bias": False},
    "flat_nomask": {"use_masking": False},
    "flat_ape": {"ape": True},
}


def run_model(name, model, x, hooks, meta):
    model.eval()
    inter = {}

    def mk(key):
        def hook(_mod, _inp, out):
            inter[key] = to_np(out)
        return hook

    handles = [m.register_forward_hook(mk(k)) for k, m in hooks]
    x = x.clone().detach().requires_grad_(True)
    y = model(x)
    y.sum().backward()
    for h in handles:
        h.remove()
    arrays = {"input": to_np(x), "output": to_np(y), "input_grad": to_np(x.grad)}
    for k, v in model.state_dict().items():
        arrays["sd/%s" % k] = to_np(v)
    for k, p in model.named_parameters():
        if p.grad is not None:
            arrays["grad/%s" % k] = to_np(p.grad)
    for k, v in inter.items():
        arrays["int/%s" % k] = v
    save_case(name, arrays, meta)


def gen_models():
    HP_DS = DataSpec(dim_in=2048, f_in=3, f_out=5, base_pix=8,
                     class_names=["c%d" % i for i in range(5)])
    FLAT_DS = DataSpec(dim_in=(32, 64), f_in=3, f_out=5, base_pix=None,
                       class_names=["c%d" % i for i in range(5)])

    for name, over in HP_CASES.items():
        torch.manual_seed(0)
        cfg = hp.SwinHPTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4],
                                         drop_path_rate=0.0, **over)
        model = hp.SwinHPTransformerSys(cfg, HP_DS)
        if over.get("rel_pos_bias") == "flat":
            with torch.no_grad():  # tables init to zeros; randomize so the bias path is exercised
                for mod in model.modules():
                    if hasattr(mod, "relative_position_bias_table"):
                        mod.relative_position_bias_table.normal_(0, 0.02)
        hooks = [("patch_embed", model.patch_embed)]
        hooks += [("enc_layer_%d" % i, l) for i, l in enumerate(model.layers)]
        hooks += [("enc_norm", model.norm)]
        hooks += [("dec_layer_up_%d" % i, l) for i, l in enumerate(model.decoder.layers_up)]
        hooks += [("dec_norm_up", model.decoder.norm_up), ("dec_up", model.decoder.up)]
        run_model(name, model, randn(2, 3, 2048), hooks,
                  {"overrides": over, "embed_dim": 12, "depths": [2, 2], "num_heads": [2, 4],
                   "data_spec": {"dim_in": 2048, "f_in": 3, "f_out": 5, "base_pix": 8}})

    for name, over in FLAT_CASES.items():
        torch.manual_seed(0)
        cfg = flat.SwinTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4],
                                         drop_path_rate=0.0, **over)
        model = flat.SwinTransformerSys(cfg, FLAT_DS)
        hooks = [("patch_embed", model.patch_embed)]
        hooks += [("enc_layer_%d" % i, l) for i, l in enumerate(model.layers)]
        hooks += [("enc_norm", model.norm)]
        hooks += [("dec_layer_up_%d" % i, l) for i, l in enumerate(model.layers_up)]
        hooks += [("dec_norm_up", model.norm_up), ("dec_up", model.up)]
        run_model(name, model, randn(2, 3, 32, 64), hooks,
                  {"overrides": over, "embed_dim": 12, "depths": [2, 2], "num_heads": [2, 4],
                   "data_spec": {"dim_in": [32, 64], "f_in": 3, "f_out": 5}})
```

- [ ] **Step 2: Run and inspect**

Run: `cd parity && uv run python generate_goldens.py --only models && du -sh ../tests/goldens/`
Expected: 11 `wrote ...` lines; total goldens size under ~20 MB (tiny models, compressed).

- [ ] **Step 3: Commit**

```bash
git add parity/generate_goldens.py tests/goldens/
git commit -m "feat: full-model golden fixtures with block-boundary intermediates"
```

---

### Task 6: `hp_windowing` port

**Files:**
- Create: `src/heal_swin_nnx/hp_windowing.py`
- Test: `tests/test_windowing.py`

**Interfaces:**
- Produces: `window_partition(x, window_size) -> (nW*B, ws, C)` and `window_reverse(windows, window_size, N) -> (B, N, C)` (jnp); `get_nest_win_idcs(window_size) -> np.ndarray (s, s) int64`; `nest_relative_position_index(window_size) -> np.ndarray (ws, ws) int64` (numpy, construction-time). Consumed by Tasks 8, 12, 13.

- [ ] **Step 1: Write the failing test**

`tests/test_windowing.py`:

```python
import jax.numpy as jnp
import numpy as np

from heal_swin_nnx.hp_windowing import (
    get_nest_win_idcs, nest_relative_position_index, window_partition, window_reverse)
from tests.parity_utils import load_case


def test_nest_win_idcs_bit_exact():
    npz, _ = load_case("indices")
    for ws in (4, 16):
        assert np.array_equal(get_nest_win_idcs(ws), npz["nest_win_idcs/ws%d" % ws])


def test_nest_relative_position_index_bit_exact():
    npz, _ = load_case("indices")
    for ws in (4, 16):
        assert np.array_equal(nest_relative_position_index(ws), npz["hp_rel_pos_index/ws%d" % ws])


def test_window_roundtrip():
    x = jnp.arange(2 * 64 * 3, dtype=jnp.float32).reshape(2, 64, 3)
    w = window_partition(x, 4)
    assert w.shape == (2 * 16, 4, 3)
    assert np.array_equal(window_reverse(w, 4, 64), x)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_windowing.py -v` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

`src/heal_swin_nnx/hp_windowing.py`:

```python
"""1D windowing on the HEALPix nested pixel sequence.

Port of references/HEAL-SWIN/heal_swin/models_torch/hp_windowing.py.
Runtime functions use jnp; index helpers are numpy and run at construction.
"""
import math

import jax.numpy as jnp
import numpy as np


def window_partition(x, window_size):
    """(B, N, C) -> (num_windows*B, window_size, C). window_size: power of 2."""
    assert (math.log(window_size) / math.log(2)) % 1 == 0
    B, N, C = x.shape
    return x.reshape(B * (N // window_size), window_size, C)


def window_reverse(windows, window_size, N):
    """(num_windows*B, window_size, C) -> (B, N, C)."""
    assert (math.log(window_size) / math.log(2)) % 1 == 0
    B = windows.shape[0] // (N // window_size)
    return windows.reshape(B, N, windows.shape[-1])


def get_nest_win_idcs(window_size):
    """(sqrt(ws), sqrt(ws)) int64 grid holding the nested-scheme index of each
    Cartesian position inside one window."""
    s = int(round(window_size ** 0.5))
    assert s * s == window_size
    result = np.zeros((s, s), dtype=np.int64)

    def fill_quadrant(idx, x, y, size):
        if size == 2:
            result[x, y + 1] = idx
            result[x, y] = idx + 1
            result[x + 1, y + 1] = idx + 2
            result[x + 1, y] = idx + 3
        else:
            fill_quadrant(idx, x, y + size // 2, size // 2)
            fill_quadrant(idx + size ** 2 // 4, x, y, size // 2)
            fill_quadrant(idx + 2 * (size ** 2 // 4), x + size // 2, y + size // 2, size // 2)
            fill_quadrant(idx + 3 * (size ** 2 // 4), x + size // 2, y, size // 2)

    fill_quadrant(0, 0, 0, s)
    return result


def nest_relative_position_index(window_size):
    """(ws, ws) int64 relative-position index for HP window attention:
    the standard 2D SWIN index, re-ordered from Cartesian to nested scheme."""
    s = int(round(window_size ** 0.5))
    coords = np.stack(np.meshgrid(np.arange(s), np.arange(s), indexing="ij"))  # 2, s, s
    flat = coords.reshape(2, -1)
    rel = flat[:, :, None] - flat[:, None, :]  # 2, ws, ws
    rel = rel.transpose(1, 2, 0).astype(np.int64)
    rel[:, :, 0] += s - 1
    rel[:, :, 1] += s - 1
    rel[:, :, 0] *= 2 * s - 1
    idx = rel.sum(-1)
    inv = np.argsort(get_nest_win_idcs(window_size).reshape(-1))
    return idx[inv][:, inv]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_windowing.py -v` — Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/hp_windowing.py tests/test_windowing.py
git commit -m "feat: hp_windowing port with bit-exact index parity"
```

---

### Task 7: `hp_shifting` — mask helper, `NoShift`, `NestRollShift`

**Files:**
- Create: `src/heal_swin_nnx/hp_shifting.py`
- Test: `tests/test_shifting.py`

**Interfaces:**
- Produces (numpy, construction-time): `get_attn_mask_from_mask(mask, window_size) -> np.float32 (nW, ws, ws)`; `nest_roll_mask(input_resolution, window_size, shift_size) -> np.float32 (nW, ws, ws)`.
- Produces (nnx wrappers, used by HP blocks in Task 13): `NoShift()` with `.attn_mask = None`, `.shift(x)`, `.shift_back(x)`; `NestRollShift(shift_size, input_resolution, window_size)` with `.attn_mask: Buffer`, `.shift/.shift_back` = `jnp.roll` by ∓shift_size on axis 1.

- [ ] **Step 1: Write the failing test**

`tests/test_shifting.py`:

```python
import jax.numpy as jnp
import numpy as np
import pytest

from heal_swin_nnx import hp_shifting as hps
from tests.parity_utils import load_case


def test_nest_roll_mask_bit_exact():
    npz, _ = load_case("indices")
    for nside in (4, 8, 16):
        for ws in (4, 16):
            key = "nest_roll/mask/ns%d_ws%d" % (nside, ws)
            if key not in npz.files:
                continue
            got = hps.nest_roll_mask(8 * nside ** 2, ws, ws // 2)
            assert np.array_equal(got, npz[key]), key


def test_nest_roll_shift_roundtrip():
    sh = hps.NestRollShift(shift_size=2, input_resolution=64, window_size=4)
    x = jnp.arange(2 * 64 * 3, dtype=jnp.float32).reshape(2, 64, 3)
    assert np.array_equal(sh.shift_back(sh.shift(x)), x)
    assert np.array_equal(np.asarray(sh.shift(x)), np.roll(np.asarray(x), -2, axis=1))


def test_noshift_is_identity():
    sh = hps.NoShift()
    x = jnp.ones((1, 8, 2))
    assert sh.attn_mask is None
    assert np.array_equal(sh.shift(x), x) and np.array_equal(sh.shift_back(x), x)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_shifting.py -v` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

`src/heal_swin_nnx/hp_shifting.py` (first section):

```python
"""HEALPix window shifting. Port of models_torch/hp_shifting.py.

All index/mask computation is numpy at construction time; the nnx wrapper
classes store the results as Buffers and apply pure gathers/rolls at runtime.

The *_TABLES constants encode base-pixel adjacency for the 8-base-pixel
fisheye subset used by the reference. Full-sphere (base_pix=12) tables are a
planned extension (see the design spec); unsupported base_pix raises
NotImplementedError loudly rather than computing garbage.
"""
import math

import jax.numpy as jnp
import numpy as np
from flax import nnx

from heal_swin_nnx.variables import Buffer


def get_attn_mask_from_mask(mask, window_size):
    """(N,) int-valued region mask -> (nW, ws, ws) float32 attention mask in {0, -100}."""
    mask_windows = np.asarray(mask, dtype=np.float32).reshape(-1, window_size)
    attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
    return np.where(attn_mask != 0, np.float32(-100.0), np.float32(0.0))


def nest_roll_mask(input_resolution, window_size, shift_size):
    img_mask = np.zeros(input_resolution, dtype=np.float32)
    slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    for cnt, s in enumerate(slices):
        img_mask[s] = cnt
    return get_attn_mask_from_mask(img_mask, window_size)


class NoShift(nnx.Module):
    def __init__(self):
        self.attn_mask = None

    def shift(self, x):
        return x

    def shift_back(self, x):
        return x


class NestRollShift(nnx.Module):
    def __init__(self, shift_size, input_resolution, window_size):
        self.shift_size = shift_size
        self.attn_mask = Buffer(jnp.asarray(
            nest_roll_mask(input_resolution, window_size, shift_size)))

    def shift(self, x):
        return jnp.roll(x, -self.shift_size, axis=1)

    def shift_back(self, x):
        return jnp.roll(x, self.shift_size, axis=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_shifting.py -v` — Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/heal_swin_nnx/hp_shifting.py tests/test_shifting.py
git commit -m "feat: hp_shifting mask helper, NoShift, NestRollShift"
```

---

### Task 8: `hp_shifting` — `NestGridShift`

**Files:**
- Modify: `src/heal_swin_nnx/hp_shifting.py`
- Test: `tests/test_shifting.py` (append)

**Interfaces:**
- Produces (numpy): `nest_grid_shift_idcs(nside, base_pix, window_size) -> np.int64 (npix,)`; `nest_grid_mask(nside, base_pix, window_size) -> np.float64 (npix,)` (raw region mask).
- Produces (nnx): `NestGridShift(nside, base_pix, window_size)` with `.shift_idcs`, `.back_shift_idcs`, `.attn_mask` Buffers and `.shift/.shift_back` gathers. Raises `NotImplementedError` for `base_pix` not in the topology tables.
- Topology tables: `NEST_GRID_BASE_PIX_OFFSETS_DIR1/2: Dict[int, Dict[int, int]]` keyed by `base_pix`.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_shifting.py`:

```python
def _grid_combos(npz):
    for nside in (4, 8, 16):
        for ws in (4, 16):
            if "nest_grid/idcs/ns%d_ws%d" % (nside, ws) in npz.files:
                yield nside, ws


def test_nest_grid_idcs_bit_exact():
    npz, _ = load_case("indices")
    for nside, ws in _grid_combos(npz):
        tag = "ns%d_ws%d" % (nside, ws)
        got = hps.nest_grid_shift_idcs(nside, 8, ws)
        assert np.array_equal(got, npz["nest_grid/idcs/%s" % tag]), tag
        assert np.array_equal(np.argsort(got), npz["nest_grid/back/%s" % tag]), tag


def test_nest_grid_masks_bit_exact():
    npz, _ = load_case("indices")
    for nside, ws in _grid_combos(npz):
        tag = "ns%d_ws%d" % (nside, ws)
        raw = hps.nest_grid_mask(nside, 8, ws)
        assert np.array_equal(raw, npz["nest_grid/mask_raw/%s" % tag]), tag
        attn = hps.get_attn_mask_from_mask(raw, ws)
        assert np.array_equal(attn, npz["nest_grid/attn_mask/%s" % tag]), tag


def test_nest_grid_module_and_unsupported_base_pix():
    sh = hps.NestGridShift(nside=16, base_pix=8, window_size=4)
    x = jnp.arange(1 * 2048 * 2, dtype=jnp.float32).reshape(1, 2048, 2)
    assert np.array_equal(sh.shift_back(sh.shift(x)), x)
    with pytest.raises(NotImplementedError):
        hps.NestGridShift(nside=16, base_pix=12, window_size=4)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_shifting.py -v -k grid` — Expected: FAIL (missing attributes).

- [ ] **Step 3: Implement**

Append to `src/heal_swin_nnx/hp_shifting.py`. This is a verbatim numpy port of the reference (`hp_shifting.py:76-306`); python int arithmetic (including negative `%` semantics) is identical to the reference, which also ran on python ints.

```python
# --- 8-base-pixel fisheye-subset topology (reference HEAL-SWIN). Keyed by base_pix
# so that full-sphere tables can be added without touching traversal logic. ---
NEST_GRID_BASE_PIX_OFFSETS_DIR1 = {8: {0: 2, 1: 2, 2: 2, 3: 6, 4: 3, 5: 3, 6: 3, 7: 3}}
NEST_GRID_BASE_PIX_OFFSETS_DIR2 = {8: {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3}}
NEST_GRID_MASKED_BASE_PIX = {8: [4, 5, 6, 7]}
NEST_GRID_LEFT_CARRY_OVER_BASE_PIX = {8: [0, 1, 2, 3]}


def _log4(x):
    return int(math.log(x) / math.log(4))


def _get_scale(idx, ws, base_pix_len):
    assert idx % ws == 0
    w_idx = idx // ws
    scale = base_pix_len
    while w_idx % scale != 0:
        scale //= 4
    return _log4(scale)


def _get_offset_dir1(idx, ws, base_pix_len, base_pix_offsets):
    assert idx % ws == 0
    while True:
        scale = _get_scale(idx, ws, base_pix_len)
        idx -= ws * 4 ** scale
        if scale >= _get_scale(idx, ws, base_pix_len):
            break
    offset = sum(ws * 4 ** power for power in range(0, scale + 1))
    if scale == _log4(base_pix_len):
        idx += ws * 4 ** scale
        offset -= base_pix_len * ws
        base_pix = idx // (base_pix_len * ws)
        offset += base_pix_offsets[base_pix] * base_pix_len * ws
    return offset


def _get_offset_dir2(idx, ws, base_pix_len, base_pix_offsets):
    assert idx % ws == 0
    scale = _get_scale(idx, ws, base_pix_len)
    while (idx % (ws * 4 ** (scale + 1))) // (ws * 4 ** scale) == 2:
        idx -= 2 * ws * 4 ** scale
        scale = _get_scale(idx, ws, base_pix_len)
    offset = sum(2 * ws * 4 ** power for power in range(0, scale))
    if scale == _log4(base_pix_len):
        base_pix = idx // (base_pix_len * ws)
        offset += base_pix_offsets[base_pix] * base_pix_len * ws
    return offset


def nest_grid_shift_idcs(nside, base_pix, window_size):
    ws = window_size
    npix = base_pix * nside ** 2
    n_windows = npix // ws
    base_pix_len = (npix // base_pix) // ws
    hws, qws = ws // 2, ws // 4
    off1 = NEST_GRID_BASE_PIX_OFFSETS_DIR1[base_pix]
    off2 = NEST_GRID_BASE_PIX_OFFSETS_DIR2[base_pix]

    dir1 = np.zeros(npix, dtype=np.int64)
    for w in range(n_windows):
        first = w * ws
        os_ = _get_offset_dir1(first, ws, base_pix_len, off1)
        dir1[first:first + hws] = np.arange(first - os_ - hws, first - os_)
        dir1[first + hws:first + ws] = np.arange(first, first + hws)
    dir1 %= npix

    dir2 = np.zeros(npix, dtype=np.int64)
    for w in range(n_windows):
        first = w * ws
        os_ = _get_offset_dir2(first, ws, base_pix_len, off2)
        dir2[first:first + qws] = np.arange(first - os_ - hws - qws, first - os_ - hws)
        dir2[first + qws:first + hws] = np.arange(first, first + qws)
        dir2[first + hws:first + hws + qws] = np.arange(first - os_ - qws, first - os_)
        dir2[first + hws + qws:first + ws] = np.arange(first + hws, first + hws + qws)
    dir2 %= npix

    result = dir1[dir2]
    assert np.array_equal(np.sort(result), np.arange(npix)), (
        "shift validation failed for nside=%d, window_size=%d" % (nside, ws))
    return result


def nest_grid_mask(nside, base_pix, window_size):
    ws = window_size
    hws, qws = ws // 2, ws // 4
    npix = base_pix * nside ** 2
    base_pix_len = (npix // base_pix) // ws
    masked = NEST_GRID_MASKED_BASE_PIX[base_pix]
    carry = NEST_GRID_LEFT_CARRY_OVER_BASE_PIX[base_pix]
    mask = np.zeros(npix)

    def right_mask_subset(first, size, mask_value):
        if size == ws:
            mask[first:first + qws] = mask_value
            mask[first + hws:first + hws + qws] = mask_value
        else:
            right_mask_subset(first, size // 4, mask_value)
            right_mask_subset(first + 2 * size // 4, size // 4, mask_value)

    def left_mask_subset(first, size, mask_value):
        if size == ws:
            mask[first:first + hws] = mask_value
        else:
            left_mask_subset(first, size // 4, mask_value)
            left_mask_subset(first + size // 4, size // 4, mask_value)

    for b, co in zip(masked, carry):
        left_mask_subset(b * base_pix_len * ws, base_pix_len * ws, b + 1)
        right_mask_subset(b * base_pix_len * ws, base_pix_len * ws, b + 1 + len(masked))
        first_co = co * base_pix_len * ws
        mask[first_co:first_co + qws] = b + 1
    return mask


class NestGridShift(nnx.Module):
    def __init__(self, nside, base_pix, window_size):
        if base_pix not in NEST_GRID_BASE_PIX_OFFSETS_DIR1:
            raise NotImplementedError(
                "NestGridShift topology tables only exist for base_pix in %s; "
                "full-sphere support is a planned extension (see design spec)"
                % sorted(NEST_GRID_BASE_PIX_OFFSETS_DIR1))
        idcs = nest_grid_shift_idcs(nside, base_pix, window_size)
        self.shift_idcs = Buffer(jnp.asarray(idcs))
        self.back_shift_idcs = Buffer(jnp.asarray(np.argsort(idcs)))
        self.attn_mask = Buffer(jnp.asarray(
            get_attn_mask_from_mask(nest_grid_mask(nside, base_pix, window_size), window_size)))

    def shift(self, x):
        return jnp.take(x, self.shift_idcs.value, axis=1)

    def shift_back(self, x):
        return jnp.take(x, self.back_shift_idcs.value, axis=1)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_shifting.py -v` — Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: NestGridShift numpy port, bit-exact vs reference"
```

---

### Task 9: `hp_shifting` — `RingShift`

**Files:**
- Modify: `src/heal_swin_nnx/hp_shifting.py`
- Test: `tests/test_shifting.py` (append)

**Interfaces:**
- Produces: `ring_shift_idcs_and_mask(nside, base_pix, window_size, shift_size) -> (np.int64 (npix,), np.int64 (npix,))`; `RingShift(nside, base_pix, window_size, shift_size)` nnx module (same surface as `NestGridShift`). Table: `RING_GET_LOST_FROM: Dict[int, Dict[int, int]]`. Raises `NotImplementedError` for unsupported `base_pix` (this assert is MISSING in the reference — we add it, per spec).

- [ ] **Step 1: Append failing tests**

```python
def test_ring_idcs_and_masks_bit_exact():
    npz, _ = load_case("indices")
    for nside in (4, 8, 16):
        for ws in (4, 16):
            tag = "ns%d_ws%d" % (nside, ws)
            if "ring/idcs/%s" % tag not in npz.files:
                continue
            idcs, raw = hps.ring_shift_idcs_and_mask(nside, 8, ws, ws // 2)
            assert np.array_equal(idcs, npz["ring/idcs/%s" % tag]), tag
            assert np.array_equal(np.argsort(idcs), npz["ring/back/%s" % tag]), tag
            assert np.array_equal(raw, npz["ring/mask_raw/%s" % tag]), tag
            attn = hps.get_attn_mask_from_mask(raw, ws)
            assert np.array_equal(attn, npz["ring/attn_mask/%s" % tag]), tag


def test_ring_module_and_unsupported_base_pix():
    sh = hps.RingShift(nside=16, base_pix=8, window_size=4, shift_size=2)
    x = jnp.arange(1 * 2048 * 2, dtype=jnp.float32).reshape(1, 2048, 2)
    assert np.array_equal(sh.shift_back(sh.shift(x)), x)
    with pytest.raises(NotImplementedError):
        hps.RingShift(nside=16, base_pix=12, window_size=4, shift_size=2)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_shifting.py -v -k ring` → FAIL.

- [ ] **Step 3: Implement**

Append (verbatim numpy port of reference `hp_shifting.py:309-404`; healpy calls unchanged):

```python
RING_GET_LOST_FROM = {8: {4: 7, 5: 4, 6: 5, 7: 6}}


def ring_shift_idcs_and_mask(nside, base_pix, window_size, shift_size):
    import healpy as hp  # local import: healpy pulls matplotlib; keep module import light

    npix = base_pix * nside ** 2
    get_lost_from = RING_GET_LOST_FROM[base_pix]

    ring_idcs = np.arange(12 * nside ** 2)
    shifted_ring_idcs = np.roll(ring_idcs, shift_size)
    shifted_ring_idcs_in_nest = hp.pixelfunc.ring2nest(nside, shifted_ring_idcs)

    nest_idcs = np.arange(npix)
    nest_idcs_in_ring = hp.pixelfunc.nest2ring(nside, nest_idcs)
    result = shifted_ring_idcs_in_nest[nest_idcs_in_ring]

    max_idx = nest_idcs.max()
    pixel_size = nside ** 2
    mask = np.zeros(npix)
    for i in range(base_pix):
        subset_slice = slice(i * pixel_size, (i + 1) * pixel_size)
        mask_subset = mask[subset_slice]
        result_subset = result[subset_slice]
        mask_subset[result_subset > max_idx] = i + 1

    lost_pix = []
    for i in range(base_pix):
        lost_pix.append(np.setdiff1d(np.arange(i * pixel_size, (i + 1) * pixel_size), result))

    unused_source_pix = []
    for i in range(4, base_pix):
        subset_slice = slice(i * pixel_size, (i + 1) * pixel_size)
        result_subset = result[subset_slice]
        source_pix = lost_pix[get_lost_from[i]]
        pix_to_be_filled = result_subset[result_subset > max_idx]
        assert pix_to_be_filled.shape[0] <= source_pix.shape[0], (
            "for base pixel %d, there were not enough source pixel" % i)
        result_subset[result_subset > max_idx] = source_pix[:pix_to_be_filled.shape[0]]
        unused_source_pix.append(source_pix[pix_to_be_filled.shape[0]:])
    unused_pix = np.concatenate(unused_source_pix).flatten()

    assert unused_pix.shape[0] == (result > max_idx).sum(), (
        "the number of unused source pixels does not match the number of pixels to be filled")
    first = 0
    for i in range(4):
        subset_slice = slice(i * pixel_size, (i + 1) * pixel_size)
        result_subset = result[subset_slice]
        no_to_be_filled = result_subset[result_subset > max_idx].shape[0]
        result_subset[result_subset > max_idx] = unused_pix[first:first + no_to_be_filled]
        first += no_to_be_filled

    result = result.astype(np.int64)
    assert np.array_equal(np.sort(result), np.arange(npix)), (
        "shift validation failed for nside=%d, window_size=%d" % (nside, window_size))
    return result, mask.astype(np.int64)


class RingShift(nnx.Module):
    def __init__(self, nside, base_pix, window_size, shift_size):
        # The reference silently assumes base_pix == 8 here; we assert loudly (spec).
        if base_pix not in RING_GET_LOST_FROM:
            raise NotImplementedError(
                "RingShift backfill tables only exist for base_pix in %s; "
                "full-sphere support is a planned extension (see design spec)"
                % sorted(RING_GET_LOST_FROM))
        idcs, raw_mask = ring_shift_idcs_and_mask(nside, base_pix, window_size, shift_size)
        self.shift_idcs = Buffer(jnp.asarray(idcs))
        self.back_shift_idcs = Buffer(jnp.asarray(np.argsort(idcs)))
        self.attn_mask = Buffer(jnp.asarray(get_attn_mask_from_mask(raw_mask, window_size)))

    def shift(self, x):
        return jnp.take(x, self.shift_idcs.value, axis=1)

    def shift_back(self, x):
        return jnp.take(x, self.back_shift_idcs.value, axis=1)
```

Note: the loops mutate `mask[subset_slice]` / `result[subset_slice]` through numpy views — this is exactly how the reference works; do not "clean it up".

- [ ] **Step 4: Run full shifting suite** — `uv run pytest tests/test_shifting.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: RingShift port with healpy ring/nest maps, bit-exact"
```

---

### Task 10: Shared leaves — `Identity`, `DropPath`, `Mlp`

**Files:**
- Create: `src/heal_swin_nnx/layers.py`
- Test: `tests/test_model.py` (start file)

**Interfaces:**
- Produces: `TRUNC_NORMAL` (kernel init, stddev 0.02); `Identity()`; `DropPath(rate, *, rngs)` with `.deterministic` toggled by `model.eval()/train()`; `Mlp(in_features, hidden_features=None, out_features=None, drop=0.0, *, rngs)`. Consumed by Tasks 12–17.

- [ ] **Step 1: Write failing tests**

`tests/test_model.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from heal_swin_nnx.layers import DropPath, Identity, Mlp


def test_identity():
    x = jnp.ones((2, 3))
    assert np.array_equal(Identity()(x), x)


def test_mlp_shapes():
    m = Mlp(8, 32, rngs=nnx.Rngs(0))
    y = m(jnp.ones((2, 5, 8)))
    assert y.shape == (2, 5, 8)


def test_droppath_eval_is_identity():
    dp = DropPath(0.5, rngs=nnx.Rngs(0))
    dp.eval()
    x = jnp.ones((4, 3, 2))
    assert np.array_equal(dp(x), x)


def test_droppath_train_drops_whole_samples():
    dp = DropPath(0.5, rngs=nnx.Rngs(0))
    dp.train()
    x = jnp.ones((512, 4))
    y = np.asarray(dp(x))
    per_sample = y.sum(axis=1)
    assert set(np.round(per_sample, 3)).issubset({0.0, 8.0})  # 4 * 1/keep, keep=0.5
    dropped = float((per_sample == 0).mean())
    assert 0.3 < dropped < 0.7  # ~Bernoulli(0.5)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_model.py -v` → FAIL.

- [ ] **Step 3: Implement**

`src/heal_swin_nnx/layers.py`:

```python
"""Shared leaf modules (identical between HP and flat models in the reference)."""
import jax
import jax.numpy as jnp
from flax import nnx

# Distributional mirror of timm's trunc_normal_(std=0.02). (jax truncates at
# +-2 sigma, timm at absolute +-2 — irrelevant for parity tests, which always
# use transferred weights.)
TRUNC_NORMAL = nnx.initializers.truncated_normal(stddev=0.02)


class Identity(nnx.Module):
    def __call__(self, x):
        return x


class DropPath(nnx.Module):
    """Stochastic depth per sample (port of timm 0.4.12 drop_path)."""

    def __init__(self, rate, *, rngs):
        self.rate = rate
        self.deterministic = False
        self.rngs = rngs

    def __call__(self, x):
        if self.deterministic or self.rate == 0.0:
            return x
        keep_prob = 1.0 - self.rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = jax.random.bernoulli(self.rngs.dropout(), keep_prob, shape)
        return jnp.where(mask, x / keep_prob, jnp.zeros_like(x))


class Mlp(nnx.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, *, rngs):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nnx.Linear(in_features, hidden_features, kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_features, out_features, kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        x = self.fc1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_model.py -v` → 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: shared leaves (Identity, DropPath, Mlp)"
```

---

### Task 11: `weight_transfer` + first leaf parity test (Mlp)

**Files:**
- Create: `src/heal_swin_nnx/weight_transfer.py`
- Test: `tests/test_parity_modules.py` (start file)

**Interfaces:**
- Produces:
  - `torch_key_to_path(key: str, prefix_map: dict | None) -> tuple` — splits a torch state_dict key, applies the prefix map (None = identity), converts digit segments to int, renames the leaf: `weight` → `scale` if the parent segment contains `"norm"`, else `weight` → `kernel`; everything else unchanged.
  - `transform_array(arr: np.ndarray, renamed_leaf: str) -> np.ndarray` — if `renamed_leaf == "kernel"`: ndim 2 → `arr.T`; ndim 3 → `arr.transpose(2, 1, 0)` (Conv1d); ndim 4 → `arr.transpose(2, 3, 1, 0)` (Conv2d). Otherwise unchanged.
  - `load_torch_state(model: nnx.Module, sd: dict[str, np.ndarray], prefix_map: dict | None = None) -> None` — applies both, skips keys ending in `attn_mask`/`relative_position_index`, asserts every non-skipped torch key was consumed AND every `nnx.Param` in the model was assigned (raise `ValueError` listing the offenders).
  - `HP_PREFIX_MAP`, `FLAT_PREFIX_MAP` (used from Task 14/17):

```python
HP_PREFIX_MAP = {"patch_embed": ("encoder", "patch_embed"), "layers": ("encoder", "layers"),
                 "norm": ("encoder", "norm"), "absolute_pos_embed": ("encoder", "absolute_pos_embed"),
                 "decoder": ("decoder",)}
FLAT_PREFIX_MAP = {"patch_embed": ("encoder", "patch_embed"), "layers": ("encoder", "layers"),
                   "norm": ("encoder", "norm"), "absolute_pos_embed": ("encoder", "absolute_pos_embed"),
                   "layers_up": ("decoder", "layers_up"), "concat_back_dim": ("decoder", "concat_back_dim"),
                   "norm_up": ("decoder", "norm_up"), "up": ("decoder", "up"), "output": ("decoder", "output")}
```

- [ ] **Step 1: Write failing test**

`tests/test_parity_modules.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from heal_swin_nnx import layers
from heal_swin_nnx.weight_transfer import load_torch_state, torch_key_to_path, transform_array
from tests.parity_utils import grads_of, load_case, state_dict_of

FWD = dict(rtol=1e-5, atol=1e-6)
GRD = dict(rtol=1e-4, atol=1e-6)


def check_param_grads(nnx_grads, torch_grads, prefix_map=None):
    flat = {tuple(str(p) for p in path): v for path, v in nnx_grads.flat_state()}
    for tkey, tgrad in torch_grads.items():
        path = tuple(str(p) for p in torch_key_to_path(tkey, prefix_map))
        leaf = path[-1]
        expected = transform_array(tgrad, leaf)
        got = np.asarray(flat[path].value)
        np.testing.assert_allclose(got, expected, err_msg=tkey, **GRD)


def test_mlp_parity():
    npz, meta = load_case("leaf_mlp")
    m = layers.Mlp(meta["in_features"], meta["hidden_features"], rngs=nnx.Rngs(0))
    load_torch_state(m, state_dict_of(npz))
    m.eval()
    x = jnp.asarray(npz["input"])
    np.testing.assert_allclose(np.asarray(m(x)), npz["output"], **FWD)
    gx = jax.grad(lambda x: m(x).sum())(x)
    np.testing.assert_allclose(np.asarray(gx), npz["input_grad"], **GRD)
    gp = nnx.grad(lambda m: m(x).sum())(m)
    check_param_grads(gp, grads_of(npz))


def test_transfer_completeness_raises_on_missing():
    import pytest
    npz, meta = load_case("leaf_mlp")
    m = layers.Mlp(meta["in_features"], meta["hidden_features"], rngs=nnx.Rngs(0))
    sd = state_dict_of(npz)
    sd.pop("fc1.bias")
    with pytest.raises(ValueError):
        load_torch_state(m, sd)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_parity_modules.py -v` → FAIL.

- [ ] **Step 3: Implement**

`src/heal_swin_nnx/weight_transfer.py`:

```python
"""Load reference torch state_dicts (saved as npz arrays) into nnx models.

The nnx module tree mirrors the torch tree, so mapping is mechanical:
prefix rewrite (encoder/decoder seam) + leaf rename + layout transform.
"""
import jax.numpy as jnp
import numpy as np
from flax import nnx

HP_PREFIX_MAP = {"patch_embed": ("encoder", "patch_embed"), "layers": ("encoder", "layers"),
                 "norm": ("encoder", "norm"), "absolute_pos_embed": ("encoder", "absolute_pos_embed"),
                 "decoder": ("decoder",)}
FLAT_PREFIX_MAP = {"patch_embed": ("encoder", "patch_embed"), "layers": ("encoder", "layers"),
                   "norm": ("encoder", "norm"), "absolute_pos_embed": ("encoder", "absolute_pos_embed"),
                   "layers_up": ("decoder", "layers_up"), "concat_back_dim": ("decoder", "concat_back_dim"),
                   "norm_up": ("decoder", "norm_up"), "up": ("decoder", "up"), "output": ("decoder", "output")}

SKIP_SUFFIXES = ("attn_mask", "relative_position_index")


def torch_key_to_path(key, prefix_map=None):
    parts = key.split(".")
    if prefix_map is not None:
        parts = list(prefix_map[parts[0]]) + parts[1:]
    if parts[-1] == "weight":
        parent = parts[-2] if len(parts) >= 2 else ""
        parts[-1] = "scale" if "norm" in parent else "kernel"
    return tuple(int(p) if p.isdigit() else p for p in parts)


def transform_array(arr, renamed_leaf):
    if renamed_leaf != "kernel":
        return arr
    if arr.ndim == 2:            # nn.Linear (out, in) -> (in, out)
        return arr.T
    if arr.ndim == 3:            # nn.Conv1d (out, in, k) -> (k, in, out)
        return arr.transpose(2, 1, 0)
    if arr.ndim == 4:            # nn.Conv2d (out, in, kh, kw) -> (kh, kw, in, out)
        return arr.transpose(2, 3, 1, 0)
    return arr


def load_torch_state(model, sd, prefix_map=None):
    state = nnx.state(model)
    flat = dict(state.flat_state())
    param_paths = {path for path, v in flat.items() if isinstance(v, nnx.VariableState)
                   and v.type is nnx.Param}
    assigned = set()
    for key, arr in sd.items():
        if key.endswith(SKIP_SUFFIXES):
            continue
        path = torch_key_to_path(key, prefix_map)
        if path not in flat:
            raise ValueError("torch key %r maps to %r which is not in the nnx state" % (key, path))
        value = transform_array(np.asarray(arr), path[-1])
        if tuple(flat[path].value.shape) != tuple(value.shape):
            raise ValueError("shape mismatch for %r: nnx %s vs torch %s"
                             % (key, flat[path].value.shape, value.shape))
        flat[path].value = jnp.asarray(value)
        assigned.add(path)
    missing = param_paths - assigned
    if missing:
        raise ValueError("nnx Params not assigned by transfer: %s" % sorted(missing))
    nnx.update(model, nnx.State.from_flat_path(flat))
```

Note: if the installed flax version exposes `flat_state()` as a list of pairs or `VariableState.type` differently, adapt the two touched lines — the contract (dict path→variable-state, param filter, `from_flat_path`) is stable.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_parity_modules.py -v` → 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: weight transfer torch->nnx with completeness checks; Mlp parity green"
```

---

### Task 12: HP `WindowAttention` + HP patch merge/expand leaves

**Files:**
- Create: `src/heal_swin_nnx/swin_hp_transformer.py` (first section)
- Test: `tests/test_parity_modules.py` (append)

**Interfaces:**
- Produces (in `heal_swin_nnx.swin_hp_transformer`): `WindowAttention(dim, window_size, num_heads, rel_pos_bias=None, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_cos_attn=False, *, rngs)` — `__call__(x, mask=None)`, x `(nW*B, ws, C)`; `PatchMerging(dim, dim_scale=2, *, rngs)`; `PatchExpand(dim, dim_scale=2, *, rngs)`; `FinalPatchExpand_X4(patch_size, dim, *, rngs)`. Consumed by Task 13/14.

- [ ] **Step 1: Append failing parity tests**

```python
from heal_swin_nnx import swin_hp_transformer as hp


def _leaf_forward_and_grads(m, npz, mask=None):
    m.eval()
    x = jnp.asarray(npz["input"])
    call = (lambda mod, x: mod(x, mask=mask)) if mask is not None else (lambda mod, x: mod(x))
    np.testing.assert_allclose(np.asarray(call(m, x)), npz["output"], **FWD)
    gx = jax.grad(lambda x: call(m, x).sum())(x)
    np.testing.assert_allclose(np.asarray(gx), npz["input_grad"], **GRD)
    gp = nnx.grad(lambda m: call(m, x).sum())(m)
    check_param_grads(gp, grads_of(npz))


def test_hp_window_attention_parity():
    for case in ("leaf_hp_attn", "leaf_hp_attn_relbias", "leaf_hp_attn_cos"):
        npz, meta = load_case(case)
        m = hp.WindowAttention(dim=meta["dim"], window_size=meta["window_size"],
                               num_heads=meta["num_heads"],
                               rel_pos_bias=meta.get("rel_pos_bias"),
                               use_cos_attn=meta.get("use_cos_attn", False),
                               rngs=nnx.Rngs(0))
        load_torch_state(m, state_dict_of(npz))
        _leaf_forward_and_grads(m, npz)


def test_hp_rel_pos_index_buffer_matches_reference():
    npz, _ = load_case("leaf_hp_attn_relbias")
    m = hp.WindowAttention(dim=12, window_size=4, num_heads=2, rel_pos_bias="flat",
                           rngs=nnx.Rngs(0))
    assert np.array_equal(np.asarray(m.relative_position_index.value),
                          npz["sd/relative_position_index"])


def test_hp_patch_merge_expand_parity():
    npz, meta = load_case("leaf_hp_patch_merging")
    m = hp.PatchMerging(dim=meta["dim"], rngs=nnx.Rngs(0))
    load_torch_state(m, state_dict_of(npz))
    _leaf_forward_and_grads(m, npz)

    npz, meta = load_case("leaf_hp_patch_expand")
    m = hp.PatchExpand(dim=meta["dim"], rngs=nnx.Rngs(0))
    load_torch_state(m, state_dict_of(npz))
    _leaf_forward_and_grads(m, npz)

    npz, meta = load_case("leaf_hp_final_expand")
    m = hp.FinalPatchExpand_X4(patch_size=meta["patch_size"], dim=meta["dim"], rngs=nnx.Rngs(0))
    load_torch_state(m, state_dict_of(npz))
    _leaf_forward_and_grads(m, npz)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_parity_modules.py -v -k hp` → FAIL.

- [ ] **Step 3: Implement**

`src/heal_swin_nnx/swin_hp_transformer.py` (start):

```python
"""HEAL-SWIN on the HEALPix grid. Port of models_torch/swin_hp_transformer.py."""
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx

from heal_swin_nnx import hp_shifting
from heal_swin_nnx.config import DataSpec, SwinHPTransformerConfig
from heal_swin_nnx.hp_windowing import (
    nest_relative_position_index, window_partition, window_reverse)
from heal_swin_nnx.layers import TRUNC_NORMAL, DropPath, Identity, Mlp
from heal_swin_nnx.variables import Buffer

LN_EPS = 1e-5  # torch nn.LayerNorm default; flax default (1e-6) breaks parity


class WindowAttention(nnx.Module):
    def __init__(self, dim, window_size, num_heads, rel_pos_bias=None, qkv_bias=True,
                 qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_cos_attn=False, *, rngs):
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.use_cos_attn = use_cos_attn
        self.rel_pos_bias = rel_pos_bias
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if use_cos_attn:
            self.logit_scale = nnx.Param(jnp.log(10.0 * jnp.ones((num_heads, 1, 1))))
        if rel_pos_bias == "flat":
            s = int(round(window_size ** 0.5))
            # zeros init: the reference's trunc_normal_ call for this table is commented out
            self.relative_position_bias_table = nnx.Param(
                jnp.zeros(((2 * s - 1) ** 2, num_heads)))
            self.relative_position_index = Buffer(
                jnp.asarray(nest_relative_position_index(window_size)))

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.attn_drop = nnx.Dropout(attn_drop, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)

    def __call__(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_cos_attn:
            qn = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
            kn = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-12)
            attn = qn @ kn.swapaxes(-2, -1)
            logit_scale = jnp.exp(jnp.minimum(self.logit_scale.value, jnp.log(1.0 / 0.01)))
            attn = attn * logit_scale
        else:
            attn = (q * self.scale) @ k.swapaxes(-2, -1)

        if self.rel_pos_bias is not None:
            bias = self.relative_position_bias_table.value[self.relative_position_index.value]
            attn = attn + bias.transpose(2, 0, 1)[None]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + mask[None, :, None]
            attn = attn.reshape(-1, self.num_heads, N, N)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class PatchMerging(nnx.Module):
    def __init__(self, dim, dim_scale=2, *, rngs):
        self.reduction = nnx.Linear(4 * dim, dim_scale * dim, use_bias=False,
                                    kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.norm = nnx.LayerNorm(4 * dim, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        assert N % 4 == 0, "x size %d is not divisible by 4 as necessary for patching." % N
        x = jnp.concatenate([x[:, 0::4], x[:, 1::4], x[:, 2::4], x[:, 3::4]], axis=-1)
        return self.reduction(self.norm(x))


class PatchExpand(nnx.Module):
    def __init__(self, dim, dim_scale=2, *, rngs):
        self.expand = (nnx.Linear(dim, dim_scale * dim, use_bias=False,
                                  kernel_init=TRUNC_NORMAL, rngs=rngs)
                       if dim_scale != 1 else Identity())
        self.norm = nnx.LayerNorm(dim * dim_scale // 4, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        x = self.expand(x)
        C = x.shape[-1]
        x = rearrange(x, "b n (p c) -> b (n p) c", p=4, c=C // 4)
        return self.norm(x)


class FinalPatchExpand_X4(nnx.Module):
    def __init__(self, patch_size, dim, *, rngs):
        self.patch_size = patch_size
        self.expand = nnx.Linear(dim, patch_size * dim, use_bias=False,
                                 kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        x = self.expand(x)
        C = x.shape[-1]
        x = rearrange(x, "b n (p c) -> b (n p) c", p=self.patch_size, c=C // self.patch_size)
        return self.norm(x)
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_parity_modules.py -v` → all PASS. If attention diverges, compare against `int/`-free leaf goldens step by step: check qkv reshape order first, then softmax mask broadcast (`mask[None, :, None]` must give `(1, nW, 1, N, N)`).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: HP WindowAttention + patch merge/expand, leaf parity green"
```

---

### Task 13: HP `SwinTransformerBlock`, `BasicLayer`, `BasicLayer_up`

**Files:**
- Modify: `src/heal_swin_nnx/swin_hp_transformer.py`
- Test: `tests/test_parity_modules.py` (append)

**Interfaces:**
- Produces: `SwinTransformerBlock(dim, input_resolution, base_pix, num_heads, window_size=4, shift_size=0, shift_strategy="nest_roll", rel_pos_bias=None, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, use_v2_norm_placement=False, use_cos_attn=False, *, rngs)` with attribute `.shifter` (a shifting module carrying `.attn_mask`); `BasicLayer(...)`/`BasicLayer_up(...)` with the reference's argument lists (downsample/upsample as bool flags) and `.blocks` lists. Consumed by Task 14.

- [ ] **Step 1: Append failing tests**

```python
def test_hp_block_parity_all_shifters():
    for case in ("leaf_hp_block_noshift", "leaf_hp_block_nestroll",
                 "leaf_hp_block_grid", "leaf_hp_block_ring"):
        npz, meta = load_case(case)
        m = hp.SwinTransformerBlock(
            dim=meta["dim"], input_resolution=meta["input_resolution"],
            base_pix=meta["base_pix"], num_heads=meta["num_heads"],
            window_size=meta["window_size"], shift_size=meta["shift_size"],
            shift_strategy=meta.get("shift_strategy", "nest_roll"), rngs=nnx.Rngs(0))
        load_torch_state(m, state_dict_of(npz))
        m.eval()
        x = jnp.asarray(npz["input"])
        np.testing.assert_allclose(np.asarray(m(x)), npz["output"], err_msg=case, **FWD)
        gx = jax.grad(lambda x: m(x).sum())(x)
        np.testing.assert_allclose(np.asarray(gx), npz["input_grad"], err_msg=case, **GRD)
        gp = nnx.grad(lambda m: m(x).sum())(m)
        check_param_grads(gp, grads_of(npz))
        if "sd/attn_mask" in npz.files:  # block-level buffer parity, bit-exact
            assert np.array_equal(np.asarray(m.shifter.attn_mask.value), npz["sd/attn_mask"])
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_parity_modules.py -v -k block` → FAIL.

- [ ] **Step 3: Implement**

Append to `swin_hp_transformer.py`:

```python
import math


class SwinTransformerBlock(nnx.Module):
    def __init__(self, dim, input_resolution, base_pix, num_heads, window_size=4, shift_size=0,
                 shift_strategy="nest_roll", rel_pos_bias=None, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 use_v2_norm_placement=False, use_cos_attn=False, *, rngs):
        self.input_resolution = input_resolution
        self.use_v2_norm_placement = use_v2_norm_placement
        self.window_size = window_size
        self.shift_size = shift_size
        if self.input_resolution <= self.window_size:
            self.shift_size = 0
            self.window_size = self.input_resolution

        self.norm1 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads,
                                    rel_pos_bias=rel_pos_bias, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                                    use_cos_attn=use_cos_attn, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop, rngs=rngs)

        nside = math.sqrt(input_resolution // base_pix)
        assert nside % 1 == 0, "nside has to be an integer in every layer"
        nside = int(nside)

        if self.shift_size > 0:
            if shift_strategy == "nest_roll":
                self.shifter = hp_shifting.NestRollShift(
                    shift_size=self.shift_size, input_resolution=self.input_resolution,
                    window_size=self.window_size)
            elif shift_strategy == "nest_grid_shift":
                self.shifter = hp_shifting.NestGridShift(
                    nside=nside, base_pix=base_pix, window_size=self.window_size)
            elif shift_strategy == "ring_shift":
                self.shifter = hp_shifting.RingShift(
                    nside=nside, base_pix=base_pix, window_size=self.window_size,
                    shift_size=self.shift_size)
            else:
                raise ValueError("unknown shift_strategy %r" % shift_strategy)
        else:
            self.shifter = hp_shifting.NoShift()

    def __call__(self, x):
        shortcut = x
        if not self.use_v2_norm_placement:
            x = self.norm1(x)

        shifted_x = self.shifter.shift(x)
        x_windows = window_partition(shifted_x, self.window_size)
        mask = None if self.shifter.attn_mask is None else self.shifter.attn_mask.value
        attn_windows = self.attn(x_windows, mask=mask)
        shifted_x = window_reverse(attn_windows, self.window_size, self.input_resolution)
        x = self.shifter.shift_back(shifted_x)

        if self.use_v2_norm_placement:
            x = shortcut + self.drop_path(self.norm1(x))
            x = x + self.drop_path(self.norm2(self.mlp(x)))
        else:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _make_blocks(dim, input_resolution, base_pix, depth, num_heads, window_size, shift_size,
                 shift_strategy, rel_pos_bias, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                 drop_path, use_v2_norm_placement, use_cos_attn, rngs):
    return [SwinTransformerBlock(
        dim=dim, input_resolution=input_resolution, base_pix=base_pix, num_heads=num_heads,
        window_size=window_size, shift_size=0 if (i % 2 == 0) else shift_size,
        shift_strategy=shift_strategy, rel_pos_bias=rel_pos_bias, mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        use_v2_norm_placement=use_v2_norm_placement, use_cos_attn=use_cos_attn, rngs=rngs)
        for i in range(depth)]


class BasicLayer(nnx.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, base_pix,
                 shift_size, shift_strategy, rel_pos_bias, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, downsample=False,
                 use_checkpoint=False, use_v2_norm_placement=False, use_cos_attn=False, *, rngs):
        self.use_checkpoint = use_checkpoint
        self.blocks = _make_blocks(dim, input_resolution, base_pix, depth, num_heads,
                                   window_size, shift_size, shift_strategy, rel_pos_bias,
                                   mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path,
                                   use_v2_norm_placement, use_cos_attn, rngs)
        self.downsample = PatchMerging(dim=dim, rngs=rngs) if downsample else None

    def __call__(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = nnx.remat(type(blk).__call__)(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayer_up(nnx.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, base_pix,
                 shift_size, shift_strategy, rel_pos_bias, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, upsample=False,
                 use_checkpoint=False, use_v2_norm_placement=False, use_cos_attn=False, *, rngs):
        self.use_checkpoint = use_checkpoint
        self.blocks = _make_blocks(dim, input_resolution, base_pix, depth, num_heads,
                                   window_size, shift_size, shift_strategy, rel_pos_bias,
                                   mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path,
                                   use_v2_norm_placement, use_cos_attn, rngs)
        self.upsample = PatchExpand(dim=dim, dim_scale=2, rngs=rngs) if upsample else None

    def __call__(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = nnx.remat(type(blk).__call__)(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_parity_modules.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: HP transformer block + basic layers, all-shifter block parity green"
```

---

### Task 14: HP `PatchEmbed`, `SwinHPEncoder`, `HPUnetDecoder`, `SwinHPTransformerSys` + forward parity

**Files:**
- Modify: `src/heal_swin_nnx/swin_hp_transformer.py`
- Test: `tests/test_parity_modules.py` (PatchEmbed leaf), `tests/test_parity_e2e.py` (new)

**Interfaces:**
- Produces: `PatchEmbed(config, data_spec, *, rngs)` — input `(B, N, f_in)`; `SwinHPEncoder(config, data_spec, *, rngs)` — `__call__(x) -> (tokens, skips)` where `skips` is the list of per-stage inputs (reference `x_downsample`); `HPUnetDecoder(config, data_spec, *, rngs)` — `__call__(tokens, skips, return_intermediates=False)`; `SwinHPTransformerSys(config, data_spec, *, rngs)` — `__call__(x) -> (B, N, f_out)` with attributes `.encoder`, `.decoder`. Consumed by Tasks 15, 18.
- Consumes: `HP_PREFIX_MAP`, `load_torch_state` (Task 11); golden cases `hp_*` (Task 5).

- [ ] **Step 1: Append PatchEmbed leaf test + write e2e forward test**

Append to `tests/test_parity_modules.py`:

```python
from heal_swin_nnx.config import DataSpec, SwinHPTransformerConfig


def test_hp_patch_embed_parity():
    npz, meta = load_case("leaf_hp_patch_embed")
    cfg = SwinHPTransformerConfig(patch_size=meta["patch_size"], embed_dim=meta["embed_dim"],
                                  depths=[2, 2], num_heads=[2, 4], drop_path_rate=0.0)
    ds = DataSpec(dim_in=meta["dim_in"], f_in=meta["f_in"], f_out=5, base_pix=8)
    m = hp.PatchEmbed(cfg, ds, rngs=nnx.Rngs(0))
    load_torch_state(m, state_dict_of(npz))
    m.eval()
    x = jnp.asarray(npz["input"]).transpose(0, 2, 1)  # torch (B,C,N) -> nnx (B,N,C)
    np.testing.assert_allclose(np.asarray(m(x)), npz["output"], **FWD)
```

Create `tests/test_parity_e2e.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from heal_swin_nnx import swin_hp_transformer as hp
from heal_swin_nnx.config import DataSpec, SwinHPTransformerConfig
from heal_swin_nnx.weight_transfer import HP_PREFIX_MAP, load_torch_state
from tests.parity_utils import grads_of, load_case, state_dict_of

E2E_FWD = dict(rtol=1e-4, atol=1e-4)
E2E_GRD = dict(rtol=1e-3, atol=1e-4)

HP_CASES = ["hp_base", "hp_grid", "hp_ring", "hp_cos_v2", "hp_relbias", "hp_ape"]


def build_hp_model(meta):
    cfg = SwinHPTransformerConfig(embed_dim=meta["embed_dim"], depths=meta["depths"],
                                  num_heads=meta["num_heads"], drop_path_rate=0.0,
                                  **meta["overrides"])
    ds = DataSpec(**meta["data_spec"])
    model = hp.SwinHPTransformerSys(cfg, ds, rngs=nnx.Rngs(0))
    return model


@pytest.mark.parametrize("case", HP_CASES)
def test_hp_e2e_forward_parity(case):
    npz, meta = load_case(case)
    model = build_hp_model(meta)
    load_torch_state(model, state_dict_of(npz), prefix_map=HP_PREFIX_MAP)
    model.eval()
    x = jnp.asarray(npz["input"]).transpose(0, 2, 1)          # (B,C,N) -> (B,N,C)
    y = np.asarray(model(x)).transpose(0, 2, 1)               # back to torch layout
    np.testing.assert_allclose(y, npz["output"], **E2E_FWD)


@pytest.mark.parametrize("case", HP_CASES)
def test_hp_encoder_boundary_parity(case):
    npz, meta = load_case(case)
    model = build_hp_model(meta)
    load_torch_state(model, state_dict_of(npz), prefix_map=HP_PREFIX_MAP)
    model.eval()
    x = jnp.asarray(npz["input"]).transpose(0, 2, 1)
    tokens, skips = model.encoder(x)
    np.testing.assert_allclose(np.asarray(tokens), npz["int/enc_norm"], **E2E_FWD)
    np.testing.assert_allclose(np.asarray(skips[1]), npz["int/enc_layer_0"], **E2E_FWD)
    inters = model.decoder(tokens, skips, return_intermediates=True)[1]
    for i, inter in enumerate(inters):
        np.testing.assert_allclose(np.asarray(inter), npz["int/dec_layer_up_%d" % i],
                                   err_msg="dec_layer_up_%d" % i, **E2E_FWD)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_parity_e2e.py -v` → FAIL (classes missing).

- [ ] **Step 3: Implement**

Append to `swin_hp_transformer.py`:

```python
class PatchEmbed(nnx.Module):
    def __init__(self, config, data_spec, *, rngs):
        assert config.patch_size % 4 == 0, "required for valid nside in deeper layers"
        self.dim_in = data_spec.dim_in
        self.num_patches = data_spec.dim_in // config.patch_size
        self.proj = nnx.Conv(data_spec.f_in, config.embed_dim,
                             kernel_size=(config.patch_size,), strides=(config.patch_size,),
                             padding="VALID", rngs=rngs)
        self.norm = (nnx.LayerNorm(config.embed_dim, epsilon=LN_EPS, rngs=rngs)
                     if config.patch_embed_norm_layer == "layernorm" else None)

    def __call__(self, x):  # (B, N, f_in) channels-last
        assert x.shape[1] == self.dim_in, (
            "Input image size (%d) doesn't match model (%d)." % (x.shape[1], self.dim_in))
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinHPEncoder(nnx.Module):
    """Compression-only backbone: patch embed + APE + encoder stages + final norm.
    Standalone-usable (tokenizer / embedder); allocates no decoder parameters."""

    def __init__(self, config: SwinHPTransformerConfig, data_spec: DataSpec, *, rngs):
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(config, data_spec, rngs=rngs)
        num_patches = self.patch_embed.num_patches
        if config.ape:
            self.absolute_pos_embed = nnx.Param(
                TRUNC_NORMAL(rngs.params(), (1, num_patches, config.embed_dim)))
        else:
            self.absolute_pos_embed = None
        self.pos_drop = nnx.Dropout(config.drop_rate, rngs=rngs)

        dpr = [float(v) for v in np.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.layers = []
        for i_layer in range(self.num_layers):
            self.layers.append(BasicLayer(
                dim=int(config.embed_dim * 2 ** i_layer),
                input_resolution=num_patches // (4 ** i_layer),
                depth=config.depths[i_layer], num_heads=config.num_heads[i_layer],
                window_size=config.window_size, base_pix=data_spec.base_pix,
                shift_size=config.shift_size, shift_strategy=config.shift_strategy,
                rel_pos_bias=config.rel_pos_bias, mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                use_cos_attn=config.use_cos_attn, drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])],
                use_v2_norm_placement=config.use_v2_norm_placement,
                downsample=i_layer < self.num_layers - 1,
                use_checkpoint=config.use_checkpoint, rngs=rngs))
        self.norm = nnx.LayerNorm(self.num_features, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed.value
        x = self.pos_drop(x)
        skips = []
        for layer in self.layers:
            skips.append(x)
            x = layer(x)
        return self.norm(x), skips


class HPUnetDecoder(nnx.Module):
    """UNet decoder head (dense per-pixel outputs). Named UnetDecoder in the reference."""

    def __init__(self, config: SwinHPTransformerConfig, data_spec: DataSpec, *, rngs):
        self.num_layers = len(config.depths)
        num_patches = data_spec.dim_in // config.patch_size
        dpr = [float(v) for v in np.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.layers_up = []
        self.concat_back_dim = []
        for i_layer in range(self.num_layers):
            down_idx = self.num_layers - 1 - i_layer
            concat_out = int(config.embed_dim * 2 ** down_idx)
            self.concat_back_dim.append(
                nnx.Linear(2 * concat_out, concat_out, kernel_init=TRUNC_NORMAL, rngs=rngs)
                if i_layer > 0 else Identity())
            if i_layer == 0:
                self.layers_up.append(PatchExpand(dim=concat_out, dim_scale=2, rngs=rngs))
            else:
                self.layers_up.append(BasicLayer_up(
                    dim=concat_out, input_resolution=num_patches // (4 ** down_idx),
                    depth=config.depths[down_idx], num_heads=config.num_heads[down_idx],
                    window_size=config.window_size, base_pix=data_spec.base_pix,
                    shift_size=config.shift_size, shift_strategy=config.shift_strategy,
                    rel_pos_bias=config.rel_pos_bias, mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                    use_cos_attn=config.use_cos_attn, drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path=dpr[sum(config.depths[:down_idx]):sum(config.depths[:down_idx + 1])],
                    use_v2_norm_placement=config.use_v2_norm_placement,
                    upsample=down_idx > 0, use_checkpoint=config.use_checkpoint, rngs=rngs))
        self.up = FinalPatchExpand_X4(patch_size=config.patch_size, dim=config.embed_dim,
                                      rngs=rngs)
        self.output = nnx.Conv(config.embed_dim, data_spec.f_out, kernel_size=(1,),
                               use_bias=False, rngs=rngs)
        self.norm_up = nnx.LayerNorm(config.embed_dim, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x, skips, return_intermediates=False):
        intermediates = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = jnp.concatenate([x, skips[self.num_layers - 1 - inx]], axis=-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
            if return_intermediates:
                intermediates.append(x)
        x = self.norm_up(x)
        x = self.up(x)
        x = self.output(x)  # (B, N, f_out) channels-last
        return (x, intermediates) if return_intermediates else x


class SwinHPTransformerSys(nnx.Module):
    def __init__(self, config: SwinHPTransformerConfig, data_spec: DataSpec, *, rngs):
        self.encoder = SwinHPEncoder(config, data_spec, rngs=rngs)
        self.decoder = HPUnetDecoder(config, data_spec, rngs=rngs)

    def __call__(self, x):
        tokens, skips = self.encoder(x)
        return self.decoder(tokens, skips)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_parity_modules.py tests/test_parity_e2e.py -v`
Expected: all PASS. Debugging order on failure: PatchEmbed leaf → `enc_layer_0` intermediate → `enc_norm` → decoder intermediates (the failing boundary names the guilty module; its leaf/block parity already passed, so suspect wiring: skip indexing, concat order, dpr slicing).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: SwinHPEncoder/HPUnetDecoder/SwinHPTransformerSys, e2e forward parity green"
```

---

### Task 15: HP gradient parity + buffer parity

**Files:**
- Test: `tests/test_parity_e2e.py` (append)

**Interfaces:**
- Consumes: everything from Task 14. No new production code expected (this task verifies training-equivalence; fixes go into existing modules if it fails).

- [ ] **Step 1: Append the tests**

```python
from heal_swin_nnx.weight_transfer import torch_key_to_path, transform_array


@pytest.mark.parametrize("case", HP_CASES)
def test_hp_e2e_gradient_parity(case):
    npz, meta = load_case(case)
    model = build_hp_model(meta)
    load_torch_state(model, state_dict_of(npz), prefix_map=HP_PREFIX_MAP)
    model.eval()
    x = jnp.asarray(npz["input"]).transpose(0, 2, 1)

    gx = jax.grad(lambda x: model(x).sum())(x)
    np.testing.assert_allclose(np.asarray(gx).transpose(0, 2, 1), npz["input_grad"], **E2E_GRD)

    gp = nnx.grad(lambda m: m(x).sum())(model)
    flat = {tuple(str(p) for p in path): v for path, v in gp.flat_state()}
    for tkey, tgrad in grads_of(npz).items():
        path = tuple(str(p) for p in torch_key_to_path(tkey, HP_PREFIX_MAP))
        expected = transform_array(tgrad, path[-1])
        np.testing.assert_allclose(np.asarray(flat[path].value), expected,
                                   err_msg=tkey, **E2E_GRD)


@pytest.mark.parametrize("case", ["hp_base", "hp_grid", "hp_ring"])
def test_hp_buffer_bit_parity(case):
    """Torch buffers (attn_mask, relative_position_index) match ours exactly."""
    npz, meta = load_case(case)
    model = build_hp_model(meta)
    for key in state_dict_of(npz):
        if not key.endswith(("attn_mask", "relative_position_index")):
            continue
        obj = model
        parts = torch_key_to_path(key, HP_PREFIX_MAP)
        for p in parts[:-1]:
            obj = obj[p] if isinstance(p, int) else getattr(obj, p)
        leaf = parts[-1]
        ours = (obj.shifter.attn_mask if leaf == "attn_mask" else
                getattr(obj, leaf))
        assert np.array_equal(np.asarray(ours.value), npz["sd/" + key]), key
```

- [ ] **Step 2: Run** — `uv run pytest tests/test_parity_e2e.py -v` — Expected: all PASS. If a gradient mismatches while forward passes, suspect: `stop_gradient`-like issues from Buffer misuse (index gathers are fine), or float accumulation order — check whether the same param's grad matches at leaf level (it did in Tasks 11–13), then bisect with the block-boundary intermediates by comparing `jax.grad` of partial compositions.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test: HP end-to-end gradient + buffer parity green"
```

---

### Task 16: Flat model modules (`swin_transformer.py`) + leaf parity

**Files:**
- Create: `src/heal_swin_nnx/swin_transformer.py`
- Test: `tests/test_parity_modules.py` (append)

**Interfaces:**
- Produces (in `heal_swin_nnx.swin_transformer`): `window_partition(x, window_size) -> (nW*B, wh, ww, C)`, `window_reverse(windows, window_size, H, W)`; `WindowAttention(dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_cos_attn=False, use_rel_pos_bias=True, *, rngs)`; `SwinTransformerBlock(dim, input_resolution, num_heads, window_size=(4,4), shift_size=(0,0), mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, use_masking=True, use_cos_attn=False, use_v2_norm_placement=False, use_rel_pos_bias=True, *, rngs)` with `.attn_mask: Buffer | None`; `PatchMerging(input_resolution, dim, *, rngs)`; `PatchExpand(input_resolution, dim, dim_scale=2, *, rngs)`; `FinalPatchExpand_X4(input_resolution, patch_size, dim, *, rngs)`; `PatchEmbed(config, data_spec, *, rngs)`. Consumed by Task 17.

- [ ] **Step 1: Append failing tests**

```python
from heal_swin_nnx import swin_transformer as flat
from heal_swin_nnx.config import SwinTransformerConfig


def test_flat_window_attention_parity():
    for case in ("leaf_flat_attn", "leaf_flat_attn_norelbias", "leaf_flat_attn_cos"):
        npz, meta = load_case(case)
        m = flat.WindowAttention(dim=meta["dim"], window_size=tuple(meta["window_size"]),
                                 num_heads=meta["num_heads"],
                                 use_rel_pos_bias=meta.get("use_rel_pos_bias", True),
                                 use_cos_attn=meta.get("use_cos_attn", False),
                                 rngs=nnx.Rngs(0))
        load_torch_state(m, state_dict_of(npz))
        _leaf_forward_and_grads(m, npz)


def test_flat_merge_expand_parity():
    npz, meta = load_case("leaf_flat_patch_merging")
    m = flat.PatchMerging(input_resolution=tuple(meta["input_resolution"]), dim=meta["dim"],
                          rngs=nnx.Rngs(0))
    load_torch_state(m, state_dict_of(npz))
    _leaf_forward_and_grads(m, npz)

    npz, meta = load_case("leaf_flat_patch_expand")
    m = flat.PatchExpand(input_resolution=tuple(meta["input_resolution"]), dim=meta["dim"],
                         rngs=nnx.Rngs(0))
    load_torch_state(m, state_dict_of(npz))
    _leaf_forward_and_grads(m, npz)

    npz, meta = load_case("leaf_flat_final_expand")
    m = flat.FinalPatchExpand_X4(input_resolution=tuple(meta["input_resolution"]),
                                 patch_size=tuple(meta["patch_size"]), dim=meta["dim"],
                                 rngs=nnx.Rngs(0))
    load_torch_state(m, state_dict_of(npz))
    _leaf_forward_and_grads(m, npz)


def test_flat_block_parity():
    for case in ("leaf_flat_block_noshift", "leaf_flat_block_shift", "leaf_flat_block_nomask"):
        npz, meta = load_case(case)
        m = flat.SwinTransformerBlock(
            dim=meta["dim"], input_resolution=tuple(meta["input_resolution"]),
            num_heads=meta["num_heads"], window_size=tuple(meta["window_size"]),
            shift_size=tuple(meta["shift_size"]), use_masking=meta.get("use_masking", True),
            rngs=nnx.Rngs(0))
        load_torch_state(m, state_dict_of(npz))
        m.eval()
        x = jnp.asarray(npz["input"])
        np.testing.assert_allclose(np.asarray(m(x)), npz["output"], err_msg=case, **FWD)
        gp = nnx.grad(lambda m: m(x).sum())(m)
        check_param_grads(gp, grads_of(npz))


def test_flat_patch_embed_parity():
    npz, meta = load_case("leaf_flat_patch_embed")
    cfg = SwinTransformerConfig(patch_size=tuple(meta["patch_size"]),
                                embed_dim=meta["embed_dim"], depths=[2, 2],
                                num_heads=[2, 4], drop_path_rate=0.0)
    ds = DataSpec(dim_in=tuple(meta["dim_in"]), f_in=meta["f_in"], f_out=5)
    m = flat.PatchEmbed(cfg, ds, rngs=nnx.Rngs(0))
    load_torch_state(m, state_dict_of(npz))
    m.eval()
    x = jnp.asarray(npz["input"]).transpose(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)
    np.testing.assert_allclose(np.asarray(m(x)), npz["output"], **FWD)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_parity_modules.py -v -k flat` → FAIL.

- [ ] **Step 3: Implement**

`src/heal_swin_nnx/swin_transformer.py` (module section — encoder/decoder land in Task 17):

```python
"""Flat 2D SWIN-UNet baseline. Port of models_torch/swin_transformer.py."""
import numpy as np

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx

from heal_swin_nnx.config import DataSpec, SwinTransformerConfig
from heal_swin_nnx.layers import TRUNC_NORMAL, DropPath, Identity, Mlp
from heal_swin_nnx.variables import Buffer

LN_EPS = 1e-5


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    return x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size[0], window_size[1], C)


def window_reverse(windows, window_size, H, W):
    B = windows.shape[0] // ((H // window_size[0]) * (W // window_size[1]))
    x = windows.reshape(B, H // window_size[0], W // window_size[1],
                        window_size[0], window_size[1], -1)
    return x.transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, x.shape[-1])


def flat_relative_position_index(window_size):
    coords = np.stack(np.meshgrid(np.arange(window_size[0]), np.arange(window_size[1]),
                                  indexing="ij"))
    flat = coords.reshape(2, -1)
    rel = flat[:, :, None] - flat[:, None, :]
    rel = rel.transpose(1, 2, 0).astype(np.int64)
    rel[:, :, 0] += window_size[0] - 1
    rel[:, :, 1] += window_size[1] - 1
    rel[:, :, 0] *= 2 * window_size[1] - 1
    return rel.sum(-1)


class WindowAttention(nnx.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, use_cos_attn=False, use_rel_pos_bias=True,
                 *, rngs):
        self.dim = dim
        self.window_size = tuple(window_size)
        self.num_heads = num_heads
        self.use_cos_attn = use_cos_attn
        self.use_rel_pos_bias = use_rel_pos_bias
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if use_cos_attn:
            self.logit_scale = nnx.Param(jnp.log(10.0 * jnp.ones((num_heads, 1, 1))))
        # table always exists in the reference, even with use_rel_pos_bias=False
        n_rel = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        self.relative_position_bias_table = nnx.Param(
            TRUNC_NORMAL(rngs.params(), (n_rel, num_heads)))
        self.relative_position_index = Buffer(
            jnp.asarray(flat_relative_position_index(self.window_size)))

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.attn_drop = nnx.Dropout(attn_drop, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)

    def __call__(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_cos_attn:
            qn = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
            kn = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-12)
            attn = qn @ kn.swapaxes(-2, -1)
            logit_scale = jnp.exp(jnp.minimum(self.logit_scale.value, jnp.log(1.0 / 0.01)))
            attn = attn * logit_scale
        else:
            attn = (q * self.scale) @ k.swapaxes(-2, -1)

        if self.use_rel_pos_bias:
            ws_area = self.window_size[0] * self.window_size[1]
            bias = self.relative_position_bias_table.value[
                self.relative_position_index.value.reshape(-1)].reshape(ws_area, ws_area, -1)
            attn = attn + bias.transpose(2, 0, 1)[None]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + mask[None, :, None]
            attn = attn.reshape(-1, self.num_heads, N, N)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).swapaxes(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


def flat_shift_mask(input_resolution, window_size, shift_size):
    H, W = input_resolution
    img_mask = np.zeros((1, H, W, 1), dtype=np.float32)
    h_slices = (slice(0, -window_size[0]), slice(-window_size[0], -shift_size[0]),
                slice(-shift_size[0], None))
    w_slices = (slice(0, -window_size[1]), slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mw = img_mask.reshape(1, H // window_size[0], window_size[0],
                          W // window_size[1], window_size[1], 1)
    mw = mw.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size[0] * window_size[1])
    attn_mask = mw[:, None, :] - mw[:, :, None]
    return np.where(attn_mask != 0, np.float32(-100.0), np.float32(0.0))


class SwinTransformerBlock(nnx.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=(4, 4), shift_size=(0, 0),
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, use_masking=True, use_cos_attn=False,
                 use_v2_norm_placement=False, use_rel_pos_bias=True, *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.window_size = tuple(window_size)
        self.shift_size = tuple(shift_size)
        self.use_v2_norm_placement = use_v2_norm_placement
        if (self.input_resolution[0] <= self.window_size[0]
                or self.input_resolution[1] <= self.window_size[1]):
            self.shift_size = (0, 0)
            self.window_size = self.input_resolution
        assert 0 <= self.shift_size[0] < self.window_size[0]
        assert 0 <= self.shift_size[1] < self.window_size[1]

        self.norm1 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                    proj_drop=drop, use_rel_pos_bias=use_rel_pos_bias,
                                    use_cos_attn=use_cos_attn, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop, rngs=rngs)

        if use_masking and (self.shift_size[0] > 0 or self.shift_size[1] > 0):
            self.attn_mask = Buffer(jnp.asarray(flat_shift_mask(
                self.input_resolution, self.window_size, self.shift_size)))
        else:
            self.attn_mask = None

    def __call__(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        if not self.use_v2_norm_placement:
            x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            # bug-for-bug with reference (swin_transformer.py:366-368): shift_size[0] twice
            shifted_x = jnp.roll(x, (-self.shift_size[0], -self.shift_size[0]), axis=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size[0] * self.window_size[1], C)
        mask = None if self.attn_mask is None else self.attn_mask.value
        attn_windows = self.attn(x_windows, mask=mask)
        attn_windows = attn_windows.reshape(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x = jnp.roll(shifted_x, (self.shift_size[0], self.shift_size[1]), axis=(1, 2))
        else:
            x = shifted_x
        x = x.reshape(B, H * W, C)

        if self.use_v2_norm_placement:
            x = shortcut + self.drop_path(self.norm1(x))
            x = x + self.drop_path(self.norm2(self.mlp(x)))
        else:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nnx.Module):
    def __init__(self, input_resolution, dim, *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.patch_size = 4
        self.reduction = nnx.Linear(self.patch_size * dim, 2 * dim, use_bias=False,
                                    kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.norm = nnx.LayerNorm(self.patch_size * dim, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W and H % 2 == 0 and W % 2 == 0
        x = x.reshape(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = jnp.concatenate([x0, x1, x2, x3], axis=-1).reshape(B, -1, self.patch_size * C)
        return self.reduction(self.norm(x))


class PatchExpand(nnx.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.expand = (nnx.Linear(dim, 2 * dim, use_bias=False, kernel_init=TRUNC_NORMAL,
                                  rngs=rngs) if dim_scale == 2 else Identity())
        self.norm = nnx.LayerNorm(dim // dim_scale, epsilon=LN_EPS, rngs=rngs)
        self.dim_scale = 4

    def __call__(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.reshape(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2,
                      c=C // self.dim_scale)
        return self.norm(x.reshape(B, -1, C // self.dim_scale))


class FinalPatchExpand_X4(nnx.Module):
    def __init__(self, input_resolution, patch_size, dim, *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.patch_size = tuple(patch_size)
        self.output_dim = dim
        self.expand = nnx.Linear(dim, self.patch_size[0] * self.patch_size[1] * dim,
                                 use_bias=False, kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.reshape(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c",
                      p1=self.patch_size[0], p2=self.patch_size[1],
                      c=C // (self.patch_size[0] * self.patch_size[1]))
        return self.norm(x.reshape(B, -1, self.output_dim))


class PatchEmbed(nnx.Module):
    def __init__(self, config, data_spec, *, rngs):
        self.dim_in = tuple(data_spec.dim_in)
        self.patches_resolution = (data_spec.dim_in[0] // config.patch_size[0],
                                   data_spec.dim_in[1] // config.patch_size[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.proj = nnx.Conv(data_spec.f_in, config.embed_dim,
                             kernel_size=tuple(config.patch_size),
                             strides=tuple(config.patch_size), padding="VALID", rngs=rngs)
        self.norm = (nnx.LayerNorm(config.embed_dim, epsilon=LN_EPS, rngs=rngs)
                     if config.patch_embed_norm_layer == "layernorm" else None)

    def __call__(self, x):  # (B, H, W, f_in) channels-last
        B, H, W, C = x.shape
        assert (H, W) == self.dim_in
        x = self.proj(x)                   # (B, Ph, Pw, embed_dim)
        x = x.reshape(B, -1, x.shape[-1])  # (B, Ph*Pw, embed_dim); row-major == torch flatten(2)
        if self.norm is not None:
            x = self.norm(x)
        return x
```

(Layout note: torch does `(B, C, Ph, Pw) -> flatten(2) -> transpose -> (B, Ph*Pw, C)`; both paths enumerate `(ph, pw)` row-major, so the plain reshape above matches exactly.)

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_parity_modules.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: flat SWIN modules, leaf parity green"
```

---

### Task 17: Flat `SwinEncoder`, `UnetDecoder`, `SwinTransformerSys` + e2e parity

**Files:**
- Modify: `src/heal_swin_nnx/swin_transformer.py`
- Test: `tests/test_parity_e2e.py` (append)

**Interfaces:**
- Produces: `BasicLayer`/`BasicLayer_up` (flat), `SwinEncoder(config, data_spec, *, rngs) -> (tokens, skips)`, `UnetDecoder(config, data_spec, *, rngs)` with `__call__(tokens, skips, return_intermediates=False)` returning `(B, H, W, f_out)`, `SwinTransformerSys(config, data_spec, *, rngs)` with `.encoder`/`.decoder`.
- Consumes: `FLAT_PREFIX_MAP` (Task 11), golden cases `flat_*` (Task 5).

- [ ] **Step 1: Append failing e2e tests**

```python
from heal_swin_nnx import swin_transformer as flat
from heal_swin_nnx.config import SwinTransformerConfig
from heal_swin_nnx.weight_transfer import FLAT_PREFIX_MAP

FLAT_CASES = ["flat_base", "flat_cos_v2", "flat_norelbias", "flat_nomask", "flat_ape"]


def build_flat_model(meta):
    cfg = SwinTransformerConfig(embed_dim=meta["embed_dim"], depths=meta["depths"],
                                num_heads=meta["num_heads"], drop_path_rate=0.0,
                                **meta["overrides"])
    ds = DataSpec(dim_in=tuple(meta["data_spec"]["dim_in"]), f_in=meta["data_spec"]["f_in"],
                  f_out=meta["data_spec"]["f_out"])
    return flat.SwinTransformerSys(cfg, ds, rngs=nnx.Rngs(0))


@pytest.mark.parametrize("case", FLAT_CASES)
def test_flat_e2e_forward_parity(case):
    npz, meta = load_case(case)
    model = build_flat_model(meta)
    load_torch_state(model, state_dict_of(npz), prefix_map=FLAT_PREFIX_MAP)
    model.eval()
    x = jnp.asarray(npz["input"]).transpose(0, 2, 3, 1)   # (B,C,H,W) -> (B,H,W,C)
    y = np.asarray(model(x)).transpose(0, 3, 1, 2)        # back to torch layout
    np.testing.assert_allclose(y, npz["output"], **E2E_FWD)


@pytest.mark.parametrize("case", FLAT_CASES)
def test_flat_e2e_gradient_parity(case):
    npz, meta = load_case(case)
    model = build_flat_model(meta)
    load_torch_state(model, state_dict_of(npz), prefix_map=FLAT_PREFIX_MAP)
    model.eval()
    x = jnp.asarray(npz["input"]).transpose(0, 2, 3, 1)
    gx = jax.grad(lambda x: model(x).sum())(x)
    np.testing.assert_allclose(np.asarray(gx).transpose(0, 3, 1, 2), npz["input_grad"],
                               **E2E_GRD)
    gp = nnx.grad(lambda m: m(x).sum())(model)
    flat_g = {tuple(str(p) for p in path): v for path, v in gp.flat_state()}
    for tkey, tgrad in grads_of(npz).items():
        path = tuple(str(p) for p in torch_key_to_path(tkey, FLAT_PREFIX_MAP))
        expected = transform_array(tgrad, path[-1])
        np.testing.assert_allclose(np.asarray(flat_g[path].value), expected,
                                   err_msg=tkey, **E2E_GRD)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_parity_e2e.py -v -k flat` → FAIL.

- [ ] **Step 3: Implement**

Append to `swin_transformer.py` (flat `BasicLayer`/`BasicLayer_up` follow the same pattern as HP but with 2D resolutions — build blocks with `shift_size=(0,0) if i % 2 == 0 else shift_size`, downsample `PatchMerging(input_resolution, dim)`, upsample `PatchExpand(input_resolution, dim, dim_scale=2)`):

```python
def _make_flat_blocks(dim, input_resolution, depth, num_heads, window_size, shift_size,
                      mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, use_masking,
                      use_cos_attn, use_v2_norm_placement, use_rel_pos_bias, rngs):
    return [SwinTransformerBlock(
        dim=dim, input_resolution=input_resolution, num_heads=num_heads,
        window_size=window_size, shift_size=(0, 0) if (i % 2 == 0) else shift_size,
        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
        attn_drop=attn_drop,
        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        use_masking=use_masking, use_cos_attn=use_cos_attn,
        use_v2_norm_placement=use_v2_norm_placement, use_rel_pos_bias=use_rel_pos_bias,
        rngs=rngs) for i in range(depth)]


class BasicLayer(nnx.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, shift_size,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, downsample=False, use_checkpoint=False, use_masking=True,
                 use_cos_attn=False, use_v2_norm_placement=False, use_rel_pos_bias=True,
                 *, rngs):
        self.use_checkpoint = use_checkpoint
        self.blocks = _make_flat_blocks(dim, input_resolution, depth, num_heads, window_size,
                                        shift_size, mlp_ratio, qkv_bias, qk_scale, drop,
                                        attn_drop, drop_path, use_masking, use_cos_attn,
                                        use_v2_norm_placement, use_rel_pos_bias, rngs)
        self.downsample = (PatchMerging(input_resolution, dim=dim, rngs=rngs)
                           if downsample else None)

    def __call__(self, x):
        for blk in self.blocks:
            x = nnx.remat(type(blk).__call__)(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayer_up(nnx.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, shift_size,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, upsample=False, use_checkpoint=False, use_masking=True,
                 use_cos_attn=False, use_v2_norm_placement=False, use_rel_pos_bias=True,
                 *, rngs):
        self.use_checkpoint = use_checkpoint
        self.blocks = _make_flat_blocks(dim, input_resolution, depth, num_heads, window_size,
                                        shift_size, mlp_ratio, qkv_bias, qk_scale, drop,
                                        attn_drop, drop_path, use_masking, use_cos_attn,
                                        use_v2_norm_placement, use_rel_pos_bias, rngs)
        self.upsample = (PatchExpand(input_resolution, dim=dim, dim_scale=2, rngs=rngs)
                         if upsample else None)

    def __call__(self, x):
        for blk in self.blocks:
            x = nnx.remat(type(blk).__call__)(blk, x) if self.use_checkpoint else blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class SwinEncoder(nnx.Module):
    def __init__(self, config: SwinTransformerConfig, data_spec: DataSpec, *, rngs):
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        H, W = data_spec.dim_in
        merge_factor = 2 ** (self.num_layers - 1)
        assert (H / (merge_factor * config.patch_size[0] * config.window_size[0])) % 1 == 0
        assert (W / (merge_factor * config.patch_size[1] * config.window_size[1])) % 1 == 0

        self.patch_embed = PatchEmbed(config, data_spec, rngs=rngs)
        pr = self.patch_embed.patches_resolution
        self.patches_resolution = pr
        if config.ape:
            self.absolute_pos_embed = nnx.Param(
                TRUNC_NORMAL(rngs.params(), (1, self.patch_embed.num_patches, config.embed_dim)))
        else:
            self.absolute_pos_embed = None
        self.pos_drop = nnx.Dropout(config.drop_rate, rngs=rngs)

        dpr = [float(v) for v in np.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(BasicLayer(
                dim=int(config.embed_dim * 2 ** i),
                input_resolution=(pr[0] // (2 ** i), pr[1] // (2 ** i)),
                depth=config.depths[i], num_heads=config.num_heads[i],
                window_size=config.window_size, shift_size=config.shift_size,
                mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                drop=config.drop_rate, attn_drop=config.attn_drop_rate,
                drop_path=dpr[sum(config.depths[:i]):sum(config.depths[:i + 1])],
                downsample=i < self.num_layers - 1, use_checkpoint=config.use_checkpoint,
                use_masking=config.use_masking, use_cos_attn=config.use_cos_attn,
                use_v2_norm_placement=config.use_v2_norm_placement,
                use_rel_pos_bias=config.use_rel_pos_bias, rngs=rngs))
        self.norm = nnx.LayerNorm(self.num_features, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed.value
        x = self.pos_drop(x)
        skips = []
        for layer in self.layers:
            skips.append(x)
            x = layer(x)
        return self.norm(x), skips


class UnetDecoder(nnx.Module):
    """Flat UNet decoder. New class: the reference inlines this in SwinTransformerSys."""

    def __init__(self, config: SwinTransformerConfig, data_spec: DataSpec, *, rngs):
        self.config = config
        self.num_layers = len(config.depths)
        pr = (data_spec.dim_in[0] // config.patch_size[0],
              data_spec.dim_in[1] // config.patch_size[1])
        self.patches_resolution = pr
        dpr = [float(v) for v in np.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.layers_up = []
        self.concat_back_dim = []
        for i_layer in range(self.num_layers):
            down_idx = self.num_layers - 1 - i_layer
            dim = int(config.embed_dim * 2 ** down_idx)
            res = (pr[0] // (2 ** down_idx), pr[1] // (2 ** down_idx))
            self.concat_back_dim.append(
                nnx.Linear(2 * dim, dim, kernel_init=TRUNC_NORMAL, rngs=rngs)
                if i_layer > 0 else Identity())
            if i_layer == 0:
                self.layers_up.append(PatchExpand(res, dim=dim, dim_scale=2, rngs=rngs))
            else:
                self.layers_up.append(BasicLayer_up(
                    dim=dim, input_resolution=res, depth=config.depths[down_idx],
                    num_heads=config.num_heads[down_idx], window_size=config.window_size,
                    shift_size=config.shift_size, mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                    drop=config.drop_rate, attn_drop=config.attn_drop_rate,
                    drop_path=dpr[sum(config.depths[:down_idx]):sum(config.depths[:down_idx + 1])],
                    upsample=i_layer < self.num_layers - 1,
                    use_checkpoint=config.use_checkpoint, use_masking=config.use_masking,
                    use_cos_attn=config.use_cos_attn,
                    use_v2_norm_placement=config.use_v2_norm_placement,
                    use_rel_pos_bias=config.use_rel_pos_bias, rngs=rngs))
        self.norm_up = nnx.LayerNorm(config.embed_dim, epsilon=LN_EPS, rngs=rngs)
        self.up = FinalPatchExpand_X4(pr, patch_size=config.patch_size, dim=config.embed_dim,
                                      rngs=rngs)
        self.output = nnx.Conv(config.embed_dim, data_spec.f_out, kernel_size=(1, 1),
                               use_bias=False, rngs=rngs)

    def __call__(self, x, skips, return_intermediates=False):
        intermediates = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = jnp.concatenate([x, skips[self.num_layers - 1 - inx]], axis=-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
            if return_intermediates:
                intermediates.append(x)
        x = self.norm_up(x)
        x = self.up(x)  # (B, H*W, embed_dim) at full resolution
        H = self.patches_resolution[0] * self.config.patch_size[0]
        W = self.patches_resolution[1] * self.config.patch_size[1]
        x = x.reshape(x.shape[0], H, W, x.shape[-1])
        x = self.output(x)  # (B, H, W, f_out)
        return (x, intermediates) if return_intermediates else x


class SwinTransformerSys(nnx.Module):
    def __init__(self, config: SwinTransformerConfig, data_spec: DataSpec, *, rngs):
        self.encoder = SwinEncoder(config, data_spec, rngs=rngs)
        self.decoder = UnetDecoder(config, data_spec, rngs=rngs)

    def __call__(self, x):
        tokens, skips = self.encoder(x)
        return self.decoder(tokens, skips)
```

Note on the flat decoder wiring vs the reference: reference `BasicLayer_up` in the flat model gets `upsample=PatchExpand if (i_layer < num_layers - 1)` — port that condition exactly as written above (`upsample=i_layer < self.num_layers - 1`; contrast with HP which uses `down_idx > 0` — they are equivalent, but keep each file's own form).

- [ ] **Step 4: Run everything** — `uv run pytest tests/ -q` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: flat SwinEncoder/UnetDecoder/SwinTransformerSys, e2e parity green"
```

---

### Task 18: JAX-native tests, public exports, docs

**Files:**
- Modify: `src/heal_swin_nnx/__init__.py`, `README.md`
- Test: `tests/test_model.py` (append)

**Interfaces:**
- Produces: `from heal_swin_nnx import SwinHPTransformerSys, SwinHPEncoder, HPUnetDecoder, SwinTransformerSys, SwinEncoder, UnetDecoder, SwinHPTransformerConfig, SwinTransformerConfig, DataSpec, Buffer`.

- [ ] **Step 1: Append failing tests**

```python
import pytest

from heal_swin_nnx import (
    Buffer, DataSpec, HPUnetDecoder, SwinHPEncoder, SwinHPTransformerConfig,
    SwinHPTransformerSys)
from heal_swin_nnx import hp_shifting as hps
from heal_swin_nnx.variables import Buffer as BufferDirect


def tiny_hp(base_pix=8, **over):
    cfg = SwinHPTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4],
                                  drop_path_rate=0.0, **over)
    ds = DataSpec(dim_in=base_pix * 16 ** 2, f_in=3, f_out=5, base_pix=base_pix)
    return SwinHPTransformerSys(cfg, ds, rngs=nnx.Rngs(0)), ds


def test_jit_matches_eager():
    model, ds = tiny_hp()
    model.eval()
    x = jax.random.normal(jax.random.key(0), (2, ds.dim_in, 3))
    np.testing.assert_allclose(np.asarray(nnx.jit(lambda m, x: m(x))(model, x)),
                               np.asarray(model(x)), rtol=1e-6, atol=1e-6)


def test_batch_independence():
    model, ds = tiny_hp()
    model.eval()
    x = jax.random.normal(jax.random.key(0), (3, ds.dim_in, 3))
    full = np.asarray(model(x))
    single = np.asarray(model(x[1:2]))
    np.testing.assert_allclose(full[1:2], single, rtol=1e-5, atol=1e-6)


def test_remat_matches_no_remat():
    m1, ds = tiny_hp()
    m2, _ = tiny_hp(use_checkpoint=True)
    # same rngs seed -> same weights
    m1.eval(); m2.eval()
    x = jax.random.normal(jax.random.key(0), (1, ds.dim_in, 3))
    np.testing.assert_allclose(np.asarray(m2(x)), np.asarray(m1(x)), rtol=1e-6, atol=1e-6)


def test_encoder_standalone_no_decoder_params():
    cfg = SwinHPTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4],
                                  drop_path_rate=0.0)
    ds = DataSpec(dim_in=2048, f_in=3, f_out=5, base_pix=8)
    enc = SwinHPEncoder(cfg, ds, rngs=nnx.Rngs(0))
    tokens, skips = enc(jnp.ones((1, 2048, 3)))
    assert tokens.shape == (1, 2048 // 4 // 4, 24)   # N/(patch*4^(L-1)), embed*2^(L-1)
    assert len(skips) == 2
    paths = [tuple(str(p) for p in path) for path, _ in nnx.state(enc, nnx.Param).flat_state()]
    assert not any("decoder" in p for path in paths for p in path)


def test_base_pix_12_nest_roll_works_grid_raises():
    model, ds = tiny_hp(base_pix=12)          # nest_roll is base_pix-agnostic
    model.eval()
    y = model(jnp.ones((1, ds.dim_in, 3)))
    assert y.shape == (1, ds.dim_in, 5)
    with pytest.raises(NotImplementedError):
        tiny_hp(base_pix=12, shift_strategy="nest_grid_shift")


def test_no_buffer_is_a_param():
    model, _ = tiny_hp(rel_pos_bias="flat")
    params = dict(nnx.state(model, nnx.Param).flat_state())
    for path in params:
        joined = "/".join(str(p) for p in path)
        assert "attn_mask" not in joined and "relative_position_index" not in joined
        assert "shift_idcs" not in joined
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_model.py -v` → FAIL on imports.

- [ ] **Step 3: Implement exports + README**

`src/heal_swin_nnx/__init__.py`:

```python
from heal_swin_nnx.config import DataSpec, SwinHPTransformerConfig, SwinTransformerConfig
from heal_swin_nnx.swin_hp_transformer import (
    HPUnetDecoder, SwinHPEncoder, SwinHPTransformerSys)
from heal_swin_nnx.swin_transformer import SwinEncoder, SwinTransformerSys, UnetDecoder
from heal_swin_nnx.variables import Buffer

__all__ = ["Buffer", "DataSpec", "HPUnetDecoder", "SwinEncoder", "SwinHPEncoder",
           "SwinHPTransformerConfig", "SwinHPTransformerSys", "SwinTransformerConfig",
           "SwinTransformerSys", "UnetDecoder"]
```

Update `README.md`: short usage section (build config + data spec, instantiate `SwinHPTransformerSys` or `SwinHPEncoder`, channels-last shapes), parity summary (goldens from the pinned reference env, forward + gradient), pointer to `parity/README.md` and the spec/plan docs.

- [ ] **Step 4: Run the full suite** — `uv run pytest tests/ -q` → ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: public API exports, JAX-native test suite, README"
```

---

## Self-Review Notes (already applied)

- Spec coverage: full option surface (3 shifters ✓ Tasks 7–9; cos-attn/v2-norm/rel-bias/APE ✓ golden matrix Tasks 4–5 + module code; remat ✓ Task 18), encoder/head seam ✓ Task 14/17, buffer safety ✓ Tasks 1/18, base_pix isolation + loud asserts ✓ Tasks 8/9/18, bit-exact index parity ✓ Tasks 6–9, forward+grad parity ✓ Tasks 11–17, flat model ✓ Tasks 16–17.
- Type consistency: `load_torch_state(model, sd, prefix_map)` / `torch_key_to_path` / `transform_array` used identically in Tasks 11, 14, 15, 17; shifter surface (`.shift/.shift_back/.attn_mask`) identical across Tasks 7–9 and consumed in Task 13.
- Known API risk: exact flax nnx `State.flat_state()` / `from_flat_path` signatures vary slightly across versions — Task 11 notes the two lines to adapt. `nnx.remat` usage in Tasks 13/17 is exercised by `test_remat_matches_no_remat`.
