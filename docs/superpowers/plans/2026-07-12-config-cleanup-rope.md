# Config Unification, Post-Parity Cleanup, and RoPE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the reference-mirroring config triple (`DataSpec` + `SwinHPTransformerConfig` + `SwinTransformerConfig`) with one pure-data Params dataclass per model, delete the torch-parity harness, keep only the SwinV2 attention path, add RoPE positional encoding (rope-vit style), and rename the API to the HealSwin/SwinUnet families in a `models/` + `hp/` layout.

**Architecture:** Two nnx model families (`HealSwin` on the HEALPix nested grid, `SwinUnet` on flat 2D) each own a serializable Params dataclass validated in `__post_init__`. Window attention is V2-only (cosine + post-norm) with a 4-way `pos_embed` switch (`none` / `rel_bias` / `rope_axial` / `rope_mixed`); RoPE is purely intra-window, so it never touches the shift machinery. HEALPix geometry lives untouched in an `hp/` subpackage.

**Tech Stack:** Python ≥3.12, JAX + Flax NNX, einops, healpy (tests), pytest via `uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-07-12-config-cleanup-design.md` — the authoritative design. Read it before starting.

## Global Constraints

- Run tests with `uv run pytest tests/ -q` (pyproject sets `JAX_PLATFORMS=cpu` and `-n 2`).
- `LN_EPS = 1e-5` everywhere (torch LayerNorm default; flax default 1e-6 is wrong here).
- Tensor convention: channels-last. HP: `(B, npix, C)`; flat: `(B, H, W, C)`.
- No PyTorch imports anywhere under `src/`.
- Params dataclasses are pure data: no `nnx.Rngs`, no arrays; `json.dumps(dataclasses.asdict(p))` must work.
- Public API after the final task is exactly: `HealSwin`, `HealSwinEncoder`, `HealSwinDecoder`, `HealSwinParams`, `SwinUnet`, `SwinEncoder`, `SwinDecoder`, `SwinParams`, `Buffer`.
- Commit after every task with the message given in the task. Do NOT push; do NOT delete the `parity-verified` tag.
- `references/` is read-only reference material — never modify it.

---

### Task 1: Tag the parity state, delete the parity harness

The torch-parity era ends here. Tag it so it stays one checkout away, then remove the harness: golden generator, goldens, parity tests, weight transfer, and the six golden-dependent test functions inside otherwise-kept test files.

**Files:**
- Delete: `parity/` (whole directory), `tests/goldens/` (whole directory), `tests/parity_utils.py`, `tests/test_parity_modules.py`, `tests/test_parity_e2e.py`, `tests/test_parity_f64.py`, `src/heal_swin_nnx/weight_transfer.py`
- Modify: `tests/test_shifting.py`, `tests/test_windowing.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: a repo with no goldens and no `tests.parity_utils` import anywhere; the `parity-verified` tag.

- [ ] **Step 1: Tag the current commit**

```bash
git tag -a parity-verified -m "Last state with torch-parity harness, goldens, and weight transfer intact"
git tag -l parity-verified
```

Expected: prints `parity-verified`.

- [ ] **Step 2: Delete the parity harness**

```bash
git rm -r -q parity tests/goldens tests/parity_utils.py \
  tests/test_parity_modules.py tests/test_parity_e2e.py tests/test_parity_f64.py \
  src/heal_swin_nnx/weight_transfer.py
```

- [ ] **Step 3: Remove golden-dependent tests from kept test files**

In `tests/test_shifting.py`, delete the import line `from tests.parity_utils import load_case` and these four test functions entirely (they compare against `load_case("indices")`):
- `test_nest_roll_mask_bit_exact`
- `test_nest_grid_idcs_bit_exact`
- `test_nest_grid_masks_bit_exact`
- `test_ring_idcs_and_masks_bit_exact`

All other tests in the file are ground-truth or round-trip tests and stay.

In `tests/test_windowing.py`, delete the import line `from tests.parity_utils import load_case` and these two test functions:
- `test_nest_win_idcs_bit_exact`
- `test_nest_relative_position_index_bit_exact`

(`get_nest_win_idcs` remains independently verified against healpy sky-adjacency by `tests/test_seam_geometry.py`.)

- [ ] **Step 4: Verify nothing references the deleted files**

```bash
grep -rn "parity_utils\|weight_transfer\|goldens" src tests
```

Expected: no output.

- [ ] **Step 5: Run the remaining tests**

```bash
uv run pytest tests/ -q
```

Expected: all tests pass (fewer than before; no errors, no collection failures).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore!: remove torch-parity harness (tagged parity-verified)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Move HEALPix geometry into the `hp/` subpackage

Pure file moves plus import updates. No behavior change — the existing tests are the regression check.

**Files:**
- Move: `src/heal_swin_nnx/hp_topology.py` → `src/heal_swin_nnx/hp/topology.py`; `src/heal_swin_nnx/hp_shifting.py` → `src/heal_swin_nnx/hp/shifting.py`; `src/heal_swin_nnx/hp_windowing.py` → `src/heal_swin_nnx/hp/windowing.py`
- Create: `src/heal_swin_nnx/hp/__init__.py` (empty)
- Modify: `src/heal_swin_nnx/hp/shifting.py` (one import), `src/heal_swin_nnx/swin_hp_transformer.py` (two imports), `tests/test_shifting.py`, `tests/test_topology.py`, `tests/test_seam_geometry.py`, `tests/test_windowing.py`

**Interfaces:**
- Consumes: Task 1 (parity harness gone).
- Produces: import paths `heal_swin_nnx.hp.topology`, `heal_swin_nnx.hp.shifting`, `heal_swin_nnx.hp.windowing` used by every later task. Module contents unchanged.

- [ ] **Step 1: Move the files**

```bash
mkdir -p src/heal_swin_nnx/hp
git mv src/heal_swin_nnx/hp_topology.py src/heal_swin_nnx/hp/topology.py
git mv src/heal_swin_nnx/hp_shifting.py src/heal_swin_nnx/hp/shifting.py
git mv src/heal_swin_nnx/hp_windowing.py src/heal_swin_nnx/hp/windowing.py
touch src/heal_swin_nnx/hp/__init__.py
git add src/heal_swin_nnx/hp/__init__.py
```

- [ ] **Step 2: Fix the intra-package import in `hp/shifting.py`**

Replace:

```python
from heal_swin_nnx import hp_topology
```

with (alias keeps every `hp_topology.` reference in the body working unmodified):

```python
from heal_swin_nnx.hp import topology as hp_topology
```

- [ ] **Step 3: Fix imports in `src/heal_swin_nnx/swin_hp_transformer.py`**

Replace:

```python
from heal_swin_nnx import hp_shifting
```

with:

```python
from heal_swin_nnx.hp import shifting as hp_shifting
```

and replace:

```python
from heal_swin_nnx.hp_windowing import (
    nest_relative_position_index, window_partition, window_reverse)
```

with:

```python
from heal_swin_nnx.hp.windowing import (
    nest_relative_position_index, window_partition, window_reverse)
```

- [ ] **Step 4: Fix test imports**

```bash
sed -i \
  -e 's/from heal_swin_nnx import hp_topology as hpt/from heal_swin_nnx.hp import topology as hpt/' \
  -e 's/from heal_swin_nnx import hp_shifting as hps/from heal_swin_nnx.hp import shifting as hps/' \
  -e 's/from heal_swin_nnx\.hp_windowing import/from heal_swin_nnx.hp.windowing import/' \
  -e 's/from heal_swin_nnx import hp_windowing/from heal_swin_nnx.hp import windowing as hp_windowing/' \
  tests/test_shifting.py tests/test_topology.py tests/test_seam_geometry.py tests/test_windowing.py
grep -rn "hp_topology\|hp_shifting\|hp_windowing" src tests | grep -v "heal_swin_nnx.hp import\|heal_swin_nnx.hp.windowing"
```

Expected: the final grep prints nothing (every remaining reference goes through the new paths). If it prints a line, update that import by hand to the same pattern.

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: move HEALPix geometry modules into hp/ subpackage

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: RoPE primitives

Model-independent building blocks: intra-window coordinates for HP windows, frequency init, rotation-table construction, and q/k rotation. TDD with a new `tests/test_rope.py`.

**Files:**
- Modify: `src/heal_swin_nnx/hp/windowing.py` (add `nest_win_coords`), `src/heal_swin_nnx/layers.py` (add `LN_EPS`, `init_rope_freqs`, `rope_rotation_table`, `apply_rope`)
- Test: `tests/test_rope.py` (new)

**Interfaces:**
- Consumes: `get_nest_win_idcs(window_size)` from `heal_swin_nnx.hp.windowing` (existing).
- Produces (used by Tasks 5 and 6):
  - `nest_win_coords(window_size: int) -> np.ndarray` — shape `(2, window_size)` float32; row 0 = x, row 1 = y Cartesian coordinate of each nested-scheme index inside one window.
  - `init_rope_freqs(head_dim: int, num_heads: int, theta: float, key=None) -> jax.Array` — shape `(2, num_heads, head_dim // 2)`; `key=None` gives axis-aligned (axial) frequencies, a PRNG key gives rope-vit's random per-head rotation init (mixed).
  - `rope_rotation_table(freqs, t_x, t_y) -> jax.Array` — freqs `(2, H, D/2)`, coords `(N,)` → table `(H, N, D/2, 2, 2)`.
  - `apply_rope(q, k, table) -> tuple[jax.Array, jax.Array]` — q, k `(B, H, N, D)` rotated in f32, returned in input dtype.
  - `LN_EPS = 1e-5` importable from `heal_swin_nnx.layers`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rope.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from heal_swin_nnx.hp.windowing import get_nest_win_idcs, nest_win_coords
from heal_swin_nnx.layers import apply_rope, init_rope_freqs, rope_rotation_table


@pytest.mark.parametrize("window_size", [4, 16, 64])
def test_nest_win_coords_roundtrips_grid(window_size):
    # derive-then-verify: coords must invert the independently tested nested->
    # Cartesian window map (get_nest_win_idcs is pinned to healpy sky adjacency
    # by test_seam_geometry.py)
    grid = get_nest_win_idcs(window_size)
    coords = nest_win_coords(window_size)
    assert coords.shape == (2, window_size) and coords.dtype == np.float32
    for i in range(window_size):
        x, y = int(coords[0, i]), int(coords[1, i])
        assert grid[x, y] == i


def test_init_rope_freqs_shapes_and_axial_alignment():
    axial = init_rope_freqs(16, 3, 10.0)
    assert axial.shape == (2, 3, 8)
    # axial: first D/4 pairs are pure-x (fy == 0), second D/4 pure-y (fx == 0)
    np.testing.assert_allclose(axial[1, :, :4], 0.0, atol=1e-7)
    np.testing.assert_allclose(axial[0, :, 4:], 0.0, atol=1e-7)
    mixed = init_rope_freqs(16, 3, 10.0, key=jax.random.key(0))
    assert mixed.shape == (2, 3, 8)
    # mixed init: heads got distinct random rotations
    assert not np.allclose(mixed[:, 0], mixed[:, 1])


def test_apply_rope_preserves_norms():
    q = jax.random.normal(jax.random.key(0), (2, 3, 8, 16))
    k = jax.random.normal(jax.random.key(1), (2, 3, 8, 16))
    freqs = init_rope_freqs(16, 3, 10.0, key=jax.random.key(2))
    t = jnp.arange(8, dtype=jnp.float32)
    table = rope_rotation_table(freqs, t, t[::-1])
    q2, k2 = apply_rope(q, k, table)
    assert q2.shape == q.shape and k2.shape == k.shape
    np.testing.assert_allclose(np.linalg.norm(np.asarray(q2), axis=-1),
                               np.linalg.norm(np.asarray(q), axis=-1), rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(k2), axis=-1),
                               np.linalg.norm(np.asarray(k), axis=-1), rtol=1e-5)


def test_rope_logits_depend_only_on_coordinate_offset():
    # (R(ti)q)^T (R(tj)k) must be invariant under a global coordinate shift —
    # holds for any freqs, so test with the mixed (random-rotation) init
    freqs = init_rope_freqs(16, 2, 10.0, key=jax.random.key(0))
    t_x = jnp.array([0.0, 1.0, 2.0, 5.0])
    t_y = jnp.array([3.0, 0.0, 2.0, 1.0])
    q = jax.random.normal(jax.random.key(1), (1, 2, 4, 16))
    k = jax.random.normal(jax.random.key(2), (1, 2, 4, 16))

    def logits(dx, dy):
        table = rope_rotation_table(freqs, t_x + dx, t_y + dy)
        q2, k2 = apply_rope(q, k, table)
        return np.asarray(q2 @ k2.swapaxes(-2, -1))

    np.testing.assert_allclose(logits(0.0, 0.0), logits(3.0, 7.0),
                               rtol=1e-4, atol=1e-5)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_rope.py -q
```

Expected: FAIL / collection error with `ImportError: cannot import name 'nest_win_coords'`.

- [ ] **Step 3: Implement `nest_win_coords` in `src/heal_swin_nnx/hp/windowing.py`**

Append at the end of the file:

```python
def nest_win_coords(window_size):
    """(2, window_size) float32: Cartesian (x, y) of each nested-scheme index
    inside one window — the inverse of ``get_nest_win_idcs``. Used as RoPE
    token coordinates; same flat-grid approximation as the rel-pos bias."""
    grid = get_nest_win_idcs(window_size)
    s = grid.shape[0]
    coords = np.zeros((2, window_size), dtype=np.float32)
    xs, ys = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")
    coords[0, grid.reshape(-1)] = xs.reshape(-1).astype(np.float32)
    coords[1, grid.reshape(-1)] = ys.reshape(-1).astype(np.float32)
    return coords
```

- [ ] **Step 4: Implement the RoPE helpers in `src/heal_swin_nnx/layers.py`**

Update the module docstring and add `LN_EPS` plus the three functions (adapted from rope-vit `init_random_2d_freqs`/`compute_cis` and GenSBI `flux1/math.py` `apply_rope`):

```python
"""Shared leaf modules and RoPE primitives used by both models."""
```

Append at the end of the file:

```python
LN_EPS = 1e-5  # torch nn.LayerNorm default; flax default (1e-6) differs


def init_rope_freqs(head_dim, num_heads, theta, key=None):
    """(2, num_heads, head_dim // 2) x/y RoPE frequency magnitudes.

    rope-vit init_random_2d_freqs: ``key=None`` gives axis-aligned frequencies
    (rope_axial — first D/4 pairs rotate with x, second D/4 with y, all heads
    identical); a PRNG key applies a random per-head rotation of the two axes
    (rope_mixed init)."""
    assert head_dim % 4 == 0, "RoPE needs head_dim divisible by 4"
    mag = 1.0 / theta ** (
        jnp.arange(0, head_dim, 4, dtype=jnp.float32)[: head_dim // 4] / head_dim)
    if key is None:
        angles = jnp.zeros((num_heads, 1), dtype=jnp.float32)
    else:
        angles = jax.random.uniform(key, (num_heads, 1)) * 2 * jnp.pi
    fx = jnp.concatenate([mag * jnp.cos(angles), mag * jnp.cos(jnp.pi / 2 + angles)], axis=-1)
    fy = jnp.concatenate([mag * jnp.sin(angles), mag * jnp.sin(jnp.pi / 2 + angles)], axis=-1)
    return jnp.stack([fx, fy])


def rope_rotation_table(freqs, t_x, t_y):
    """freqs (2, H, D/2) + coords (N,) -> (H, N, D/2, 2, 2) rotation matrices."""
    angles = (t_x[None, :, None] * freqs[0][:, None, :]
              + t_y[None, :, None] * freqs[1][:, None, :])  # (H, N, D/2)
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    return jnp.stack([cos, -sin, sin, cos], axis=-1).reshape(*angles.shape, 2, 2)


def apply_rope(q, k, table):
    """Rotate q, k (B, H, N, D) by table (H, N, D/2, 2, 2). Computed in f32
    (angle precision), returned in the input dtype. Norm-preserving."""
    def rot(x):
        xr = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 1, 2)
        out = table[..., 0] * xr[..., 0] + table[..., 1] * xr[..., 1]
        return out.reshape(*x.shape).astype(x.dtype)
    return rot(q), rot(k)
```

- [ ] **Step 5: Run the new tests**

```bash
uv run pytest tests/test_rope.py -q
```

Expected: all 6 PASS (3 parametrized + 3).

- [ ] **Step 6: Run the full suite (no regressions)**

```bash
uv run pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/heal_swin_nnx/hp/windowing.py src/heal_swin_nnx/layers.py tests/test_rope.py
git commit -m "feat: RoPE primitives — intra-window coords, freq init, rotation apply

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `HealSwinParams` dataclass

The unified HP Params with all validation up front. Lives in the new `models/healswin.py` (model classes join it in Task 5). Replaces `test_dataspec.py` with `test_params.py`.

**Files:**
- Create: `src/heal_swin_nnx/models/__init__.py` (empty), `src/heal_swin_nnx/models/healswin.py`
- Delete: `tests/test_dataspec.py`
- Test: `tests/test_params.py` (new)

**Interfaces:**
- Consumes: nothing (pure dataclass; stdlib only).
- Produces (used by Tasks 5 and 7):
  - `HealSwinParams(nside, in_channels, out_channels, base_pixels=None, patch_size=4, window_size=4, embed_dim=96, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), mlp_ratio=4.0, qkv_bias=True, pos_embed="rope_mixed", rope_theta=10.0, patch_embed_norm=False, shift_strategy="nest_grid_shift_exact", drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, use_checkpoint=False)`
  - Properties: `p.npix -> int` (`len(base_pixels) * nside**2`), `p.shift_size -> int` (`window_size // 2`).
  - After `__post_init__`, `base_pixels`, `depths`, `num_heads` are tuples.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_params.py`:

```python
import dataclasses
import json

import pytest

from heal_swin_nnx.models.healswin import HealSwinParams


def test_defaults_full_sphere():
    p = HealSwinParams(nside=16, in_channels=3, out_channels=5)
    assert p.base_pixels == tuple(range(12))
    assert p.npix == 12 * 16 ** 2
    assert p.shift_size == 2
    assert p.pos_embed == "rope_mixed"
    assert p.shift_strategy == "nest_grid_shift_exact"
    assert p.depths == (2, 2, 2, 2) and p.num_heads == (3, 6, 12, 24)


def test_partial_coverage_and_tuple_coercion():
    p = HealSwinParams(nside=16, in_channels=1, out_channels=1,
                       base_pixels=[8, 9, 10, 11], depths=[2, 2], num_heads=[2, 4],
                       embed_dim=16)
    assert p.base_pixels == (8, 9, 10, 11)
    assert p.npix == 4 * 16 ** 2
    assert p.depths == (2, 2)


@pytest.mark.parametrize("bad", [[0, 0, 1], [3, 2], [-1, 0], [11, 12]])
def test_invalid_base_pixels_rejected(bad):
    with pytest.raises(ValueError):
        HealSwinParams(nside=16, in_channels=1, out_channels=1, base_pixels=bad)


def test_params_is_pure_serializable_data():
    p = HealSwinParams(nside=16, in_channels=3, out_channels=5)
    json.dumps(dataclasses.asdict(p))  # must not raise


def test_nside_must_be_power_of_two():
    with pytest.raises(ValueError):
        HealSwinParams(nside=12, in_channels=1, out_channels=1)


def test_not_enough_resolution_for_stages():
    # nside=4, patch_size=4 -> 4 patches/face; 3 stages need divisibility by 4^2
    with pytest.raises(ValueError):
        HealSwinParams(nside=4, in_channels=1, out_channels=1,
                       depths=(2, 2, 2), num_heads=(2, 2, 2), embed_dim=8)


def test_depths_num_heads_length_mismatch_rejected():
    with pytest.raises(ValueError):
        HealSwinParams(nside=16, in_channels=1, out_channels=1,
                       depths=(2, 2), num_heads=(2, 4, 8))


def test_rope_requires_head_dim_divisible_by_4():
    # embed_dim=12, heads (2, 4): head dims 6, 6 -> invalid for RoPE...
    with pytest.raises(ValueError):
        HealSwinParams(nside=16, in_channels=1, out_channels=1,
                       embed_dim=12, depths=(2, 2), num_heads=(2, 4))
    # ...but fine without positional encoding
    HealSwinParams(nside=16, in_channels=1, out_channels=1,
                   embed_dim=12, depths=(2, 2), num_heads=(2, 4), pos_embed="none")
    # and fine with head dims divisible by 4
    HealSwinParams(nside=16, in_channels=1, out_channels=1,
                   embed_dim=16, depths=(2, 2), num_heads=(2, 4))


def test_embed_dim_must_divide_by_heads():
    with pytest.raises(ValueError):
        HealSwinParams(nside=16, in_channels=1, out_channels=1,
                       embed_dim=16, depths=(2, 2), num_heads=(3, 4))


def test_unknown_enum_values_rejected():
    with pytest.raises(ValueError):
        HealSwinParams(nside=16, in_channels=1, out_channels=1, pos_embed="learned")
    with pytest.raises(ValueError):
        HealSwinParams(nside=16, in_channels=1, out_channels=1, shift_strategy="roll")
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_params.py -q
```

Expected: collection error — `ModuleNotFoundError: No module named 'heal_swin_nnx.models'`.

- [ ] **Step 3: Implement**

```bash
mkdir -p src/heal_swin_nnx/models && touch src/heal_swin_nnx/models/__init__.py
```

Create `src/heal_swin_nnx/models/healswin.py`:

```python
"""HealSwin: HEALPix-native Swin V2 U-Net (diverged from the HEAL-SWIN reference)."""
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

POS_EMBEDS = ("none", "rel_bias", "rope_axial", "rope_mixed")
SHIFT_STRATEGIES = ("nest_roll", "nest_grid_shift", "nest_grid_shift_exact", "ring_shift")


@dataclass
class HealSwinParams:
    """Pure-data description of a HealSwin model (architecture + geometry).

    Serializable: ``json.dumps(dataclasses.asdict(params))`` works, so a run's
    exact configuration can be logged and compared."""

    # data / geometry
    nside: int                       # HEALPix resolution of the input map
    in_channels: int
    out_channels: int
    base_pixels: Optional[Union[Tuple[int, ...], Sequence[int]]] = None  # None -> full sphere

    # architecture
    patch_size: int = 4
    window_size: int = 4
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 2, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    pos_embed: Literal["none", "rel_bias", "rope_axial", "rope_mixed"] = "rope_mixed"
    rope_theta: float = 10.0
    patch_embed_norm: bool = False
    shift_strategy: Literal["nest_roll", "nest_grid_shift", "nest_grid_shift_exact",
                            "ring_shift"] = "nest_grid_shift_exact"

    # regularization / training
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    use_checkpoint: bool = False

    def __post_init__(self):
        if self.base_pixels is None:
            self.base_pixels = tuple(range(12))
        self.base_pixels = tuple(self.base_pixels)
        self.depths = tuple(self.depths)
        self.num_heads = tuple(self.num_heads)

        if any(not 0 <= b <= 11 for b in self.base_pixels):
            raise ValueError("base_pixels must be in [0, 11], got %r" % (self.base_pixels,))
        if any(a >= b for a, b in zip(self.base_pixels, self.base_pixels[1:])):
            raise ValueError(
                "base_pixels must be strictly increasing (canonical NEST subset order), "
                "got %r" % (self.base_pixels,))
        if self.pos_embed not in POS_EMBEDS:
            raise ValueError("pos_embed must be one of %r, got %r"
                             % (POS_EMBEDS, self.pos_embed))
        if self.shift_strategy not in SHIFT_STRATEGIES:
            raise ValueError("shift_strategy must be one of %r, got %r"
                             % (SHIFT_STRATEGIES, self.shift_strategy))
        if len(self.depths) != len(self.num_heads):
            raise ValueError("depths (%d) and num_heads (%d) must have equal length"
                             % (len(self.depths), len(self.num_heads)))
        if self.patch_size <= 0 or self.patch_size % 4 != 0:
            raise ValueError("patch_size must be a positive multiple of 4 "
                             "(valid nside in deeper layers), got %d" % self.patch_size)
        if self.window_size <= 0 or self.window_size & (self.window_size - 1):
            raise ValueError("window_size must be a power of two, got %d" % self.window_size)
        if self.nside <= 0 or self.nside & (self.nside - 1):
            raise ValueError("nside must be a power of two, got %d" % self.nside)
        if self.nside ** 2 % self.patch_size:
            raise ValueError("nside^2 (%d) must be divisible by patch_size (%d)"
                             % (self.nside ** 2, self.patch_size))
        n_stages = len(self.depths)
        if (self.nside ** 2 // self.patch_size) % 4 ** (n_stages - 1):
            raise ValueError(
                "nside^2/patch_size (%d) must be divisible by 4^(n_stages-1) (%d): "
                "every encoder stage needs an integer per-face nside"
                % (self.nside ** 2 // self.patch_size, 4 ** (n_stages - 1)))
        for i, heads in enumerate(self.num_heads):
            dim = self.embed_dim * 2 ** i
            if dim % heads:
                raise ValueError("stage %d: dim %d not divisible by num_heads %d"
                                 % (i, dim, heads))
            if self.pos_embed in ("rope_axial", "rope_mixed") and (dim // heads) % 4:
                raise ValueError(
                    "stage %d: head_dim %d must be divisible by 4 for RoPE "
                    "(2D frequency split)" % (i, dim // heads))

    @property
    def npix(self):
        return len(self.base_pixels) * self.nside ** 2

    @property
    def shift_size(self):
        return self.window_size // 2
```

- [ ] **Step 4: Run the new tests**

```bash
uv run pytest tests/test_params.py -q
```

Expected: all PASS.

- [ ] **Step 5: Delete the superseded DataSpec tests and run the suite**

```bash
git rm -q tests/test_dataspec.py
uv run pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: HealSwinParams — unified, validated, serializable model config

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: HealSwin model family

The big one: rewrite the HP model in `models/healswin.py` — V2-only attention with the `pos_embed` switch, Params threaded down, renamed classes — then rewrite `test_model.py`, add pos-embed forward tests to `test_rope.py`, delete `swin_hp_transformer.py`, and re-point `__init__.py`.

**Files:**
- Modify: `src/heal_swin_nnx/models/healswin.py` (append model classes), `src/heal_swin_nnx/__init__.py`, `tests/test_rope.py` (append)
- Delete: `src/heal_swin_nnx/swin_hp_transformer.py`
- Test: `tests/test_model.py` (rewrite)

**Interfaces:**
- Consumes: `HealSwinParams` (Task 4); `nest_win_coords`, `nest_relative_position_index`, `window_partition`, `window_reverse` from `heal_swin_nnx.hp.windowing`; `shifting.NoShift/NestRollShift/NestGridShift/NestGridShiftExact/RingShift` from `heal_swin_nnx.hp.shifting`; `LN_EPS`, `TRUNC_NORMAL`, `DropPath`, `Identity`, `Mlp`, `apply_rope`, `init_rope_freqs`, `rope_rotation_table` from `heal_swin_nnx.layers`; `Buffer` from `heal_swin_nnx.variables`.
- Produces (used by Tasks 6–7 and by users):
  - `HealSwinEncoder(params: HealSwinParams, *, rngs)` — `__call__(x: (B, npix, in_channels)) -> (tokens, skips)`
  - `HealSwinDecoder(params: HealSwinParams, *, rngs)` — `__call__(tokens, skips) -> (B, npix, out_channels)`
  - `HealSwin(params: HealSwinParams, *, rngs)` — `__call__(x) -> (B, npix, out_channels)`
  - Internal (mirrored by Task 6's flat file): `WindowAttention(params, dim, num_heads, window_size, *, rngs)`, `HealSwinBlock(params, dim, input_resolution, num_heads, shifted, drop_path, *, rngs)`, `EncoderStage(params, dim, input_resolution, depth, num_heads, drop_path, downsample, *, rngs)`, `DecoderStage(... upsample ...)`.

- [ ] **Step 1: Rewrite `tests/test_model.py` (the failing tests)**

Replace the whole file. Notes vs. the old version: `embed_dim=16` (not 12) so the default `rope_mixed` satisfies head_dim % 4 == 0; `pos_embed` exercised explicitly; encoder token dim is `16 * 2 = 32`.

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from heal_swin_nnx import HealSwin, HealSwinEncoder, HealSwinParams
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


def tiny_params(**over):
    kw = dict(nside=16, in_channels=3, out_channels=5, base_pixels=tuple(range(8)),
              embed_dim=16, depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0)
    kw.update(over)
    return HealSwinParams(**kw)


def tiny_hp(**over):
    p = tiny_params(**over)
    return HealSwin(p, rngs=nnx.Rngs(0)), p


def test_jit_matches_eager():
    model, p = tiny_hp()
    model.eval()
    x = jax.random.normal(jax.random.key(0), (2, p.npix, 3))
    np.testing.assert_allclose(np.asarray(nnx.jit(lambda m, x: m(x))(model, x)),
                               np.asarray(model(x)), rtol=1e-6, atol=1e-6)


def test_batch_independence():
    model, p = tiny_hp()
    model.eval()
    x = jax.random.normal(jax.random.key(0), (3, p.npix, 3))
    full = np.asarray(model(x))
    single = np.asarray(model(x[1:2]))
    np.testing.assert_allclose(full[1:2], single, rtol=1e-5, atol=1e-6)


def test_remat_matches_no_remat():
    m1, p = tiny_hp()
    m2, _ = tiny_hp(use_checkpoint=True)
    # same rngs seed -> same weights
    m1.eval(); m2.eval()
    x = jax.random.normal(jax.random.key(0), (1, p.npix, 3))
    np.testing.assert_allclose(np.asarray(m2(x)), np.asarray(m1(x)), rtol=1e-6, atol=1e-6)


def test_encoder_standalone_no_decoder_params():
    p = tiny_params()
    enc = HealSwinEncoder(p, rngs=nnx.Rngs(0))
    tokens, skips = enc(jnp.ones((1, p.npix, 3)))
    assert tokens.shape == (1, p.npix // 4 // 4, 32)   # N/(patch*4^(L-1)), embed*2^(L-1)
    assert len(skips) == 2
    paths = [tuple(str(q) for q in path)
             for path, _ in nnx.to_flat_state(nnx.state(enc, nnx.Param))]
    assert not any("decoder" in q for path in paths for q in path)


@pytest.mark.parametrize("strategy",
                         ["nest_roll", "nest_grid_shift", "nest_grid_shift_exact",
                          "ring_shift"])
@pytest.mark.parametrize("base_pixels", [tuple(range(12)), (8, 9, 10, 11)])
def test_forward_full_sphere_and_south_cap(base_pixels, strategy):
    model, p = tiny_hp(base_pixels=base_pixels, shift_strategy=strategy)
    model.eval()
    x = jax.random.normal(jax.random.key(0), (2, p.npix, 3))
    y = model(x)
    assert y.shape == (2, p.npix, 5)
    assert np.isfinite(np.asarray(y)).all()


def test_params_are_json_loggable_next_to_model():
    import dataclasses, json
    _, p = tiny_hp()
    json.dumps(dataclasses.asdict(p))


def test_no_buffer_is_a_param():
    model, _ = tiny_hp(pos_embed="rel_bias")
    params = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))
    for path in params:
        joined = "/".join(str(q) for q in path)
        assert "attn_mask" not in joined and "relative_position_index" not in joined
        assert "shift_idcs" not in joined


def test_rope_buffers_and_params_sorted_correctly():
    mixed, _ = tiny_hp(pos_embed="rope_mixed")
    param_paths = ["/".join(str(q) for q in path)
                   for path, _ in nnx.to_flat_state(nnx.state(mixed, nnx.Param))]
    assert any("rope_freqs" in p for p in param_paths)      # learned freqs train
    assert not any("rope_coords" in p for p in param_paths)  # coords are Buffers

    axial, _ = tiny_hp(pos_embed="rope_axial")
    param_paths = ["/".join(str(q) for q in path)
                   for path, _ in nnx.to_flat_state(nnx.state(axial, nnx.Param))]
    assert not any("rope_table" in p for p in param_paths)   # fixed table is a Buffer
```

- [ ] **Step 2: Append the model-level pos-embed forward tests to `tests/test_rope.py`**

```python
def test_healswin_forward_all_pos_embeds():
    from flax import nnx
    from heal_swin_nnx import HealSwin, HealSwinParams
    for pos_embed in ["none", "rel_bias", "rope_axial", "rope_mixed"]:
        p = HealSwinParams(nside=16, in_channels=2, out_channels=3,
                           base_pixels=(8, 9, 10, 11), embed_dim=16,
                           depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0,
                           pos_embed=pos_embed)
        model = HealSwin(p, rngs=nnx.Rngs(0))
        model.eval()
        y = model(jnp.ones((1, p.npix, 2)))
        assert y.shape == (1, p.npix, 3), pos_embed
        assert np.isfinite(np.asarray(y)).all(), pos_embed
```

- [ ] **Step 3: Run to verify failure**

```bash
uv run pytest tests/test_model.py tests/test_rope.py -q
```

Expected: FAIL — `ImportError: cannot import name 'HealSwin'`.

- [ ] **Step 4: Append the model classes to `src/heal_swin_nnx/models/healswin.py`**

Extend the imports at the top of the file to:

```python
"""HealSwin: HEALPix-native Swin V2 U-Net (diverged from the HEAL-SWIN reference)."""
import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx

from heal_swin_nnx.hp import shifting
from heal_swin_nnx.hp.windowing import (
    nest_relative_position_index, nest_win_coords, window_partition, window_reverse)
from heal_swin_nnx.layers import (
    LN_EPS, TRUNC_NORMAL, DropPath, Identity, Mlp, apply_rope, init_rope_freqs,
    rope_rotation_table)
from heal_swin_nnx.variables import Buffer
```

Then append after `HealSwinParams` (everything below is new code replacing `swin_hp_transformer.py`; V2 cosine attention and post-norm placement are unconditional, `pos_embed` selects the positional encoding, Params is threaded down with only stage-local values as extra args):

```python
class WindowAttention(nnx.Module):
    """Swin V2 window attention: cosine similarity with learned logit scale,
    positional encoding selected by ``params.pos_embed``."""

    def __init__(self, params, dim, num_heads, window_size, *, rngs):
        self.num_heads = num_heads
        self.pos_embed = params.pos_embed
        head_dim = dim // num_heads
        self.logit_scale = nnx.Param(jnp.log(10.0 * jnp.ones((num_heads, 1, 1))))

        if self.pos_embed == "rel_bias":
            s = int(round(window_size ** 0.5))
            assert s * s == window_size, "rel_bias needs a square (power-of-4) window"
            self.relative_position_bias_table = nnx.Param(
                TRUNC_NORMAL(rngs.params(), ((2 * s - 1) ** 2, num_heads)))
            self.relative_position_index = Buffer(
                jnp.asarray(nest_relative_position_index(window_size)))
        elif self.pos_embed in ("rope_axial", "rope_mixed"):
            coords = jnp.asarray(nest_win_coords(window_size))  # (2, window_size)
            if self.pos_embed == "rope_mixed":
                self.rope_freqs = nnx.Param(init_rope_freqs(
                    head_dim, num_heads, params.rope_theta, key=rngs.params()))
                self.rope_coords = Buffer(coords)
            else:
                freqs = init_rope_freqs(head_dim, num_heads, params.rope_theta)
                self.rope_table = Buffer(rope_rotation_table(freqs, coords[0], coords[1]))

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=params.qkv_bias,
                              kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.attn_drop = nnx.Dropout(params.attn_drop_rate, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.proj_drop = nnx.Dropout(params.drop_rate, rngs=rngs)

    def __call__(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
        k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-12)
        if self.pos_embed == "rope_mixed":
            coords = self.rope_coords[...]
            table = rope_rotation_table(self.rope_freqs[...], coords[0], coords[1])
            q, k = apply_rope(q, k, table)
        elif self.pos_embed == "rope_axial":
            q, k = apply_rope(q, k, self.rope_table[...])
        attn = q @ k.swapaxes(-2, -1)
        logit_scale = jnp.exp(jnp.minimum(self.logit_scale[...], jnp.log(1.0 / 0.01)))
        attn = attn * logit_scale

        if self.pos_embed == "rel_bias":
            bias = self.relative_position_bias_table[...][self.relative_position_index[...]]
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


class FinalPatchExpand(nnx.Module):
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


class HealSwinBlock(nnx.Module):
    def __init__(self, params, dim, input_resolution, num_heads, shifted, drop_path, *, rngs):
        self.input_resolution = input_resolution
        self.window_size = min(params.window_size, input_resolution)
        shift_size = params.shift_size if (shifted
                                           and input_resolution > params.window_size) else 0

        self.norm1 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.attn = WindowAttention(params, dim, num_heads, self.window_size, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * params.mlp_ratio), drop=params.drop_rate, rngs=rngs)

        nside = math.isqrt(input_resolution // len(params.base_pixels))
        assert nside * nside * len(params.base_pixels) == input_resolution, \
            "nside has to be an integer in every layer"

        if shift_size > 0:
            if params.shift_strategy == "nest_roll":
                self.shifter = shifting.NestRollShift(
                    shift_size=shift_size, input_resolution=input_resolution,
                    window_size=self.window_size)
            elif params.shift_strategy == "nest_grid_shift":
                self.shifter = shifting.NestGridShift(
                    nside=nside, base_pixels=params.base_pixels,
                    window_size=self.window_size)
            elif params.shift_strategy == "nest_grid_shift_exact":
                self.shifter = shifting.NestGridShiftExact(
                    nside=nside, base_pixels=params.base_pixels,
                    window_size=self.window_size)
            else:  # "ring_shift" — Params validated the enum
                self.shifter = shifting.RingShift(
                    nside=nside, base_pixels=params.base_pixels,
                    window_size=self.window_size, shift_size=shift_size)
        else:
            self.shifter = shifting.NoShift()

    def __call__(self, x):
        shortcut = x
        shifted_x = self.shifter.shift(x)
        x_windows = window_partition(shifted_x, self.window_size)
        mask = None if self.shifter.attn_mask is None else self.shifter.attn_mask[...]
        attn_windows = self.attn(x_windows, mask=mask)
        shifted_x = window_reverse(attn_windows, self.window_size, self.input_resolution)
        x = self.shifter.shift_back(shifted_x)

        x = shortcut + self.drop_path(self.norm1(x))
        return x + self.drop_path(self.norm2(self.mlp(x)))


def _make_blocks(params, dim, input_resolution, depth, num_heads, drop_path, rngs):
    return [HealSwinBlock(params, dim, input_resolution, num_heads,
                          shifted=(i % 2 == 1), drop_path=drop_path[i], rngs=rngs)
            for i in range(depth)]


class EncoderStage(nnx.Module):
    def __init__(self, params, dim, input_resolution, depth, num_heads, drop_path,
                 downsample, *, rngs):
        self.use_checkpoint = params.use_checkpoint
        self.blocks = nnx.List(_make_blocks(params, dim, input_resolution, depth,
                                            num_heads, drop_path, rngs))
        self.downsample = PatchMerging(dim=dim, rngs=rngs) if downsample else None

    def __call__(self, x):
        for blk in self.blocks:
            x = nnx.remat(type(blk).__call__)(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class DecoderStage(nnx.Module):
    def __init__(self, params, dim, input_resolution, depth, num_heads, drop_path,
                 upsample, *, rngs):
        self.use_checkpoint = params.use_checkpoint
        self.blocks = nnx.List(_make_blocks(params, dim, input_resolution, depth,
                                            num_heads, drop_path, rngs))
        self.upsample = PatchExpand(dim=dim, dim_scale=2, rngs=rngs) if upsample else None

    def __call__(self, x):
        for blk in self.blocks:
            x = nnx.remat(type(blk).__call__)(blk, x) if self.use_checkpoint else blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nnx.Module):
    def __init__(self, params, *, rngs):
        self.npix = params.npix
        self.num_patches = params.npix // params.patch_size
        self.proj = nnx.Conv(params.in_channels, params.embed_dim,
                             kernel_size=(params.patch_size,), strides=(params.patch_size,),
                             padding="VALID", rngs=rngs)
        self.norm = (nnx.LayerNorm(params.embed_dim, epsilon=LN_EPS, rngs=rngs)
                     if params.patch_embed_norm else None)

    def __call__(self, x):  # (B, N, in_channels) channels-last
        assert x.shape[1] == self.npix, (
            "Input map size (%d) doesn't match model (%d)." % (x.shape[1], self.npix))
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


def _drop_path_schedule(params):
    return [float(v) for v in np.linspace(0, params.drop_path_rate, sum(params.depths))]


class HealSwinEncoder(nnx.Module):
    """Compression-only backbone: patch embed + encoder stages + final norm.
    Standalone-usable (tokenizer / embedder); allocates no decoder parameters."""

    def __init__(self, params, *, rngs):
        self.params = params
        self.num_layers = len(params.depths)
        self.num_features = int(params.embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(params, rngs=rngs)
        self.pos_drop = nnx.Dropout(params.drop_rate, rngs=rngs)

        num_patches = self.patch_embed.num_patches
        dpr = _drop_path_schedule(params)
        layers = []
        for i in range(self.num_layers):
            layers.append(EncoderStage(
                params, dim=int(params.embed_dim * 2 ** i),
                input_resolution=num_patches // 4 ** i,
                depth=params.depths[i], num_heads=params.num_heads[i],
                drop_path=dpr[sum(params.depths[:i]):sum(params.depths[:i + 1])],
                downsample=i < self.num_layers - 1, rngs=rngs))
        self.layers = nnx.List(layers)
        self.norm = nnx.LayerNorm(self.num_features, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        skips = []
        for layer in self.layers:
            skips.append(x)
            x = layer(x)
        return self.norm(x), skips


class HealSwinDecoder(nnx.Module):
    """UNet decoder head producing dense per-pixel outputs."""

    def __init__(self, params, *, rngs):
        self.num_layers = len(params.depths)
        num_patches = params.npix // params.patch_size
        dpr = _drop_path_schedule(params)
        layers_up = []
        concat_back_dim = []
        for i_layer in range(self.num_layers):
            down_idx = self.num_layers - 1 - i_layer
            dim = int(params.embed_dim * 2 ** down_idx)
            concat_back_dim.append(
                nnx.Linear(2 * dim, dim, kernel_init=TRUNC_NORMAL, rngs=rngs)
                if i_layer > 0 else Identity())
            if i_layer == 0:
                layers_up.append(PatchExpand(dim=dim, dim_scale=2, rngs=rngs))
            else:
                layers_up.append(DecoderStage(
                    params, dim=dim, input_resolution=num_patches // 4 ** down_idx,
                    depth=params.depths[down_idx], num_heads=params.num_heads[down_idx],
                    drop_path=dpr[sum(params.depths[:down_idx]):
                                  sum(params.depths[:down_idx + 1])],
                    upsample=down_idx > 0, rngs=rngs))
        self.layers_up = nnx.List(layers_up)
        self.concat_back_dim = nnx.List(concat_back_dim)
        self.norm_up = nnx.LayerNorm(params.embed_dim, epsilon=LN_EPS, rngs=rngs)
        self.up = FinalPatchExpand(patch_size=params.patch_size, dim=params.embed_dim,
                                   rngs=rngs)
        self.output = nnx.Conv(params.embed_dim, params.out_channels, kernel_size=(1,),
                               use_bias=False, rngs=rngs)

    def __call__(self, x, skips):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = jnp.concatenate([x, skips[self.num_layers - 1 - inx]], axis=-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)
        x = self.up(x)
        return self.output(x)  # (B, npix, out_channels) channels-last


class HealSwin(nnx.Module):
    """HEALPix Swin V2 U-Net: HealSwinEncoder + HealSwinDecoder."""

    def __init__(self, params, *, rngs):
        self.encoder = HealSwinEncoder(params, rngs=rngs)
        self.decoder = HealSwinDecoder(params, rngs=rngs)

    def __call__(self, x):
        tokens, skips = self.encoder(x)
        return self.decoder(tokens, skips)
```

Deliberate divergences from the deleted `swin_hp_transformer.py` (all spec-approved): no `ape`, no `use_v2_norm_placement`/`use_cos_attn` branches (V2 always), `qk_scale` gone, rel-bias table init is `TRUNC_NORMAL` (was zeros only to mirror a commented-out reference line), `return_intermediates` dropped (parity-only hook), decoder `norm_up` applied before `up` exactly as before.

- [ ] **Step 5: Delete the old HP model and re-point `__init__.py`**

```bash
git rm -q src/heal_swin_nnx/swin_hp_transformer.py
```

Replace `src/heal_swin_nnx/__init__.py` with (flat model still on the old API until Task 6):

```python
from heal_swin_nnx.config import DataSpec, SwinTransformerConfig
from heal_swin_nnx.models.healswin import (
    HealSwin, HealSwinDecoder, HealSwinEncoder, HealSwinParams)
from heal_swin_nnx.swin_transformer import SwinEncoder, SwinTransformerSys, UnetDecoder
from heal_swin_nnx.variables import Buffer

__all__ = ["Buffer", "DataSpec", "HealSwin", "HealSwinDecoder", "HealSwinEncoder",
           "HealSwinParams", "SwinEncoder", "SwinTransformerConfig",
           "SwinTransformerSys", "UnetDecoder"]
```

- [ ] **Step 6: Run the suite**

```bash
uv run pytest tests/ -q
```

Expected: all pass. If `test_remat_matches_no_remat` or jit tests fail with tracer errors, the bug is in `WindowAttention.__call__` reading Buffers without `[...]` — every Buffer/Param read must go through `[...]`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat!: HealSwin family — V2-only attention, pos_embed switch incl. RoPE, params threading

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: SwinUnet (flat) model family

Mirror treatment for the flat model in `models/swin.py`: `SwinParams`, V2-only, same `pos_embed` enum (default `rel_bias`), renamed classes. Deletes `swin_transformer.py` and `config.py`; final `__init__.py`.

**Files:**
- Create: `src/heal_swin_nnx/models/swin.py`
- Delete: `src/heal_swin_nnx/swin_transformer.py`, `src/heal_swin_nnx/config.py`
- Modify: `src/heal_swin_nnx/__init__.py`, `tests/test_params.py` (append), `tests/test_rope.py` (append), `tests/test_buffers.py` (replace config test)

**Interfaces:**
- Consumes: `LN_EPS`, `TRUNC_NORMAL`, `DropPath`, `Identity`, `Mlp`, `apply_rope`, `init_rope_freqs`, `rope_rotation_table` from `heal_swin_nnx.layers`; `Buffer` from `heal_swin_nnx.variables`. (Nothing from `hp/` — flat model is HEALPix-free.)
- Produces:
  - `SwinParams(img_size, in_channels, out_channels, patch_size=(4, 4), window_size=(4, 4), embed_dim=96, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), mlp_ratio=4.0, qkv_bias=True, pos_embed="rel_bias", rope_theta=10.0, use_masking=True, patch_embed_norm=False, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, use_checkpoint=False)` — int `img_size`/`patch_size`/`window_size` coerced to 2-tuples; properties `patches_resolution` and `shift_size` (= `(w0 // 2, w1 // 2)`).
  - `SwinEncoder(params, *, rngs)` — `__call__(x: (B, H, W, C)) -> (tokens, skips)`; `SwinDecoder(params, *, rngs)` — `__call__(tokens, skips) -> (B, H, W, out_channels)`; `SwinUnet(params, *, rngs)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_params.py`:

```python
from heal_swin_nnx.models.swin import SwinParams


def test_swin_params_defaults_and_coercion():
    # default depths (2, 2, 2, 2) -> img must divide patch*window*2^(L-1) = 128
    p = SwinParams(img_size=(128, 256), in_channels=3, out_channels=5)
    assert p.patch_size == (4, 4) and p.window_size == (4, 4)
    assert p.shift_size == (2, 2)
    assert p.patches_resolution == (32, 64)
    assert p.pos_embed == "rel_bias"
    p2 = SwinParams(img_size=64, in_channels=1, out_channels=1, window_size=8,
                    depths=(2, 2), num_heads=(2, 4), embed_dim=16)
    assert p2.img_size == (64, 64) and p2.window_size == (8, 8)
    assert p2.shift_size == (4, 4)


def test_swin_params_serializable():
    import dataclasses, json
    json.dumps(dataclasses.asdict(SwinParams(img_size=(128, 128), in_channels=3,
                                             out_channels=5)))


def test_swin_params_rejects_indivisible_geometry():
    import pytest
    # H=60 not divisible by patch*window*2^(L-1) = 4*4*8
    with pytest.raises(ValueError):
        SwinParams(img_size=(60, 64), in_channels=1, out_channels=1)


def test_swin_params_rope_head_dim_check():
    import pytest
    with pytest.raises(ValueError):
        SwinParams(img_size=(32, 32), in_channels=1, out_channels=1, embed_dim=12,
                   depths=(2, 2), num_heads=(2, 4), pos_embed="rope_mixed")
    SwinParams(img_size=(32, 32), in_channels=1, out_channels=1, embed_dim=16,
               depths=(2, 2), num_heads=(2, 4), pos_embed="rope_mixed")
```

Append to `tests/test_rope.py`:

```python
def test_swinunet_forward_all_pos_embeds():
    from flax import nnx
    from heal_swin_nnx import SwinParams, SwinUnet
    for pos_embed in ["none", "rel_bias", "rope_axial", "rope_mixed"]:
        p = SwinParams(img_size=(32, 64), in_channels=2, out_channels=3,
                       embed_dim=16, depths=(2, 2), num_heads=(2, 4),
                       drop_path_rate=0.0, pos_embed=pos_embed)
        model = SwinUnet(p, rngs=nnx.Rngs(0))
        model.eval()
        y = model(jnp.ones((1, 32, 64, 2)))
        assert y.shape == (1, 32, 64, 3), pos_embed
        assert np.isfinite(np.asarray(y)).all(), pos_embed
```

Replace `tests/test_buffers.py` entirely (Toy tests unchanged; the config-defaults test now targets the new Params):

```python
import jax.numpy as jnp
from flax import nnx

from heal_swin_nnx import HealSwinParams, SwinParams
from heal_swin_nnx.variables import Buffer


class Toy(nnx.Module):
    def __init__(self):
        self.w = nnx.Param(jnp.ones((3,)))
        self.idx = Buffer(jnp.arange(3))

    def __call__(self):
        return (self.w * self.idx).sum()


def test_buffer_excluded_from_params():
    m = Toy()
    params = nnx.state(m, nnx.Param)
    flat = dict(nnx.to_flat_state(params))
    assert ("w",) in flat
    assert not any("idx" in path for path in flat)


def test_buffer_not_differentiated():
    m = Toy()
    grads = nnx.grad(lambda m: m())(m)  # default wrt=nnx.Param
    flat = dict(nnx.to_flat_state(grads))
    assert ("w",) in flat
    assert not any("idx" in path for path in flat)


def test_params_construct_with_defaults():
    hp = HealSwinParams(nside=16, in_channels=3, out_channels=5)
    assert hp.patch_size == 4 and hp.window_size == 4 and hp.shift_size == 2
    assert hp.shift_strategy == "nest_grid_shift_exact" and hp.pos_embed == "rope_mixed"
    flat = SwinParams(img_size=(128, 128), in_channels=3, out_channels=5)
    assert flat.patch_size == (4, 4) and flat.window_size == (4, 4)
    assert flat.shift_size == (2, 2)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_params.py tests/test_rope.py tests/test_buffers.py -q
```

Expected: FAIL — `ModuleNotFoundError`/`ImportError` on `heal_swin_nnx.models.swin` and `SwinParams`.

- [ ] **Step 3: Create `src/heal_swin_nnx/models/swin.py`**

```python
"""Flat 2D Swin V2 U-Net (SwinUnet) — the planar sibling of HealSwin."""
from dataclasses import dataclass
from typing import Literal, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx

from heal_swin_nnx.layers import (
    LN_EPS, TRUNC_NORMAL, DropPath, Identity, Mlp, apply_rope, init_rope_freqs,
    rope_rotation_table)
from heal_swin_nnx.models.healswin import POS_EMBEDS
from heal_swin_nnx.variables import Buffer


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


@dataclass
class SwinParams:
    """Pure-data description of a flat SwinUnet model. Serializable."""

    img_size: Union[int, Tuple[int, int]]
    in_channels: int
    out_channels: int

    patch_size: Union[int, Tuple[int, int]] = (4, 4)
    window_size: Union[int, Tuple[int, int]] = (4, 4)
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 2, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    pos_embed: Literal["none", "rel_bias", "rope_axial", "rope_mixed"] = "rel_bias"
    rope_theta: float = 10.0
    use_masking: bool = True
    patch_embed_norm: bool = False

    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    use_checkpoint: bool = False

    def __post_init__(self):
        self.img_size = _pair(self.img_size)
        self.patch_size = _pair(self.patch_size)
        self.window_size = _pair(self.window_size)
        self.depths = tuple(self.depths)
        self.num_heads = tuple(self.num_heads)

        if self.pos_embed not in POS_EMBEDS:
            raise ValueError("pos_embed must be one of %r, got %r"
                             % (POS_EMBEDS, self.pos_embed))
        if len(self.depths) != len(self.num_heads):
            raise ValueError("depths (%d) and num_heads (%d) must have equal length"
                             % (len(self.depths), len(self.num_heads)))
        merge = 2 ** (len(self.depths) - 1)
        for a in range(2):
            div = merge * self.patch_size[a] * self.window_size[a]
            if self.img_size[a] % div:
                raise ValueError(
                    "img_size[%d]=%d must be divisible by patch*window*2^(n_stages-1)=%d"
                    % (a, self.img_size[a], div))
        for i, heads in enumerate(self.num_heads):
            dim = self.embed_dim * 2 ** i
            if dim % heads:
                raise ValueError("stage %d: dim %d not divisible by num_heads %d"
                                 % (i, dim, heads))
            if self.pos_embed in ("rope_axial", "rope_mixed") and (dim // heads) % 4:
                raise ValueError(
                    "stage %d: head_dim %d must be divisible by 4 for RoPE"
                    % (i, dim // heads))

    @property
    def patches_resolution(self):
        return (self.img_size[0] // self.patch_size[0],
                self.img_size[1] // self.patch_size[1])

    @property
    def shift_size(self):
        return (self.window_size[0] // 2, self.window_size[1] // 2)


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


def flat_win_coords(window_size):
    """(2, Wh*Ww) float32: (x, y) of each token in a row-major-flattened
    (Wh, Ww) window. x = column, y = row (rope-vit init_t_xy convention)."""
    n = np.arange(window_size[0] * window_size[1])
    return np.stack([(n % window_size[1]).astype(np.float32),
                     (n // window_size[1]).astype(np.float32)])


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


class WindowAttention(nnx.Module):
    """Swin V2 window attention (flat 2D windows), positional encoding
    selected by ``params.pos_embed``."""

    def __init__(self, params, dim, num_heads, window_size, *, rngs):
        self.window_size = tuple(window_size)
        self.num_heads = num_heads
        self.pos_embed = params.pos_embed
        head_dim = dim // num_heads
        self.logit_scale = nnx.Param(jnp.log(10.0 * jnp.ones((num_heads, 1, 1))))

        if self.pos_embed == "rel_bias":
            n_rel = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
            self.relative_position_bias_table = nnx.Param(
                TRUNC_NORMAL(rngs.params(), (n_rel, num_heads)))
            self.relative_position_index = Buffer(
                jnp.asarray(flat_relative_position_index(self.window_size)))
        elif self.pos_embed in ("rope_axial", "rope_mixed"):
            coords = jnp.asarray(flat_win_coords(self.window_size))
            if self.pos_embed == "rope_mixed":
                self.rope_freqs = nnx.Param(init_rope_freqs(
                    head_dim, num_heads, params.rope_theta, key=rngs.params()))
                self.rope_coords = Buffer(coords)
            else:
                freqs = init_rope_freqs(head_dim, num_heads, params.rope_theta)
                self.rope_table = Buffer(rope_rotation_table(freqs, coords[0], coords[1]))

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=params.qkv_bias,
                              kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.attn_drop = nnx.Dropout(params.attn_drop_rate, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.proj_drop = nnx.Dropout(params.drop_rate, rngs=rngs)

    def __call__(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
        k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-12)
        if self.pos_embed == "rope_mixed":
            coords = self.rope_coords[...]
            table = rope_rotation_table(self.rope_freqs[...], coords[0], coords[1])
            q, k = apply_rope(q, k, table)
        elif self.pos_embed == "rope_axial":
            q, k = apply_rope(q, k, self.rope_table[...])
        attn = q @ k.swapaxes(-2, -1)
        logit_scale = jnp.exp(jnp.minimum(self.logit_scale[...], jnp.log(1.0 / 0.01)))
        attn = attn * logit_scale

        if self.pos_embed == "rel_bias":
            ws_area = self.window_size[0] * self.window_size[1]
            bias = self.relative_position_bias_table[...][
                self.relative_position_index[...].reshape(-1)].reshape(ws_area, ws_area, -1)
            attn = attn + bias.transpose(2, 0, 1)[None]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + mask[None, :, None]
            attn = attn.reshape(-1, self.num_heads, N, N)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).swapaxes(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinBlock(nnx.Module):
    def __init__(self, params, dim, input_resolution, num_heads, shifted, drop_path,
                 *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.window_size = tuple(params.window_size)
        shift_size = params.shift_size if shifted else (0, 0)
        if (self.input_resolution[0] <= self.window_size[0]
                or self.input_resolution[1] <= self.window_size[1]):
            shift_size = (0, 0)
            self.window_size = self.input_resolution
        self.shift_size = shift_size

        self.norm1 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.attn = WindowAttention(params, dim, num_heads, self.window_size, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * params.mlp_ratio), drop=params.drop_rate, rngs=rngs)

        if params.use_masking and (shift_size[0] > 0 or shift_size[1] > 0):
            self.attn_mask = Buffer(jnp.asarray(flat_shift_mask(
                self.input_resolution, self.window_size, shift_size)))
        else:
            self.attn_mask = None

    def __call__(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.reshape(B, H, W, C)

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            shifted_x = jnp.roll(x, (-self.shift_size[0], -self.shift_size[1]), axis=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size[0] * self.window_size[1], C)
        mask = None if self.attn_mask is None else self.attn_mask[...]
        attn_windows = self.attn(x_windows, mask=mask)
        attn_windows = attn_windows.reshape(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x = jnp.roll(shifted_x, (self.shift_size[0], self.shift_size[1]), axis=(1, 2))
        else:
            x = shifted_x
        x = x.reshape(B, H * W, C)

        x = shortcut + self.drop_path(self.norm1(x))
        return x + self.drop_path(self.norm2(self.mlp(x)))


class PatchMerging(nnx.Module):
    def __init__(self, input_resolution, dim, *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.reduction = nnx.Linear(4 * dim, 2 * dim, use_bias=False,
                                    kernel_init=TRUNC_NORMAL, rngs=rngs)
        self.norm = nnx.LayerNorm(4 * dim, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W and H % 2 == 0 and W % 2 == 0
        x = x.reshape(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = jnp.concatenate([x0, x1, x2, x3], axis=-1).reshape(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class PatchExpand(nnx.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, *, rngs):
        self.input_resolution = tuple(input_resolution)
        self.expand = (nnx.Linear(dim, 2 * dim, use_bias=False, kernel_init=TRUNC_NORMAL,
                                  rngs=rngs) if dim_scale == 2 else Identity())
        self.norm = nnx.LayerNorm(dim // dim_scale, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.reshape(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        return self.norm(x.reshape(B, -1, C // 4))


class FinalPatchExpand(nnx.Module):
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
    def __init__(self, params, *, rngs):
        self.img_size = params.img_size
        self.num_patches = params.patches_resolution[0] * params.patches_resolution[1]
        self.proj = nnx.Conv(params.in_channels, params.embed_dim,
                             kernel_size=tuple(params.patch_size),
                             strides=tuple(params.patch_size), padding="VALID", rngs=rngs)
        self.norm = (nnx.LayerNorm(params.embed_dim, epsilon=LN_EPS, rngs=rngs)
                     if params.patch_embed_norm else None)

    def __call__(self, x):  # (B, H, W, in_channels) channels-last
        B, H, W, C = x.shape
        assert (H, W) == self.img_size
        x = self.proj(x)                   # (B, Ph, Pw, embed_dim)
        x = x.reshape(B, -1, x.shape[-1])  # row-major flatten
        if self.norm is not None:
            x = self.norm(x)
        return x


def _make_blocks(params, dim, input_resolution, depth, num_heads, drop_path, rngs):
    return [SwinBlock(params, dim, input_resolution, num_heads,
                      shifted=(i % 2 == 1), drop_path=drop_path[i], rngs=rngs)
            for i in range(depth)]


class EncoderStage(nnx.Module):
    def __init__(self, params, dim, input_resolution, depth, num_heads, drop_path,
                 downsample, *, rngs):
        self.use_checkpoint = params.use_checkpoint
        self.blocks = nnx.List(_make_blocks(params, dim, input_resolution, depth,
                                            num_heads, drop_path, rngs))
        self.downsample = (PatchMerging(input_resolution, dim=dim, rngs=rngs)
                           if downsample else None)

    def __call__(self, x):
        for blk in self.blocks:
            x = nnx.remat(type(blk).__call__)(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class DecoderStage(nnx.Module):
    def __init__(self, params, dim, input_resolution, depth, num_heads, drop_path,
                 upsample, *, rngs):
        self.use_checkpoint = params.use_checkpoint
        self.blocks = nnx.List(_make_blocks(params, dim, input_resolution, depth,
                                            num_heads, drop_path, rngs))
        self.upsample = (PatchExpand(input_resolution, dim=dim, dim_scale=2, rngs=rngs)
                         if upsample else None)

    def __call__(self, x):
        for blk in self.blocks:
            x = nnx.remat(type(blk).__call__)(blk, x) if self.use_checkpoint else blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


def _drop_path_schedule(params):
    return [float(v) for v in np.linspace(0, params.drop_path_rate, sum(params.depths))]


class SwinEncoder(nnx.Module):
    def __init__(self, params, *, rngs):
        self.params = params
        self.num_layers = len(params.depths)
        self.num_features = int(params.embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(params, rngs=rngs)
        self.pos_drop = nnx.Dropout(params.drop_rate, rngs=rngs)

        pr = params.patches_resolution
        dpr = _drop_path_schedule(params)
        layers = []
        for i in range(self.num_layers):
            layers.append(EncoderStage(
                params, dim=int(params.embed_dim * 2 ** i),
                input_resolution=(pr[0] // 2 ** i, pr[1] // 2 ** i),
                depth=params.depths[i], num_heads=params.num_heads[i],
                drop_path=dpr[sum(params.depths[:i]):sum(params.depths[:i + 1])],
                downsample=i < self.num_layers - 1, rngs=rngs))
        self.layers = nnx.List(layers)
        self.norm = nnx.LayerNorm(self.num_features, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        skips = []
        for layer in self.layers:
            skips.append(x)
            x = layer(x)
        return self.norm(x), skips


class SwinDecoder(nnx.Module):
    def __init__(self, params, *, rngs):
        self.params = params
        self.num_layers = len(params.depths)
        pr = params.patches_resolution
        dpr = _drop_path_schedule(params)
        layers_up = []
        concat_back_dim = []
        for i_layer in range(self.num_layers):
            down_idx = self.num_layers - 1 - i_layer
            dim = int(params.embed_dim * 2 ** down_idx)
            res = (pr[0] // 2 ** down_idx, pr[1] // 2 ** down_idx)
            concat_back_dim.append(
                nnx.Linear(2 * dim, dim, kernel_init=TRUNC_NORMAL, rngs=rngs)
                if i_layer > 0 else Identity())
            if i_layer == 0:
                layers_up.append(PatchExpand(res, dim=dim, dim_scale=2, rngs=rngs))
            else:
                layers_up.append(DecoderStage(
                    params, dim=dim, input_resolution=res, depth=params.depths[down_idx],
                    num_heads=params.num_heads[down_idx],
                    drop_path=dpr[sum(params.depths[:down_idx]):
                                  sum(params.depths[:down_idx + 1])],
                    upsample=i_layer < self.num_layers - 1, rngs=rngs))
        self.layers_up = nnx.List(layers_up)
        self.concat_back_dim = nnx.List(concat_back_dim)
        self.norm_up = nnx.LayerNorm(params.embed_dim, epsilon=LN_EPS, rngs=rngs)
        self.up = FinalPatchExpand(pr, patch_size=params.patch_size,
                                   dim=params.embed_dim, rngs=rngs)
        self.output = nnx.Conv(params.embed_dim, params.out_channels, kernel_size=(1, 1),
                               use_bias=False, rngs=rngs)

    def __call__(self, x, skips):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = jnp.concatenate([x, skips[self.num_layers - 1 - inx]], axis=-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)
        x = self.up(x)  # (B, H*W, embed_dim) at full resolution
        H, W = self.params.img_size
        x = x.reshape(x.shape[0], H, W, x.shape[-1])
        return self.output(x)  # (B, H, W, out_channels)


class SwinUnet(nnx.Module):
    """Flat 2D Swin V2 U-Net: SwinEncoder + SwinDecoder."""

    def __init__(self, params, *, rngs):
        self.encoder = SwinEncoder(params, rngs=rngs)
        self.decoder = SwinDecoder(params, rngs=rngs)

    def __call__(self, x):
        tokens, skips = self.encoder(x)
        return self.decoder(tokens, skips)
```

Deliberate divergences from the deleted `swin_transformer.py`: the double-`shift_size[0]` roll bug is fixed (was bug-for-bug with the reference; irrelevant for the square windows the derived shift produces, wrong for rectangular ones); `final_upsample`, `ape`, `qk_scale`, V1 attention, and `return_intermediates` are gone; the rel-bias table is only allocated when `pos_embed == "rel_bias"` (the reference allocated it always).

- [ ] **Step 4: Delete the old flat model + config; final `__init__.py`**

```bash
git rm -q src/heal_swin_nnx/swin_transformer.py src/heal_swin_nnx/config.py
```

Replace `src/heal_swin_nnx/__init__.py` with:

```python
from heal_swin_nnx.models.healswin import (
    HealSwin, HealSwinDecoder, HealSwinEncoder, HealSwinParams)
from heal_swin_nnx.models.swin import SwinDecoder, SwinEncoder, SwinParams, SwinUnet
from heal_swin_nnx.variables import Buffer

__all__ = ["Buffer", "HealSwin", "HealSwinDecoder", "HealSwinEncoder", "HealSwinParams",
           "SwinDecoder", "SwinEncoder", "SwinParams", "SwinUnet"]
```

- [ ] **Step 5: Verify nothing references the deleted modules**

```bash
grep -rn "heal_swin_nnx.config\|swin_transformer\|SwinTransformerSys\|SwinHPTransformer\|DataSpec\|UnetDecoder" src tests
```

Expected: no output.

- [ ] **Step 6: Run the suite**

```bash
uv run pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat!: SwinUnet family — SwinParams, V2-only, pos_embed switch; drop legacy config module

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Docs, packaging metadata, final verification

**Files:**
- Modify: `README.md`, `pyproject.toml` (description only)

**Interfaces:**
- Consumes: the final public API from Task 6.
- Produces: user-facing docs matching the shipped API.

- [ ] **Step 1: Update `pyproject.toml` description**

Replace the `description = ...` line with:

```toml
description = "HealSwin: spherical HEALPix Swin V2 U-Net in JAX/Flax NNX, with RoPE and seam-exact shifted windows"
```

- [ ] **Step 2: Rewrite the README's Usage, Full-sphere, and Tests sections**

Keep the overall structure; replace the old-API content:

- Intro sentence: drop "golden-value parity" phrasing; the port diverged deliberately after reaching parity (tag `parity-verified` holds the last parity-checked state).
- Usage example:

```python
import jax.numpy as jnp
from flax import nnx
from heal_swin_nnx import HealSwin, HealSwinParams

params = HealSwinParams(nside=16, in_channels=3, out_channels=5,
                        embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
model = HealSwin(params, rngs=nnx.Rngs(0))

x = jnp.ones((2, params.npix, params.in_channels))  # channels-last: (B, N, C)
y = model(x)                                        # (B, params.npix, params.out_channels)
```

- Note that `HealSwinParams` is pure serializable data (`json.dumps(dataclasses.asdict(params))`) and lists the notable defaults: `shift_strategy="nest_grid_shift_exact"`, `pos_embed="rope_mixed"`.
- Positional-encoding section (new): one paragraph — `pos_embed` is one of `"none"`, `"rel_bias"` (flat relative-position bias table), `"rope_axial"`, `"rope_mixed"` (rope-vit-style rotary embeddings computed on intra-window coordinates; `rope_mixed` learns per-head frequencies). RoPE requires head dims divisible by 4.
- Flat-model paragraph: `SwinParams(img_size=(H, W), ...)` + `SwinUnet`, inputs `(B, H, W, C)`; encoders/decoders standalone (`HealSwinEncoder`/`SwinEncoder` return `(tokens, skips)`).
- Public API list: the nine names from the Global Constraints.
- Full-sphere / partial-coverage example updated:

```python
from heal_swin_nnx import HealSwinParams

params = HealSwinParams(nside=256, in_channels=1, out_channels=1,
                        base_pixels=(8, 9, 10, 11))  # south polar cap
```

  (keep the existing shift-strategy bullet list — it is still accurate).
- Tests section: remove all mentions of bit-exact/golden/parity tests; describe the suite as ground-truth geometry checks (healpy adjacency, permutation/round-trip invariants), RoPE property tests, and JAX-native behavior tests. Keep `uv run pytest tests/ -q`.

- [ ] **Step 3: Check for leftover stale references**

```bash
grep -rn "SwinHPTransformer\|SwinTransformerSys\|DataSpec\|dim_in\|f_in\|f_out\|parity" README.md docs/superpowers/specs/2026-07-12-config-cleanup-design.md --include="*.md" -l
grep -rn "parity\|golden" README.md
```

Expected: the spec file may match (it documents the history — fine); README must have no matches except, optionally, one mention of the `parity-verified` tag.

- [ ] **Step 4: Full verification**

```bash
uv run pytest tests/ -q
uv run python -c "
import dataclasses, json
import jax.numpy as jnp
from flax import nnx
import heal_swin_nnx as h
assert sorted(h.__all__) == ['Buffer', 'HealSwin', 'HealSwinDecoder', 'HealSwinEncoder',
                             'HealSwinParams', 'SwinDecoder', 'SwinEncoder', 'SwinParams',
                             'SwinUnet']
p = h.HealSwinParams(nside=16, in_channels=2, out_channels=3, embed_dim=16,
                     depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0)
json.dumps(dataclasses.asdict(p))
m = h.HealSwin(p, rngs=nnx.Rngs(0)); m.eval()
assert m(jnp.ones((1, p.npix, 2))).shape == (1, p.npix, 3)
print('OK')
"
```

Expected: all tests pass; script prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: README and packaging for the diverged HealSwin API

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
