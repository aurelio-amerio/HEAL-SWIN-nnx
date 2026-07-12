# HEAL-SWIN-nnx

A [Flax NNX](https://flax.readthedocs.io/en/latest/nnx/index.html) port of
[HEAL-SWIN](references/HEAL-SWIN) (a HEALPix-native Swin Transformer U-Net),
plus the standard flat-grid Swin Transformer it shares code with. Pure
JAX/Flax; no PyTorch import anywhere in `src/`. The port reached verified
torch-parity (git tag `parity-verified` holds that state) and then diverged
deliberately — both models are now Swin V2-only (cosine attention,
post-norm); the HEALPix model additionally defaults to rotary positional
embeddings and seam-exact shifted windows.

## Install

```bash
uv sync
```

## Usage

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

`HealSwinParams` is pure, serializable data — `json.dumps(dataclasses.asdict(params))`
works, so a run's exact configuration can be logged and compared. Notable
defaults: `shift_strategy="nest_grid_shift_exact"`, `pos_embed="rope_mixed"`.

### Positional encoding

`pos_embed` selects one of `"none"`, `"rel_bias"` (flat relative-position bias
table), `"rope_axial"`, or `"rope_mixed"` (rope-vit-style rotary embeddings
computed on intra-window coordinates; `"rope_mixed"` additionally learns
per-head rotation frequencies). RoPE requires head dims divisible by 4.
`HealSwinParams` defaults to `"rope_mixed"`; `SwinParams` defaults to
`"rel_bias"`.

### Flat-grid model

The flat-grid counterpart mirrors this shape: `SwinParams(img_size=(H, W), ...)`
+ `SwinUnet`, with inputs/outputs as `(B, H, W, C)`. Both models expose their
encoder (`HealSwinEncoder` / `SwinEncoder`, returning `(tokens, skips)`) and
decoder (`HealSwinDecoder` / `SwinDecoder`) standalone, so the encoder can be
used without ever building decoder parameters.

Public API: `HealSwin`, `HealSwinEncoder`, `HealSwinDecoder`, `HealSwinParams`,
`SwinUnet`, `SwinEncoder`, `SwinDecoder`, `SwinParams`, `Buffer` (the
`nnx.Variable` subclass used for non-trainable index/mask state, excluded from
`nnx.Param` filters).

## Full sphere and partial coverage

Models cover the full sphere by default (all 12 HEALPix base pixels). Experiments
that only see part of the sky select the base pixels they cover — e.g. a
ground-based south-pole telescope observing the four southern faces:

```python
from heal_swin_nnx import HealSwinParams

params = HealSwinParams(nside=256, in_channels=1, out_channels=1,
                        base_pixels=(8, 9, 10, 11))  # south polar cap
```

Inputs are the concatenation of the selected faces' NEST-ordered pixels.
Shift strategies:

- `nest_roll` — 1D roll on the NEST sequence (cheapest, coarsest).
- `nest_grid_shift` — the reference HEAL-SWIN hierarchical grid shift; face-seam
  windows that glue geometrically wrong edges are attention-masked.
- `nest_grid_shift_exact` — seam-exact variant: window content crosses face seams
  with the correct pixels and orientation wherever the two faces' local frames
  align (all polar-to-equatorial seams). Attention masking remains at the 8 pinch
  points, at the 90°-rotated south-south seams, and at coverage borders for
  partial-sky models.
- `ring_shift` — shift along HEALPix iso-latitude rings; exact on the full sphere.

## Tests

```bash
uv run pytest tests/ -q
```

Covers: ground-truth geometry checks against healpy adjacency
(`test_topology.py`, `test_seam_geometry.py`); permutation/round-trip
invariants for shifting and windowing (`test_shifting.py`,
`test_windowing.py`); RoPE property tests — coordinate-frame round-trips and
rotation-table invariants (`test_rope.py`); param validation and
serialization (`test_params.py`); buffer/param separation
(`test_buffers.py`); and JAX-native behavior — jit/eager equivalence, batch
independence, `nnx.remat` matching non-remat, standalone encoder, and all
HEALPix shift strategies (`test_model.py`).

## Design docs

- [`docs/superpowers/specs/2026-07-12-healswin-nnx-port-design.md`](docs/superpowers/specs/2026-07-12-healswin-nnx-port-design.md) —
  original port design spec (module surface, config shape, buffer strategy,
  shift-strategy semantics).
- [`docs/superpowers/plans/2026-07-12-healswin-nnx-port.md`](docs/superpowers/plans/2026-07-12-healswin-nnx-port.md) —
  the task-by-task implementation plan the port followed.
- [`docs/superpowers/specs/2026-07-12-config-cleanup-design.md`](docs/superpowers/specs/2026-07-12-config-cleanup-design.md) —
  design spec for the config unification, cleanup, and RoPE work that followed.
