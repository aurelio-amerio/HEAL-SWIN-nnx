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

## Examples

The spherical GRF simulation-based-inference example (HealSwin encoder +
gensbi Flux1 flow matching) lives in
[GenSBI-examples](https://github.com/aurelio-amerio/GenSBI-examples) under
`examples/sbi-benchmarks/spherical_grf/`.

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
  windows that glue geometrically wrong edges are attention-masked. Its index math
  requires the deepest stage to hold a full window (bottleneck `nside² ≥ window_size`);
  `HealSwinParams` rejects configs that bottleneck below that. The other three
  strategies handle a unit bottleneck.
- `nest_grid_shift_exact` — seam-exact variant: window content crosses face seams
  with the correct pixels and orientation wherever the two faces' local frames
  align (all polar-to-equatorial seams). Attention masking remains at the 8 pinch
  points, at the 90°-rotated south-south seams, and at coverage borders for
  partial-sky models.
- `ring_shift` — shift along HEALPix iso-latitude rings; exact on the full sphere.

## Strategy cost

`scripts/bench_strategies.py` times each strategy two ways — the shift op in
isolation and a full `HealSwin` forward+backward step — so you can pick a
strategy on geometry, not guesswork:

```bash
uv run python scripts/bench_strategies.py           # full sphere, nside 16 and 64
```

Numbers below are from one CPU run (jax 0.10.2, full sphere, `window_size=4`,
`batch=2`); treat them as ratios, not absolutes. **fwd+bwd** is the timed
forward+backward; **build** is the one-time construction cost, paid once when the
model is built and done host-side in NumPy/healpy.

Shift op alone (a gather forward + its scatter backward), channels=96:

| strategy                | fwd+bwd n16 | fwd+bwd n64 | build n16 | build n64 |
| ----------------------- | ----------: | ----------: | --------: | --------: |
| `nest_roll`             |      0.5 ms |     26.6 ms |     11 ms |     16 ms |
| `nest_grid_shift`       |      1.3 ms |     45.4 ms |     30 ms |    270 ms |
| `nest_grid_shift_exact` |      1.4 ms |     45.9 ms |    788 ms |   19.1 s  |
| `ring_shift`            |      1.4 ms |     46.3 ms |      7 ms |    128 ms |

Full model (`embed_dim=48`, `depths=(2,2,2)`, `num_heads=(2,4,8)`):

| strategy                | fwd+bwd n16 | fwd+bwd n64 | build n16 | build n64 |
| ----------------------- | ----------: | ----------: | --------: | --------: |
| `nest_roll`             |      111 ms |      806 ms |    178 ms |    213 ms |
| `nest_grid_shift`       |      112 ms |      841 ms |    198 ms |    415 ms |
| `nest_grid_shift_exact` |      113 ms |      819 ms |    562 ms |    9.7 s  |
| `ring_shift`            |      128 ms |      815 ms |    183 ms |    347 ms |

Takeaways:

- **Per-step compute barely depends on the strategy.** In the full model all four
  land within ~4% at nside 64 and ~13% at nside 16 — the shift is a thin slice of
  a Swin block, dwarfed by attention and MLPs. Choose the strategy for geometric
  fidelity; it costs almost nothing at run time.
- **The three index strategies are the same runtime op.** `nest_grid_shift`,
  `nest_grid_shift_exact` and `ring_shift` are each a single `jnp.take` gather over
  a precomputed index buffer, so in isolation they time within ~2% of one another.
  `nest_roll` is a contiguous `jnp.roll` and runs ~1.7–2.5× cheaper on CPU (no
  scatter/gather) — a gap that mostly washes out once attention dominates.
- **The real divergence is one-time build cost.** `nest_roll` (just a mask) and
  `ring_shift` (healpy round-trips) are cheap; `nest_grid_shift` is cheap; but
  `nest_grid_shift_exact`'s seam geometry is expensive and scales steeply with
  `nside` (~0.8 s at 16 → ~19 s at 64 per shifter). It is still paid only once at
  construction — negligible against a real training run, but noticeable when
  building many small models (tests, quick sweeps).

On GPU the run-time gaps shrink further (gather/scatter and roll are all cheap
on-device); build cost is host-side and unchanged.

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
