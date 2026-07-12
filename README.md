# HEAL-SWIN-nnx

A [Flax NNX](https://flax.readthedocs.io/en/latest/nnx/index.html) port of
[HEAL-SWIN](references/HEAL-SWIN) (a HEALPix-native Swin Transformer U-Net),
plus the standard flat-grid Swin Transformer it shares code with. Pure
JAX/Flax; no PyTorch import anywhere in `src/`.

## Install

```bash
uv sync
```

## Usage

```python
import jax.numpy as jnp
from flax import nnx
from heal_swin_nnx import DataSpec, SwinHPTransformerConfig, SwinHPTransformerSys

cfg = SwinHPTransformerConfig(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])
ds = DataSpec(dim_in=8 * 16 ** 2, f_in=3, f_out=5, base_pix=8)  # npix = base_pix * nside^2
model = SwinHPTransformerSys(cfg, ds, rngs=nnx.Rngs(0))

x = jnp.ones((2, ds.dim_in, ds.f_in))   # channels-last: (B, N, C)
y = model(x)                            # (B, ds.dim_in, ds.f_out)
```

The flat-grid counterpart mirrors this shape: `SwinTransformerConfig` +
`DataSpec(dim_in=(H, W), ...)` + `SwinTransformerSys`, with inputs/outputs as
`(B, H, W, C)`. Both models expose their encoder (`SwinHPEncoder` /
`SwinEncoder`, returning `(tokens, skips)`) and decoder (`HPUnetDecoder` /
`UnetDecoder`) standalone, so the encoder can be used without ever building
decoder parameters.

Public API: `SwinHPTransformerSys`, `SwinHPEncoder`, `HPUnetDecoder`,
`SwinTransformerSys`, `SwinEncoder`, `UnetDecoder`, `SwinHPTransformerConfig`,
`SwinTransformerConfig`, `DataSpec`, `Buffer` (the `nnx.Variable` subclass
used for non-trainable index/mask state, excluded from `nnx.Param` filters).

## Tests

```bash
uv run pytest tests/ -q
```

Covers: bit-exact HEALPix shifting/windowing indices and masks against the
reference (`test_shifting.py`, `test_windowing.py`), buffer/param separation
(`test_buffers.py`), module- and full-model forward + gradient parity against
golden fixtures generated from the pinned reference implementation
(`test_parity_modules.py`, `test_parity_e2e.py`), and JAX-native behavior —
jit/eager equivalence, batch independence, `nnx.remat` matching non-remat,
standalone encoder, all three HEALPix shift strategies (`base_pix=12` with
`nest_roll`, `NotImplementedError` for `nest_grid_shift`), and buffers never
leaking into `nnx.Param` state (`test_model.py`).

## Parity

Golden fixtures (`tests/goldens/*.npz`) are generated once from the pinned
legacy reference environment (Python 3.8, torch 1.8.0+cpu, timm 0.4.12,
healpy 1.15.2) and committed; the main test suite only reads them, it never
needs torch. Forward outputs match to `rtol=atol=1e-4`; gradients to
`rtol=1e-3, atol=1e-4` (three ill-conditioned float32 cases — `hp_ring`,
`hp_cos_v2`, `flat_cos_v2` — are documented and individually loosened in
`tests/test_parity_e2e.py`). `tests/test_parity_f64.py` reruns those three
sensitive cases (plus two controls) against float64 goldens under
`jax_enable_x64`, matching to ~1e-9; since both implementations agree that
tightly once precision noise is removed, the loosened float32 tolerances are
confirmed to be precision-only, not an algorithmic gap. See
[`parity/README.md`](parity/README.md) for how the fixtures are (re)generated
and the reference clamp-bug patch that generation requires.

## Design docs

- [`docs/superpowers/specs/2026-07-12-healswin-nnx-port-design.md`](docs/superpowers/specs/2026-07-12-healswin-nnx-port-design.md) —
  design spec (module surface, config/DataSpec shape, buffer strategy,
  shift-strategy semantics, parity approach).
- [`docs/superpowers/plans/2026-07-12-healswin-nnx-port.md`](docs/superpowers/plans/2026-07-12-healswin-nnx-port.md) —
  the task-by-task implementation plan this port followed.
