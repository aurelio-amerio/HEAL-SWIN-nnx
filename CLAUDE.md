# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                  # install (uv-managed project, Python >= 3.12)
uv run pytest tests/ -q                  # run all tests (pytest is configured: JAX_PLATFORMS=cpu, -n 2 via xdist)
uv run pytest tests/test_model.py -q     # one file
uv run pytest tests/test_rope.py::test_name -q   # one test
```

Tests always run on CPU (`JAX_PLATFORMS=cpu` is set in `pyproject.toml`), even though the package depends on `jax[cuda12]`.

## What this is

A Flax NNX port of HEAL-SWIN — a HEALPix-native Swin Transformer U-Net — plus the flat-grid Swin U-Net it shares code with. Pure JAX/Flax: **no PyTorch imports anywhere in `src/`**. The port reached verified torch-parity (git tag `parity-verified` preserves that state) and then deliberately diverged: both models are Swin V2-only (cosine attention, post-norm), and the HEALPix model defaults to rotary positional embeddings (`pos_embed="rope_mixed"`) and seam-exact shifted windows (`shift_strategy="nest_grid_shift_exact"`).

`references/` holds read-only upstream checkouts consulted during the port (HEAL-SWIN reference implementation, Map2Patches, rope-vit) — never edit or import from them at runtime.

## Architecture

Public API (all in `src/heal_swin_nnx/__init__.py`): `HealSwin`/`HealSwinEncoder`/`HealSwinDecoder`/`HealSwinParams` (spherical), `SwinUnet`/`SwinEncoder`/`SwinDecoder`/`SwinParams` (flat grid), and `Buffer`.

- `models/healswin.py` — the spherical model. `HealSwinParams` is a pure-data dataclass (JSON-serializable via `dataclasses.asdict`) that does all validation in `__post_init__`; new config knobs belong there, with a validation rule and a `test_params.py` case. Input/output is channels-last `(B, npix, C)` in NEST order over the selected base pixels.
- `models/swin.py` — the flat-grid counterpart, mirrored structure, `(B, H, W, C)`.
- `hp/topology.py` — HEALPix face adjacency/orientation ground work (construction-time numpy). **Must never import healpy**: healpy is the independent ground truth the test suite checks against, so using it here would be circular. Per-pixel Python loops are fine in this module — only the model forward path is performance-critical.
- `hp/shifting.py` — the four shift strategies (`nest_roll`, `nest_grid_shift`, `nest_grid_shift_exact`, `ring_shift`) as index permutations + attention masks.
- `hp/windowing.py` — window partition/reverse and intra-window NEST coordinates.
- `layers.py` — shared building blocks: Mlp, DropPath, RoPE (frequency init, rotation tables, `apply_rope`).
- `variables.py` — `Buffer`, an `nnx.Variable` subclass for non-trainable index/mask state. All precomputed permutations, masks, and position indices must be stored as `Buffer` (not `nnx.Param`) so optimizers and `nnx.grad` never touch them while they still travel through `nnx.split`/`nnx.jit`.

Key invariant threading both models: geometry (permutations, masks, RoPE coordinates) is computed once at construction time in numpy and stored in Buffers; the traced forward pass only does gathers and dense math. Models support partial sky coverage via `base_pixels` (strictly increasing subset of the 12 HEALPix faces); `params.npix = len(base_pixels) * nside**2`.

## Testing philosophy

Tests verify against independent ground truth (healpy adjacency for geometry) and via properties/invariants (permutation round-trips, jit/eager equivalence, batch independence, remat equivalence) rather than golden values. Follow that style: a new shift strategy or positional encoding gets a geometry/invariant test, not a snapshot.

## Design docs

`docs/superpowers/specs/` and `docs/superpowers/plans/` hold the design specs and implementation plans this codebase was built from (port design, config cleanup + RoPE, full-sphere extension). Consult them for rationale before changing module boundaries or config semantics.
