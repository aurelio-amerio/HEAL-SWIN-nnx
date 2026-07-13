# Spherical MNIST classification with HealSwin — design

**Date:** 2026-07-13
**Status:** approved (brainstorming), pending implementation plan

## Goal

An `examples/` demo that trains a HealSwin-based classifier on spherical MNIST
at `nside=64` and reports test accuracy. MNIST digits are ray-traced onto
full-sphere HEALPix maps (NEST order) under random rotations; the model must
classify the digit (10 classes). Train on 100,000 projected samples, validate on
10,000 fixed test samples.

This is an **example**, not library code — it lives entirely under `examples/`
and adds no new `src/` code and no new pytest suite. Validation is a smoke check
plus an actual training run that reaches high test accuracy.

## Decisions (from brainstorming)

- **Data generation:** on-the-fly (lazy) projection inside grain workers — no
  up-front RAM materialization. (Materializing 1e5 maps at nside 64 would be
  ~20 GB.)
- **Epoch semantics:** *fixed specs, re-projected.* A fixed list of
  `(mnist_index, euler_angles, delta_theta, delta_phi)` specs is pre-drawn once
  from a seeded RNG; every epoch re-projects the **same** deterministic maps. No
  per-epoch augmentation. Fully reproducible per seed.
- **Deliverable:** a self-contained Python script (`mnist_healpix_classify.py`),
  runnable headless via `uv run --extra examples`.
- **Model scale:** lightweight demo backbone — `embed_dim=48`,
  `depths=(2,2,2)`, `num_heads=(3,6,12)` (3 stages).
- **Classifier head:** encoder-only + global mean-pool + linear (see below); the
  U-Net decoder is not used.
- **Shared node:** grain worker count is capped at 8 (matches the sbibm-jax
  `_MAX_WORKERS_CAP` convention for this cluster).

## Components

### 1. `examples/mnist_healpix_dataset.py` — refactor to lazy projection

Split "what to project" (cheap specs, held in RAM) from "the projected map"
(expensive, computed lazily per record).

- Load MNIST once via HuggingFace `datasets` (`images`, `labels`, ~37 MB).
- Pre-draw a **fixed** spec list of length `num_samples` from one seeded
  `np.random.default_rng(seed)`:
  - `idx` — `mnist_index`, drawn with replacement so `num_samples` may exceed the
    split size.
  - `delta_theta`, `delta_phi` — each uniform in `delta_range` (degrees).
  - `angles` — three Euler angles, uniform in `[0, 360)` (same convention as the
    notebook's `rot=(120., 45., 10.)`; not measure-uniform on SO(3), adequate for
    a demo).
  Each spec is a handful of floats + an int, so 1e5 specs is negligible RAM.
- Build a picklable projection callable (a small class or `functools.partial`)
  holding references to `images`, `labels`, `nside`. Given a spec it runs
  `img2healpix(..., nest=True)` and returns
  `{"image": (npix,) float32, "label": int}`.
- Return `grain.MapDataset.source(specs).map(project_fn)` — **unbatched,
  unshuffled**. The caller composes shuffle/batch/prefetch. `.map` is lazy, so
  the ray-trace runs per-record inside the consuming iterator/workers.
- Keep the `__main__` smoke check (build a small dataset, pull one record, print
  shape/label).

Public signature stays:
`make_mnist_healpix_dataset(num_samples, nside=64, split="train", delta_range=(50., 100.), seed=0) -> grain.MapDataset`.

**Note on the projection callable being picklable:** under `mp_prefetch` grain
pickles the transform into each worker, copying the `images` array (~37 MB per
worker × ≤8 workers). Acceptable. The callable uses only numpy + healpy (no JAX),
so workers touch no accelerator.

### 2. `examples/mnist_healpix_classify.py` — classifier + training driver

**Model** — a small `nnx.Module`, `HealSwinClassifier`:

```
x (B, npix, 1)
  -> HealSwinEncoder(params)      # returns (tokens, skips); skips discarded
  -> tokens (B, N_bottleneck, D)  # encoder already applies its final LayerNorm
  -> mean over token axis         # (B, D)
  -> nnx.Linear(D, num_classes)   # (B, 10) logits
```

`params = HealSwinParams(nside=64, in_channels=1, out_channels=10,
embed_dim=48, depths=(2,2,2), num_heads=(3,6,12))`, defaults otherwise
(`pos_embed="rope_mixed"`, `shift_strategy="nest_grid_shift_exact"`, full sphere
= all 12 base pixels). `out_channels` is required by the dataclass but unused by
the head. `D = embed_dim * 2**(n_stages-1) = 48 * 4 = 192`;
`N_bottleneck = 12 * ((nside**2 / patch_size) / 4**(n_stages-1)) = 12 * 64 = 768`.

This config passes every `HealSwinParams.__post_init__` rule (verified):
nside is a power of two; `nside²%patch_size=0`; `(nside²/patch_size)%4^(stages-1)
= 1024%16 = 0`; window_size 4 is a power of four; each stage `dim%heads=0`
(48/3, 96/6, 192/12 = 16).

**Data pipeline** (per split, mirroring sbibm-jax):

```
ds = make_mnist_healpix_dataset(N, nside=64, split=..., seed=...)
pipe = (ds.shuffle(seed).to_iter_dataset()
          .batch(batch_size)
          .mp_prefetch(grain.multiprocessing.MultiprocessingOptions(num_workers=W)))
```

- Train: `N=100_000`, `split="train"`, shuffle on, `seed=0`.
- Test: `N=10_000`, `split="test"`, **no shuffle**, `seed=1`, one pass per eval.
- `W = min(8, os.cpu_count()-2)` or similar; capped at 8.
- Batches arrive as numpy dict `{"image": (B, npix), "label": (B,)}`; the train
  step adds the channel axis → `(B, npix, 1)` and casts to the model dtype.

**Training loop** — pure NNX/optax, streamlined (referenced GenSBI but written
fresh):

- Optimizer: AdamW with a cosine-decay learning-rate schedule (with short warmup).
- Loss: `optax.softmax_cross_entropy_with_integer_labels`.
- `nnx.Optimizer(model, optax_tx)`; `train_step` computed via `nnx.value_and_grad`,
  wrapped in `nnx.jit`. `eval_step` (logits → argmax) also jitted.
- Loop a fixed number of epochs (each epoch = one pass over the 100k specs). After
  each epoch, run the full 10k test set and print epoch, mean train loss, and test
  accuracy.
- All tunables (`nside`, `batch_size`, `epochs`, `lr`, `embed_dim`, `depths`,
  `num_heads`, `num_workers`, seeds) are module-level constants near the top.

**Determinism / RNG:** dataset RNG is seeded per split; model init uses
`nnx.Rngs(seed)`; dropout defaults (`drop_rate=0`, `attn_drop_rate=0`,
`drop_path_rate=0.1`) mean the model has stochastic drop-path at train time —
call the model in train vs eval mode accordingly (`model.train()` /
`model.eval()`), or set `drop_path_rate=0` for a fully deterministic demo (decide
at implementation; default keep 0.1 with proper train/eval toggling).

## Data flow

```
seed ─► specs [(idx, angles, dθ, dφ)] ─► grain source
                                           │ .map(project)  (lazy, in workers)
                                           ▼
        {"image": (npix,), "label": int} ─► shuffle ─► batch ─► mp_prefetch
                                           ▼
        {"image": (B, npix), "label": (B,)} numpy ─► add channel ─► (B, npix, 1)
                                           ▼
        HealSwinClassifier ─► logits (B, 10) ─► CE loss / argmax accuracy
```

## Error handling

- Example-level: rely on library validation (`HealSwinParams.__post_init__`,
  grain, healpy). No custom error framework.
- Worker robustness: projection is deterministic and total; a bad spec would
  raise and surface through grain — acceptable for a demo.

## Testing / validation

No new pytest suite (this is an example). Validation is:

1. `uv run --extra examples python examples/mnist_healpix_dataset.py` — smoke check
   builds a tiny lazy dataset and pulls one projected record.
2. A short training run (few epochs, possibly reduced N) that shows test accuracy
   climbing well above chance, confirming the pipeline trains end-to-end. The
   author confirms an actual run before declaring done.

## Out of scope (YAGNI)

- Precompute-to-disk caching, infinite streaming, per-epoch re-randomized poses.
- Reusing the U-Net decoder; measure-uniform SO(3) sampling; checkpointing /
  logging frameworks; CLI arg parsing (module constants suffice).
```

## Files touched

- `examples/mnist_healpix_dataset.py` — refactored (lazy).
- `examples/mnist_healpix_classify.py` — new.
- No `src/` changes, no new tests.
