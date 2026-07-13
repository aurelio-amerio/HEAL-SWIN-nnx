# Spherical MNIST Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `examples/` demo that trains a lightweight HealSwin classifier on spherical MNIST (nside 64) and reports test accuracy.

**Architecture:** Refactor the existing dataset module to project MNIST digits onto HEALPix maps *lazily* inside grain workers (fixed pre-drawn specs, re-projected each epoch, no RAM materialization). A new driver script wraps `HealSwinEncoder` with a mean-pool + linear head and trains it with a streamlined NNX/optax loop.

**Tech Stack:** JAX / Flax NNX, optax, grain (lazy `MapDataset` + `mp_prefetch`), HuggingFace `datasets`, healpy (projection only, in workers).

## Global Constraints

- No new `src/` code and no new pytest suite — everything lives under `examples/`.
- No PyTorch anywhere.
- Grain worker count capped at 8 (shared node; matches sbibm-jax `_MAX_WORKERS_CAP`).
- Run everything via `uv run --extra examples ...` (the `examples` extra provides grain/datasets/scipy/matplotlib).
- Maps are channels-last `(B, npix, C)` in **NEST** order, full sphere: `npix = 12 * nside**2 = 49152` at nside 64.
- Model config is fixed lightweight: `embed_dim=48, depths=(2,2,2), num_heads=(3,6,12)`; derived `D = 192`, `N_bottleneck = 768`.
- Reproducible per seed: dataset RNG seeded per split; model init via `nnx.Rngs(seed)`.

---

### Task 1: Refactor `mnist_healpix_dataset.py` to lazy projection

**Files:**
- Modify (full rewrite): `examples/mnist_healpix_dataset.py`

**Interfaces:**
- Consumes: `projections.img2healpix(img, nside, delta_theta, delta_phi, rot, nest=True) -> (map, hits)` (already in `examples/projections.py`); `datasets.load_dataset`; `grain`.
- Produces: `make_mnist_healpix_dataset(num_samples, nside=64, split="train", delta_range=(50.0, 100.0), seed=0) -> grain.MapDataset` whose records are `{"image": (npix,) float32, "label": int}` in NEST order, **unbatched and unshuffled**. Also a module-level picklable `_Projector` callable.

- [ ] **Step 1: Rewrite the module for lazy projection**

Replace the entire contents of `examples/mnist_healpix_dataset.py` with:

```python
# -*- coding: utf-8 -*-
"""Lazy grain dataset of MNIST digits projected onto HEALPix maps.

Each record is a single MNIST digit ray-traced onto a full-sphere HEALPix map
(NEST ordering, via the local :mod:`projections` module) under a fixed random
rotation and angular extent. A fixed list of ``num_samples`` projection *specs*
(digit index, Euler angles, angular extents) is pre-drawn once from a seeded
RNG; the expensive ray-trace runs **lazily** inside grain workers, so the same
deterministic maps are produced every epoch without materializing them in RAM.

Digits are sampled with replacement, so ``num_samples`` may exceed the MNIST
split size.

Run headless as a smoke check:

    uv run --extra examples python examples/mnist_healpix_dataset.py
"""

from __future__ import annotations

import numpy as np
import healpy as hp
from datasets import load_dataset

from projections import img2healpix


class _Projector:
    """Picklable per-record projection: spec index -> {image, label}.

    Holds the shared MNIST arrays and the pre-drawn spec arrays. grain pickles
    this into each ``mp_prefetch`` worker (copying ``images`` ~37 MB/worker).
    Uses only numpy + healpy, so workers touch no accelerator.
    """

    def __init__(self, images, labels, idx, angles, delta_theta, delta_phi, nside):
        self.images = images
        self.labels = labels
        self.idx = idx
        self.angles = angles
        self.delta_theta = delta_theta
        self.delta_phi = delta_phi
        self.nside = nside

    def __call__(self, i: int):
        j = int(self.idx[i])
        rot = hp.rotator.Rotator(rot=tuple(self.angles[i]))
        hp_map, _hits = img2healpix(
            self.images[j],
            nside=self.nside,
            delta_theta=float(self.delta_theta[i]),
            delta_phi=float(self.delta_phi[i]),
            rot=rot,
            nest=True,
        )
        return {"image": hp_map.astype(np.float32), "label": int(self.labels[j])}


def make_mnist_healpix_dataset(
    num_samples: int,
    nside: int = 64,
    split: str = "train",
    delta_range: tuple[float, float] = (50.0, 100.0),
    seed: int = 0,
):
    """Build a lazy grain dataset of HEALPix-projected MNIST digits.

    Args:
        num_samples: Number of records. May exceed the split size (digits are
            drawn with replacement).
        nside: HEALPix ``NSIDE`` of the output maps (``npix = 12 * nside**2``).
        split: MNIST split to draw from (``"train"`` or ``"test"``).
        delta_range: ``(low, high)`` degrees; ``delta_theta`` and ``delta_phi``
            are each drawn independently and uniformly from this range.
        seed: Seed for the single numpy RNG driving indices, extents, and
            rotation angles (fully reproducible per seed).

    Returns:
        A ``grain.MapDataset`` of ``{"image": (npix,) float32, "label": int}``
        records in NEST order, unbatched and unshuffled. Compose
        ``.shuffle(seed).to_iter_dataset().batch(...).mp_prefetch(...)`` to feed
        a training loop.
    """
    import grain

    rng = np.random.default_rng(seed)

    mnist = load_dataset("ylecun/mnist")[split]
    images = np.asarray(mnist["image"], dtype=np.float64)  # (M, 28, 28)
    labels = np.asarray(mnist["label"], dtype=np.int64)    # (M,)
    n_mnist = images.shape[0]

    # Fixed specs, pre-drawn once (tiny in RAM; the maps stay lazy).
    idx = rng.integers(0, n_mnist, size=num_samples)
    delta_theta = rng.uniform(delta_range[0], delta_range[1], size=num_samples)
    delta_phi = rng.uniform(delta_range[0], delta_range[1], size=num_samples)
    # Three Euler angles per sample (same form as the notebook's
    # ``rot=(120., 45., 10.)``); uniform in [0, 360) is not measure-uniform on
    # SO(3) but is adequate for a demo.
    angles = rng.uniform(0.0, 360.0, size=(num_samples, 3))

    projector = _Projector(images, labels, idx, angles, delta_theta, delta_phi, nside)
    return grain.MapDataset.range(num_samples).map(projector)


if __name__ == "__main__":
    ds = make_mnist_healpix_dataset(num_samples=8, nside=64, seed=0)
    rec = ds[0]
    print("dataset length:", len(ds))
    print("image shape:", rec["image"].shape, rec["image"].dtype)
    print("label:", rec["label"])
    assert rec["image"].shape == (12 * 64 ** 2,)
    assert rec["image"].dtype == np.float32
    assert 0 <= rec["label"] <= 9
    print("smoke check OK")
```

- [ ] **Step 2: Run the smoke check and verify it passes**

Run: `uv run --extra examples python examples/mnist_healpix_dataset.py`
Expected (after the MNIST download/cache): prints `dataset length: 8`, `image shape: (49152,) float32`, a `label:` in 0-9, and ends with `smoke check OK` (no assertion error).

- [ ] **Step 3: Commit**

```bash
git add examples/mnist_healpix_dataset.py
git commit -m "refactor(examples): lazy grain projection for MNIST->HEALPix dataset

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Classifier + training driver `mnist_healpix_classify.py`

**Files:**
- Create: `examples/mnist_healpix_classify.py`

**Interfaces:**
- Consumes: `make_mnist_healpix_dataset(...)` (Task 1); `heal_swin_nnx.HealSwinEncoder`, `heal_swin_nnx.HealSwinParams`; `flax.nnx`, `optax`, `grain`.
- Produces: a runnable `__main__` training script (no importable public API required).

Key verified facts to rely on:
- `HealSwinEncoder(params, *, rngs)`; `encoder(x)` returns `(tokens, skips)` where `tokens` is `(B, N_bottleneck, D)` already LayerNorm'd; discard `skips`.
- `nnx.Optimizer(model, tx, wrt=nnx.Param)`; `optimizer.update(model, grads)`.
- grain `.batch()` on dict records auto-stacks → `{"image": (B, npix), "label": (B,)}` numpy arrays.
- `model.train()` / `model.eval()` toggle `DropPath.deterministic`.

- [ ] **Step 1: Write the classifier module and a forward-shape smoke check**

Create `examples/mnist_healpix_classify.py`:

```python
# -*- coding: utf-8 -*-
"""Train a lightweight HealSwin classifier on spherical MNIST (nside 64).

MNIST digits are ray-traced onto full-sphere HEALPix maps under random
rotations (see :mod:`mnist_healpix_dataset`). A HEALPix-native Swin encoder
compresses each map to bottleneck tokens; mean-pooling + a linear head predict
the digit class. Trains on 100k projected samples, validates on 10k fixed test
samples.

Run headless (GPU recommended):

    uv run --extra examples python examples/mnist_healpix_classify.py
"""

from __future__ import annotations

import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import grain

from heal_swin_nnx import HealSwinEncoder, HealSwinParams
from mnist_healpix_dataset import make_mnist_healpix_dataset

# --- config (tune here) --------------------------------------------------
NSIDE = 64
NUM_CLASSES = 10
TRAIN_SAMPLES = 100_000
TEST_SAMPLES = 10_000
BATCH_SIZE = 128
EPOCHS = 10
PEAK_LR = 3e-4
WEIGHT_DECAY = 0.05
WARMUP_FRAC = 0.05
EMBED_DIM = 48
DEPTHS = (2, 2, 2)
NUM_HEADS = (3, 6, 12)
NUM_WORKERS = min(8, max(1, (os.cpu_count() or 2) - 2))
SEED = 0
# ------------------------------------------------------------------------


class HealSwinClassifier(nnx.Module):
    """HealSwin encoder + mean-pool over tokens + linear classification head."""

    def __init__(self, params: HealSwinParams, num_classes: int, *, rngs: nnx.Rngs):
        self.encoder = HealSwinEncoder(params, rngs=rngs)
        self.head = nnx.Linear(self.encoder.num_features, num_classes, rngs=rngs)

    def __call__(self, x):  # x: (B, npix, in_channels)
        tokens, _skips = self.encoder(x)          # (B, N_bottleneck, D)
        pooled = jnp.mean(tokens, axis=1)          # (B, D)
        return self.head(pooled)                   # (B, num_classes)


def make_params() -> HealSwinParams:
    return HealSwinParams(
        nside=NSIDE,
        in_channels=1,
        out_channels=NUM_CLASSES,  # required by dataclass; unused by the head
        embed_dim=EMBED_DIM,
        depths=DEPTHS,
        num_heads=NUM_HEADS,
    )


if __name__ == "__main__" and os.environ.get("SMOKE") == "1":
    # Forward-shape smoke check: no data, just a random map.
    model = HealSwinClassifier(make_params(), NUM_CLASSES, rngs=nnx.Rngs(0))
    model.eval()
    npix = 12 * NSIDE ** 2
    x = jnp.zeros((2, npix, 1), dtype=jnp.float32)
    logits = model(x)
    print("logits shape:", logits.shape)
    assert logits.shape == (2, NUM_CLASSES)
    print("forward smoke check OK")
```

- [ ] **Step 2: Run the forward-shape smoke check and verify it passes**

Run: `SMOKE=1 JAX_PLATFORMS=cpu uv run --extra examples python examples/mnist_healpix_classify.py`
Expected: prints `logits shape: (2, 10)` then `forward smoke check OK`, no assertion error. (This also proves the lightweight config passes every `HealSwinParams` validation rule.)

- [ ] **Step 3: Add the data loaders and training loop**

Append to `examples/mnist_healpix_classify.py`, **above** the `if __name__ == "__main__" and ... SMOKE` block (so the smoke block stays last), the following helpers:

```python
def make_loader(ds, batch_size, num_workers, shuffle_seed=None):
    """Compose a lazy grain dataset into a batched, prefetched iterator.

    shuffle_seed=None -> no shuffle (deterministic order, for eval).
    """
    pipe = ds
    if shuffle_seed is not None:
        pipe = pipe.shuffle(shuffle_seed)
    pipe = pipe.to_iter_dataset().batch(batch_size)
    if num_workers:
        pipe = pipe.mp_prefetch(grain.MultiprocessingOptions(num_workers=num_workers))
    return pipe


def to_model_inputs(batch):
    """grain numpy batch -> (images (B, npix, 1) float32, labels (B,) int32)."""
    images = jnp.asarray(batch["image"], dtype=jnp.float32)[..., None]
    labels = jnp.asarray(batch["label"], dtype=jnp.int32)
    return images, labels


@nnx.jit
def train_step(model, optimizer, images, labels):
    def loss_fn(model):
        logits = model(images)
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, images):
    return model(images).argmax(axis=-1)


def evaluate(model, test_ds):
    model.eval()
    loader = make_loader(test_ds, BATCH_SIZE, NUM_WORKERS, shuffle_seed=None)
    correct = 0
    total = 0
    for batch in loader:
        images, labels = to_model_inputs(batch)
        preds = eval_step(model, images)
        correct += int((preds == labels).sum())
        total += int(labels.shape[0])
    model.train()
    return correct / total


def main():
    print(f"workers={NUM_WORKERS} batch={BATCH_SIZE} epochs={EPOCHS} "
          f"nside={NSIDE} embed_dim={EMBED_DIM} depths={DEPTHS}")

    train_ds = make_mnist_healpix_dataset(TRAIN_SAMPLES, nside=NSIDE, split="train", seed=SEED)
    test_ds = make_mnist_healpix_dataset(TEST_SAMPLES, nside=NSIDE, split="test", seed=SEED + 1)

    steps_per_epoch = math.ceil(TRAIN_SAMPLES / BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=PEAK_LR,
        warmup_steps=int(WARMUP_FRAC * total_steps),
        decay_steps=total_steps, end_value=PEAK_LR * 0.01,
    )
    tx = optax.adamw(schedule, weight_decay=WEIGHT_DECAY)

    model = HealSwinClassifier(make_params(), NUM_CLASSES, rngs=nnx.Rngs(SEED))
    model.train()
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    for epoch in range(EPOCHS):
        t0 = time.time()
        loader = make_loader(train_ds, BATCH_SIZE, NUM_WORKERS, shuffle_seed=SEED + epoch)
        running = 0.0
        nsteps = 0
        for batch in loader:
            images, labels = to_model_inputs(batch)
            loss = train_step(model, optimizer, images, labels)
            running += float(loss)
            nsteps += 1
        acc = evaluate(model, test_ds)
        print(f"epoch {epoch:2d}  train_loss {running / max(nsteps, 1):.4f}  "
              f"test_acc {acc:.4f}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__" and os.environ.get("SMOKE") != "1":
    main()
```

- [ ] **Step 4: Run a short reduced training run and verify accuracy climbs**

The script's bare imports (`from mnist_healpix_dataset import ...`, and `from projections import ...` inside the dataset module) resolve because Python puts the script's own directory (`examples/`) on `sys.path[0]` when you run it by path — so `uv run --extra examples python examples/mnist_healpix_classify.py` works with no packaging.

Verify the loop end-to-end on a small budget: **temporarily** edit the config constants to `TRAIN_SAMPLES = 2000`, `TEST_SAMPLES = 1000`, `EPOCHS = 2`, then run:

`uv run --extra examples python examples/mnist_healpix_classify.py`

Expected: two `epoch` lines print; `test_acc` on epoch 1 is clearly above chance (> 0.30, typically much higher) and not decreasing on epoch 2. Then **restore** the constants to `TRAIN_SAMPLES = 100_000`, `TEST_SAMPLES = 10_000`, `EPOCHS = 10` before committing.

- [ ] **Step 5: Commit**

```bash
git add examples/mnist_healpix_classify.py
git commit -m "feat(examples): HealSwin spherical MNIST classifier + training loop

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Full validation run (final acceptance)**

Run the full training (GPU recommended; this is the real deliverable):

`uv run --extra examples python examples/mnist_healpix_classify.py`

Expected: 10 `epoch` lines; `test_acc` rises across epochs and reaches a high value (well above chance) on the 10k fixed test set. Record the final accuracy in the commit message or a short note. If it trains but underperforms, tune `PEAK_LR` / `EPOCHS` (no structural change needed).

---

## Self-Review

**Spec coverage:**
- Lazy on-the-fly projection in grain workers → Task 1 (`_Projector` + `.map`, `mp_prefetch` in Task 2 loader). ✓
- Fixed specs, re-projected each epoch → Task 1 pre-draws specs once; deterministic per seed. ✓
- Standalone script deliverable → Task 2. ✓
- Lightweight backbone (48 / (2,2,2) / (3,6,12)) → `make_params`. ✓
- Encoder + mean-pool + linear head → `HealSwinClassifier`. ✓
- Train 100k / validate 10k fixed test → constants + `evaluate`. ✓
- 8-worker cap → `NUM_WORKERS = min(8, ...)`. ✓
- Streamlined NNX/optax loop (AdamW + warmup-cosine, softmax CE, jitted steps) → Task 2. ✓
- No new pytest suite; smoke checks + training run as validation → Steps 2 & 4/6. ✓
- Train/eval toggling for drop-path → `model.train()`/`model.eval()` around steps and eval. ✓

**Placeholder scan:** Step 4 describes a temporary-constants verification (an explicit, bounded manual edit-and-restore), not a code placeholder. All code steps contain complete code. No TBD/TODO in shipped code.

**Type consistency:** `make_mnist_healpix_dataset` signature and record shape match between tasks; `HealSwinEncoder(params, *, rngs)` and `(tokens, skips)` return match verified source; `optimizer.update(model, grads)` and `nnx.Optimizer(..., wrt=nnx.Param)` match flax 0.12.7; `encoder.num_features` used for head width matches the encoder attribute.
```
