# -*- coding: utf-8 -*-
"""Train a lightweight HealSwin classifier on spherical MNIST (nside 64).

MNIST digits are ray-traced onto full-sphere HEALPix maps under random
rotations (see :mod:`mnist_healpix_dataset`). A HEALPix-native Swin encoder
compresses each map to bottleneck tokens; mean-pooling + a linear head predict
the digit class. Trains on 100k projected samples, validates on 10k fixed test
samples.

Run headless. The script defaults to the GPU (``JAX_PLATFORMS=cuda``) and will
fail fast on a machine with no CUDA device — set ``JAX_PLATFORMS=cpu`` to force
CPU. Spawned grain data-loader workers are always pinned to CPU (see below).

    uv run --extra examples python examples/mnist_healpix_classify.py

Or submit it to a GPU node via HTCondor: ``examples/sub/mnist_healpix_classify.sub``.
"""

from __future__ import annotations

import os
import sys

# grain's mp_prefetch spawns worker processes that re-import this module
# (multiprocessing "spawn"). Force those workers onto CPU so they never try to
# grab the GPU the main process is training on — the workers only do numpy +
# healpy projection and touch no accelerator. The main process defaults to the
# GPU; an explicit JAX_PLATFORMS from the caller (e.g. the SMOKE check below)
# still wins via setdefault.
if __name__ != "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ.setdefault("JAX_PLATFORMS", "cuda")

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from absl import flags

import grain

from heal_swin_nnx import HealSwinEncoder, HealSwinParams
from mnist_healpix_dataset import make_mnist_healpix_dataset

# grain's mp_prefetch reads absl flags (e.g. --grain_enable_multiprocess_worker_
# profiling). Running this script as a plain `python ...` (not through
# absl.app.run) leaves them unparsed, so grain raises UnparsedFlagAccessError on
# the first prefetch. Parse argv once here to mark the flags parsed.
if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv, known_only=True)

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
    return correct / max(total, 1)


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
