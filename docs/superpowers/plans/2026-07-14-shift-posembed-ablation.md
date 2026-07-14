# Shift × pos-embed ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parameterized HealSwin MNIST training script plus a single HTCondor submit that sweeps 6 (shift_strategy, pos_embed) configs, each writing a per-epoch CSV.

**Architecture:** One new example script `examples/mnist_healpix_ablation.py` (a copy of `mnist_healpix_classify.py` with `argparse` for `--pos-embed`/`--shift-strategy` and CSV output), one shared run wrapper, and one `.sub` that queues the 6 deduped configs over a `from ( ... )` list. No library/model changes.

**Tech Stack:** JAX/Flax NNX, optax, grain, `uv run`, HTCondor. Existing example scripts under `examples/` and submit files under `examples/sub/`.

## Global Constraints

- No PyTorch imports anywhere in `src/` (this plan touches only `examples/`, so N/A but do not add torch).
- Tests/local checks run on CPU (`JAX_PLATFORMS=cpu`). Training is submitted to GPU nodes, never run locally.
- The base `examples/mnist_healpix_classify.py` is left untouched.
- Config matrix is exactly the 6 deduped rows below; do not add `pos_embed="none"` or other combinations.
- Argparse choices: `--pos-embed` ∈ {`rel_bias`,`rope_axial`,`rope_mixed`}, `--shift-strategy` ∈ {`nest_roll`,`nest_grid_shift`,`nest_grid_shift_exact`,`ring_shift`}.
- Hyperparameters identical to the base script: nside 64, embed_dim 32, depths (2,2,6,2), num_heads (4,8,16,16), window_size 16, batch 128, epochs 50, seed 0.

The 6 deduped configs:

| # | shift_strategy         | pos_embed  |
|---|------------------------|------------|
| 1 | nest_roll              | rope_mixed |
| 2 | nest_grid_shift        | rope_mixed |
| 3 | nest_grid_shift_exact  | rope_mixed |
| 4 | ring_shift             | rope_mixed |
| 5 | nest_grid_shift_exact  | rel_bias   |
| 6 | nest_grid_shift_exact  | rope_axial |

---

## File Structure

- `examples/mnist_healpix_ablation.py` — **create.** Parameterized trainer; argparse → `HealSwinParams(pos_embed=..., shift_strategy=...)`; per-epoch CSV to `examples/ablation_results/<pos_embed>__<shift_strategy>.csv`. Retains the `SMOKE=1` forward-shape check (now requiring the two args).
- `examples/ablation_results/` — **create (dir).** Output CSVs land here; created at runtime by the script. A `.gitignore` keeps the dir tracked but its CSVs untracked.
- `examples/sub/run_mnist_healpix_ablation.sh` — **create.** Executable wrapper: `$1=repo_root $2=pos_embed $3=shift_strategy` → `uv run python examples/mnist_healpix_ablation.py --pos-embed $2 --shift-strategy $3`.
- `examples/sub/mnist_healpix_ablation_a100.sub` — **create.** A100 submit that queues the 6 rows with per-job logs.

---

### Task 1: Parameterized training script with CSV output

**Files:**
- Create: `examples/mnist_healpix_ablation.py`
- Verify with: `SMOKE=1 JAX_PLATFORMS=cpu uv run python examples/mnist_healpix_ablation.py --pos-embed <pe> --shift-strategy <ss>`

**Interfaces:**
- Consumes: `heal_swin_nnx.HealSwinEncoder`, `heal_swin_nnx.HealSwinParams`; local `mnist_healpix_dataset.make_mnist_healpix_dataset` (same dir).
- Produces: CLI `--pos-embed`, `--shift-strategy`; CSV file `examples/ablation_results/<pos_embed>__<shift_strategy>.csv` with comment header + columns `epoch,train_loss,test_acc,time_per_epoch_s`.

- [ ] **Step 1: Create the script**

Create `examples/mnist_healpix_ablation.py` with exactly this content:

```python
# -*- coding: utf-8 -*-
"""Ablation trainer: HealSwin spherical-MNIST classifier over (shift, pos_embed).

A parameterized copy of ``mnist_healpix_classify.py``. Takes ``--pos-embed`` and
``--shift-strategy`` and forwards them into ``HealSwinParams``; everything else
(nside 64, embed_dim 32, depths (2,2,6,2), 50 epochs, seed 0) is identical so
runs are directly comparable. Per-epoch metrics are written to a CSV under
``examples/ablation_results/`` for easy plotting.

Run headless. The script defaults to the GPU (``JAX_PLATFORMS=cuda``) and will
fail fast on a machine with no CUDA device — set ``JAX_PLATFORMS=cpu`` to force
CPU. Spawned grain data-loader workers are always pinned to CPU.

    uv run python examples/mnist_healpix_ablation.py \\
        --pos-embed rope_mixed --shift-strategy nest_roll

Or submit all 6 configs to A100 nodes via HTCondor:
``condor_submit examples/sub/mnist_healpix_ablation_a100.sub``.
"""

from __future__ import annotations

import os
import sys

# grain's mp_prefetch spawns worker processes that re-import this module
# (multiprocessing "spawn"). Force those workers onto CPU so they never try to
# grab the GPU the main process is training on — the workers only do numpy +
# healpy projection and touch no accelerator. The main process defaults to the
# GPU; an explicit JAX_PLATFORMS from the caller (e.g. the SMOKE check) still
# wins via setdefault.
if __name__ != "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ.setdefault("JAX_PLATFORMS", "cuda")

import argparse
import csv
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

# grain's mp_prefetch reads absl flags. Running this script as a plain
# `python ...` (not through absl.app.run) leaves them unparsed, so grain raises
# UnparsedFlagAccessError on the first prefetch. Parse argv once here (known_only
# so our own --pos-embed/--shift-strategy are ignored) to mark the flags parsed.
if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv, known_only=True)

POS_EMBEDS = ("rel_bias", "rope_axial", "rope_mixed")
SHIFT_STRATEGIES = ("nest_roll", "nest_grid_shift", "nest_grid_shift_exact", "ring_shift")

# --- config (tune here) --------------------------------------------------
NSIDE = 64
NUM_CLASSES = 10
TRAIN_SAMPLES = 100_000
TEST_SAMPLES = 10_000
BATCH_SIZE = 128
EPOCHS = 50
PEAK_LR = 3e-4
WEIGHT_DECAY = 0.05
WARMUP_FRAC = 0.05
EMBED_DIM = 32
DEPTHS = (2, 2, 6, 2)
NUM_HEADS = (4, 8, 16, 16)
WINDOW_SIZE = 16  # 4x4 windows; max allowed by the 4x4-per-face bottleneck of this 4-stage config
NUM_WORKERS = min(8, max(1, (os.cpu_count() or 2) - 2))
SEED = 0
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "ablation_results")
# ------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pos-embed", required=True, choices=POS_EMBEDS)
    p.add_argument("--shift-strategy", required=True, choices=SHIFT_STRATEGIES)
    # known_only: ignore absl/grain flags that may share argv.
    args, _unknown = p.parse_known_args()
    return args


class HealSwinClassifier(nnx.Module):
    """HealSwin encoder + mean-pool over tokens + linear classification head."""

    def __init__(self, params: HealSwinParams, num_classes: int, *, rngs: nnx.Rngs):
        self.encoder = HealSwinEncoder(params, rngs=rngs)
        self.head = nnx.Linear(self.encoder.num_features, num_classes, rngs=rngs)

    def __call__(self, x):  # x: (B, npix, in_channels)
        tokens, _skips = self.encoder(x)          # (B, N_bottleneck, D)
        pooled = jnp.mean(tokens, axis=1)          # (B, D)
        return self.head(pooled)                   # (B, num_classes)


def make_params(pos_embed: str, shift_strategy: str) -> HealSwinParams:
    return HealSwinParams(
        nside=NSIDE,
        in_channels=1,
        out_channels=NUM_CLASSES,  # required by dataclass; unused by the head
        embed_dim=EMBED_DIM,
        depths=DEPTHS,
        num_heads=NUM_HEADS,
        window_size=WINDOW_SIZE,
        pos_embed=pos_embed,
        shift_strategy=shift_strategy,
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


def main(args):
    pos_embed = args.pos_embed
    shift_strategy = args.shift_strategy
    config_str = (f"pos_embed={pos_embed} shift_strategy={shift_strategy} "
                  f"nside={NSIDE} embed_dim={EMBED_DIM} depths={DEPTHS} "
                  f"window_size={WINDOW_SIZE} batch={BATCH_SIZE} epochs={EPOCHS} "
                  f"seed={SEED}")
    print(config_str)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, f"{pos_embed}__{shift_strategy}.csv")
    results_file = open(results_path, "w", newline="")
    results_file.write("# " + config_str + "\n")
    writer = csv.writer(results_file)
    writer.writerow(["epoch", "train_loss", "test_acc", "time_per_epoch_s"])
    results_file.flush()

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

    model = HealSwinClassifier(make_params(pos_embed, shift_strategy), NUM_CLASSES,
                               rngs=nnx.Rngs(SEED))
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
        dt = time.time() - t0
        train_loss = running / max(nsteps, 1)
        print(f"epoch {epoch:2d}  train_loss {train_loss:.4f}  "
              f"test_acc {acc:.4f}  ({dt:.1f}s)")
        writer.writerow([epoch, f"{train_loss:.6f}", f"{acc:.6f}", f"{dt:.2f}"])
        results_file.flush()

    results_file.close()


if __name__ == "__main__":
    _args = parse_args()
    if os.environ.get("SMOKE") == "1":
        # Forward-shape smoke check: no data, just a random map.
        model = HealSwinClassifier(
            make_params(_args.pos_embed, _args.shift_strategy),
            NUM_CLASSES, rngs=nnx.Rngs(0))
        model.eval()
        npix = 12 * NSIDE ** 2
        x = jnp.zeros((2, npix, 1), dtype=jnp.float32)
        logits = model(x)
        print("logits shape:", logits.shape)
        assert logits.shape == (2, NUM_CLASSES)
        print("forward smoke check OK")
    else:
        main(_args)
```

- [ ] **Step 2: Verify the script fails fast on a bad choice**

Run: `JAX_PLATFORMS=cpu uv run python examples/mnist_healpix_ablation.py --pos-embed none --shift-strategy nest_roll`
Expected: argparse exits non-zero with `invalid choice: 'none'` (confirms choices are enforced).

- [ ] **Step 3: Run the SMOKE forward check for all 6 configs on CPU**

Run:
```bash
cd /lustre/ific.uv.es/ml/ific088/github/HEAL-SWIN-nnx
for cfg in "rope_mixed nest_roll" "rope_mixed nest_grid_shift" \
           "rope_mixed nest_grid_shift_exact" "rope_mixed ring_shift" \
           "rel_bias nest_grid_shift_exact" "rope_axial nest_grid_shift_exact"; do
  set -- $cfg
  echo "=== pos_embed=$1 shift=$2 ==="
  SMOKE=1 JAX_PLATFORMS=cpu uv run python examples/mnist_healpix_ablation.py \
      --pos-embed "$1" --shift-strategy "$2" || exit 1
done
```
Expected: each prints `logits shape: (2, 10)` then `forward smoke check OK`; loop exits 0.

- [ ] **Step 4: Create the results-dir gitignore so CSVs stay untracked**

Create `examples/ablation_results/.gitignore` with:
```
*.csv
```

- [ ] **Step 5: Commit**

```bash
git add examples/mnist_healpix_ablation.py examples/ablation_results/.gitignore
git commit -m "feat(examples): parameterized shift/pos-embed ablation trainer with CSV output"
```

---

### Task 2: HTCondor run wrapper + submit file

**Files:**
- Create: `examples/sub/run_mnist_healpix_ablation.sh`
- Create: `examples/sub/mnist_healpix_ablation_a100.sub`

**Interfaces:**
- Consumes: `examples/mnist_healpix_ablation.py` CLI from Task 1 (`--pos-embed`, `--shift-strategy`).
- Produces: a single `condor_submit` that launches 6 jobs, logs under `examples/sub/condor_logs/ablation_<pos_embed>_<shift_strategy>.{log,out,err}`.

- [ ] **Step 1: Create the run wrapper**

Create `examples/sub/run_mnist_healpix_ablation.sh` with exactly this content:

```bash
#!/bin/bash
# HTCondor executable wrapper for the HealSwin shift/pos-embed ablation.
#
#   $1 = repo root (the uv project directory containing pyproject.toml)
#   $2 = pos_embed       (rel_bias | rope_axial | rope_mixed)
#   $3 = shift_strategy  (nest_roll | nest_grid_shift | nest_grid_shift_exact | ring_shift)
#
# Runs the training through uv (grain / datasets / healpy are project deps).
set -euo pipefail

cd "$1"

# The script picks the device itself: JAX_PLATFORMS=cuda for the main process,
# cpu for spawned grain data-loader workers. Unset any inherited value so a
# stray JAX_PLATFORMS=cpu from the submit environment can't force CPU-only
# training.
unset JAX_PLATFORMS
exec uv run python examples/mnist_healpix_ablation.py \
    --pos-embed "$2" --shift-strategy "$3"
```

- [ ] **Step 2: Make the wrapper executable**

Run: `chmod +x examples/sub/run_mnist_healpix_ablation.sh`
Expected: no output; `ls -l examples/sub/run_mnist_healpix_ablation.sh` shows the `x` bit.

- [ ] **Step 3: Create the submit file**

Create `examples/sub/mnist_healpix_ablation_a100.sub` with exactly this content:

```
# HTCondor submit file: HealSwin shift x pos-embed ablation on spherical MNIST.
#
#     condor_submit examples/sub/mnist_healpix_ablation_a100.sub
#
# Queues the 6 deduped (pos_embed, shift_strategy) configs as separate jobs.
# `initialdir` makes the executable and condor_logs/ paths below resolve
# relative to this file's own directory, so it can be submitted from anywhere.
# Edit `repo_root` if your checkout lives elsewhere.
repo_root = /lustre/ific.uv.es/ml/ific088/github/HEAL-SWIN-nnx

universe   = vanilla
initialdir = $(repo_root)/examples/sub
executable = run_mnist_healpix_ablation.sh
arguments  = "$(repo_root) $(pos_embed) $(shift_strategy)"
getenv     = True

request_cpus   = 8
request_memory = 32 GB
request_gpus   = 1

+UseNvidiaA100 = True

log    = condor_logs/ablation_$(pos_embed)_$(shift_strategy).log
output = condor_logs/ablation_$(pos_embed)_$(shift_strategy).out
error  = condor_logs/ablation_$(pos_embed)_$(shift_strategy).err

queue pos_embed, shift_strategy from (
    rope_mixed, nest_roll
    rope_mixed, nest_grid_shift
    rope_mixed, nest_grid_shift_exact
    rope_mixed, ring_shift
    rel_bias, nest_grid_shift_exact
    rope_axial, nest_grid_shift_exact
)
```

- [ ] **Step 4: Dry-run the submit description on the submit host**

Run: `condor_submit -dry-run /dev/stdout examples/sub/mnist_healpix_ablation_a100.sub`
Expected: prints 6 job ClassAds (one per config) and exits 0 without queuing. The `Args`/`arguments` of each ClassAd should show the correct `<repo_root> <pos_embed> <shift_strategy>` triple. If `condor_submit` is unavailable on this host, skip and note that the user submits from the schedd.

- [ ] **Step 5: Commit**

```bash
git add examples/sub/run_mnist_healpix_ablation.sh examples/sub/mnist_healpix_ablation_a100.sub
git commit -m "feat(examples): HTCondor A100 submit queuing the 6 ablation configs"
```

---

## Self-Review

**Spec coverage:**
- Parameterized script + argparse + `make_params` forwarding → Task 1, Step 1. ✓
- CSV output with `#` config header + `epoch,train_loss,test_acc,time_per_epoch_s` → Task 1, Step 1 (`main`). ✓
- `SMOKE=1` retained, requires args → Task 1, Steps 1 & 3. ✓
- 6 deduped configs, A100, one `queue ... from` → Task 2, Step 3. ✓
- Per-job logs → Task 2, Step 3. ✓
- Run wrapper mirroring existing pattern → Task 2, Step 1. ✓
- Base `mnist_healpix_classify.py` untouched → no task modifies it. ✓
- Local verification is SMOKE-only on CPU; no training run locally → Task 1 Step 3, Task 2 Step 4. ✓

**Placeholder scan:** No TBD/TODO; all file contents given in full. ✓

**Type/name consistency:** `parse_args()` returns an object with `.pos_embed`/`.shift_strategy` (argparse dest of `--pos-embed`/`--shift-strategy`), consumed by `main(args)` and the SMOKE block; `make_params(pos_embed, shift_strategy)` signature matches both call sites. CSV filename `<pos_embed>__<shift_strategy>.csv` and log names `ablation_$(pos_embed)_$(shift_strategy)` are consistent within their tasks. ✓

**Note on `parse_known_args`:** argparse uses `parse_known_args` so it tolerates any absl/grain flags sharing argv; the absl parse above uses `known_only=True` for the reverse. Only `--pos-embed`/`--shift-strategy` are consumed by argparse.
