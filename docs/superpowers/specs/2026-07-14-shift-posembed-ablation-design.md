# Shift × positional-embedding ablation for HealSwin MNIST

**Date:** 2026-07-14
**Status:** Approved (design)

## Goal

Benchmark the HealSwin spherical-MNIST classifier across the library's shift
strategies and positional-embedding strategies, so the effect of each can be
compared on a single downstream task. Produce per-run CSVs that are trivial to
parse and plot at the end of training.

## Ablation matrix

Two sweeps, sharing the classifier config of `examples/mnist_healpix_classify.py`
(nside 64, embed_dim 32, depths (2,2,6,2), num_heads (4,8,16,16), window_size 16,
50 epochs, seed 0):

- **Shift sweep** — `pos_embed` fixed at `rope_mixed` (library default), varying
  `shift_strategy` over all four: `nest_roll`, `nest_grid_shift`,
  `nest_grid_shift_exact`, `ring_shift`.
- **Embedding sweep** — `shift_strategy` fixed at `nest_grid_shift_exact`,
  varying `pos_embed` over `rel_bias`, `rope_axial`, `rope_mixed`.

The `nest_grid_shift_exact` + `rope_mixed` cell is shared by both sweeps and is
**deduped**, giving **6 unique configs**:

| # | shift_strategy         | pos_embed  |
|---|------------------------|------------|
| 1 | nest_roll              | rope_mixed |
| 2 | nest_grid_shift        | rope_mixed |
| 3 | nest_grid_shift_exact  | rope_mixed |
| 4 | ring_shift             | rope_mixed |
| 5 | nest_grid_shift_exact  | rel_bias   |
| 6 | nest_grid_shift_exact  | rope_axial |

All six validate against `HealSwinParams` at this config: `nest_grid_shift`'s
bottleneck is 64 pix/face ≥ window_size 16; RoPE head-dims are 8,8,8,16, all
divisible by 4; `rel_bias`'s window (16) is a perfect square.

## Components

### 1. Parameterized training script — `examples/mnist_healpix_ablation.py`

A near-copy of `examples/mnist_healpix_classify.py`. Only functional changes:

- `argparse` with two **required** arguments:
  - `--pos-embed` ∈ {`rel_bias`, `rope_axial`, `rope_mixed`}
  - `--shift-strategy` ∈ {`nest_roll`, `nest_grid_shift`, `nest_grid_shift_exact`, `ring_shift`}
  (choices restricted to the enums used in this ablation; the library also
  accepts `pos_embed="none"` but it is out of scope here.)
- `make_params()` forwards both into `HealSwinParams(...)`.
- Output goes to `examples/ablation_results/<pos_embed>__<shift_strategy>.csv`
  (directory created if absent).
- `SMOKE=1` forward-shape check retained; it uses the parsed config (or a default
  when args are absent) so it can construct/run each config on CPU with no data.

Everything else — dataset, optimizer, schedule, train/eval loops, hyperparameters
— is identical, so runs are directly comparable. `examples/mnist_healpix_classify.py`
is left untouched.

### 2. Output format (CSV)

Per-run file `examples/ablation_results/<pos_embed>__<shift_strategy>.csv`:

```
# pos_embed=rope_mixed shift_strategy=nest_roll nside=64 embed_dim=32 depths=(2,2,6,2) window_size=16 batch=128 epochs=50 seed=0
epoch,train_loss,test_acc,time_per_epoch_s
0,1.8423,0.5210,42.7
1,...
```

- First line: a `#`-prefixed comment carrying the full run config (config is also
  encoded in the filename).
- Second line: CSV header `epoch,train_loss,test_acc,time_per_epoch_s`.
- One row per epoch, written with the stdlib `csv` module and flushed each epoch
  so a killed job leaves valid partial results.

Config is **not** duplicated into data columns — this is a small benchmark and
each file is parsed individually for plotting.

### 3. Run wrapper — `examples/sub/run_mnist_healpix_ablation.sh`

Mirrors the existing `run_mnist_healpix_classify.sh`. Args: `$1=repo_root`,
`$2=pos_embed`, `$3=shift_strategy`. Does `cd "$1"`, `unset JAX_PLATFORMS`
(so the script's own device selection wins), then:

```
exec uv run python examples/mnist_healpix_ablation.py \
     --pos-embed "$2" --shift-strategy "$3"
```

### 4. HTCondor submit — `examples/sub/mnist_healpix_ablation_a100.sub`

Mirrors `mnist_healpix_classify_a100.sub`: `universe=vanilla`,
`initialdir=$(repo_root)/examples/sub`, `getenv=True`, `request_cpus=8`,
`request_memory=32 GB`, `request_gpus=1`, `+UseNvidiaA100 = True`.

- `arguments = "$(repo_root) $(pos_embed) $(shift_strategy)"`
- Per-job logs: `condor_logs/ablation_$(pos_embed)_$(shift_strategy).{log,out,err}`
- A single `queue pos_embed, shift_strategy from ( ... )` block listing the 6
  deduped rows, so one `condor_submit` launches all six jobs.

## Verification

Before submission, run the `SMOKE=1` forward-shape check on CPU for all 6 configs
(fast, no data, no GPU) to confirm each constructs and completes a forward pass.
Training itself is **not** run locally — it is submitted to GPU nodes via the
`.sub` file (the established GPU-job workflow).

## Out of scope

- `pos_embed="none"` and any shift/embed combinations beyond the 6 above.
- Aggregation/plotting scripts — the CSVs are parsed ad hoc after training.
- Changes to the base `mnist_healpix_classify.py` or the model/library code.
