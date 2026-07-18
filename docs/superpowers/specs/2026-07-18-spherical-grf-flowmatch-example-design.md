# Spherical GRF flow-matching example — design

Date: 2026-07-18
Status: approved (brainstormed with user)

## Context

Showcase `HealSwinEncoder` as a spherical-image encoder for simulation-based
inference: reconstruct the (unimodal, 3-dim) posterior of the `spherical_grf`
task from sbibm-jax at nside 64 with flow matching, using gensbi's `Flux1`
as the posterior model and the encoder's nside-2 bottleneck tokens as the
conditioning stream. First iteration trains on the task's **online** sampler;
once the offline HF dataset is published the loader swap is constructor-only.
If the example produces good posteriors it will migrate to gensbi-examples
(and HealSwin, in some capacity, into gensbi), so it deliberately mirrors
`GenSBI-examples/examples/sbi-benchmarks/lensing/train-lensing.py` in
pipeline usage while keeping the single-file constants-block style of
`examples/mnist_healpix_classify.py`.

## Deliverables

- `examples/spherical_grf_flowmatch.py` — self-contained training +
  evaluation script (constants block at top, no YAML).
- `examples/sub/spherical_grf_flowmatch.sub` + `examples/sub/run_spherical_grf_flowmatch.sh`
  — HTCondor A100 submit files following the existing pattern, including
  `requirements = (TARGET.Gpus_Capability >= 8.0)`.

No changes to `src/heal_swin_nnx/` and no new dependencies: gensbi 0.4.0 and
sbibm-jax are already provided by the `gensbi` extra
(`gensbi = ["gensbi[examples]"]` in `pyproject.toml`).

## Task facts (verified against sbibm-jax source)

- `SphericalGRF`: theta = `(logA, n, alpha)`, `dim_theta = 3`, independent
  uniform prior with `low = (-2, -3, -0.5)`, `high = (2, 0, 0.5)`.
- Observations: flat HEALPix maps, **RING**-ordered, `npix = 49152`
  (nside 64), single channel, float32.
- Reference: 10 canonical observations with `get_observation(i)`,
  `get_true_parameters(i)`, and precomputed
  `get_reference_posterior_samples(i)` → `(10000, 3)`.
- `jax_healpy` is **not** installed → online simulation uses the healpy/NumPy
  backend on CPU.

## Data pipeline

`OnlineTaskDataset` from `sbibm_jax.data` (verified in
`src/sbibm_jax/data/dataset.py`):

```python
ds = OnlineTaskDataset(
    "spherical_grf",
    task_kwargs={},          # local mode: shapes from the task, no Hub metadata
    ordering="nest",         # ring→nest permutation applied in collate
    normalize=True,
    stats={"theta_mean": ..., "theta_std": ..., "x_mean": ..., "x_std": ...},
)
train_loader = ds.get_online_train_loader(BATCH_SIZE, seed=SEED, num_workers=N)
val_loader   = ds.get_online_train_loader(VAL_BATCH,  seed=SEED + 1, num_workers=...)
```

- Batches arrive as `(theta (B, 3, 1), x (B, 49152, 1))` — tokenized,
  NEST-ordered, normalized; identical to what the future offline
  `TaskDataset(..., ordering="nest", normalize=True)` yields. The later swap
  replaces the constructor + loader calls only.
- Simulation runs in grain CPU spawn workers (`_worker_init` pins workers to
  CPU inside sbibm-jax); the GPU stays free for the training step.
- **Stats**: theta mean/std analytic from the uniform prior box
  (`mean = (low+high)/2`, `std = (high-low)/sqrt(12)`); x mean/std are global
  scalars (isotropic field) computed once at startup from a warmup batch of
  ~512 sims with a fixed dedicated seed, printed to the results file so they
  can be hardcoded later. Stats dict keys as above (verified
  `dataset.py:86-92`).

## Model

**Encoder** — `HealSwinEncoder(HealSwinParams(...))`:

| knob | value |
|---|---|
| nside / in_channels | 64 / 1 |
| embed_dim | 32 |
| depths | (2, 2, 6, 2, 2) |
| num_heads | (4, 8, 16, 16, 16) |
| window_size | 16 |
| pos_embed / shift_strategy | defaults (`rope_mixed`, `nest_grid_shift_exact`) |
| out_channels | 1 (required by the dataclass, unused by the encoder) |

Resolution walk: patch embed (patch_size 4) nside 64→32, then 4 patch
mergings 32→16→8→4→2; the fifth stage runs at nside 2. Bottleneck output:
**48 tokens × 512 features** (`num_features = 32·2⁴`). All
`HealSwinParams.__post_init__` constraints verified (head dims 8, 8, 8, 16,
32 — all divisible by 4 as RoPE requires; `nest_grid_shift_exact` supports
the small bottleneck).

**Posterior model** — `Flux1(Flux1Params(...))`:

| knob | value |
|---|---|
| in_channels | 1 (θ tokens) |
| context_in_dim | 512 (= encoder `num_features`) |
| dim_obs / dim_cond | 3 / 48 |
| depth / depth_single_blocks | 4 / 4 |
| num_heads | 6 |
| axes_dim | [64] → hidden_size 384 |
| mlp_ratio / qkv_bias | 4 / True |
| vec_in_dim / guidance_embed | None / False |
| id_embedding_strategy | ("absolute", "pos1d") |
| param_dtype | float32 |

`("absolute", "pos1d")`: learned id table for the 3 θ tokens, **sinusoidal**
(pos1d) ids for the 48 spherical tokens. Everything float32 in this first
iteration.

**Wrapper** — small `nnx.Module` in the example (analogous to
`LensingModel`):

```python
class SphericalGRFModel(nnx.Module):
    def __init__(self, encoder, flux): ...
    def __call__(self, t, obs, obs_ids, cond, cond_ids,
                 conditioned=True, guidance=None):
        tokens, _skips = self.encoder(cond)   # (B, 49152, 1) -> (B, 48, 512)
        return self.flux(t=t, obs=obs, obs_ids=obs_ids, cond=tokens,
                         cond_ids=cond_ids, conditioned=conditioned,
                         guidance=guidance)
```

The encoder is deterministic at eval time, so the `encoder_key` /
`sample_batched` model-extras gotcha in gensbi does not apply here.

## Training

`ConditionalPipeline(model, train_loader, val_loader, dim_obs=3, dim_cond=48,
method=FlowMatchingMethod(), ch_obs=1, ch_cond=512,
id_embedding_strategy=("absolute", "pos1d"), training_config=...)`.

- Constants block feeds `training_config` on top of gensbi defaults:
  `nsteps ≈ 20_000`, batch 128, EMA + early stopping as per defaults,
  `checkpoint_dir = examples/checkpoints/spherical_grf/`.
- `pipeline.train(nnx.Rngs(SEED), save_model=True)`; a `RESTORE` flag in the
  constants block switches to `pipeline.restore_model()` for eval-only runs.

## Evaluation

For 2–3 canonical observations (`i = 1, 2, 3`):

1. `task.get_observation(i)` → ring→nest permute + normalize + tokenize with
   the *same* transforms as training data (reuse the dataset's `_x_perm` /
   `normalize_x`).
2. `pipeline.sample_batched(...)` → unnormalize theta.
3. Plots to `examples/imgs/`: (a) overlay corner of flow posterior vs
   `get_reference_posterior_samples(i)`, (b) **separate** corner plots of
   each (in case the overlay hides one under the other), (c) true θ marked
   in all plots.

Plus TARP coverage (`gensbi.diagnostics.run_tarp` / `plot_tarp`) on ~200
freshly simulated (θ, x) pairs from a held-out seed. Metrics and progress
lines go to `examples/spherical_grf_flowmatch_results.txt`.

## Runtime plumbing

Same guards as `mnist_healpix_classify.py`:

- Spawned workers get `JAX_PLATFORMS=cpu`; main process defaults to `cuda`
  (`os.environ.setdefault`), overridable by the caller.
- absl flags parsed once at import so grain's `mp_prefetch` doesn't raise
  `UnparsedFlagAccessError`.
- `SMOKE=1` mode: CPU forward-shape check (encoder + Flux1 on zeros, no
  data, no training) so the wiring can be tested on the login node.
- Training itself is submitted by the user via HTCondor; nothing in the
  script assumes an interactive GPU.

## Verification

1. `SMOKE=1 JAX_PLATFORMS=cpu uv run python examples/spherical_grf_flowmatch.py`
   — shape check passes on the login node.
2. Short-run sanity: a few training steps with tiny `nsteps` on CPU to
   confirm the pipeline loop, checkpointing, and eval path execute.
3. Full training on the A100 via the submit file (user-submitted); success =
   flow posterior visually matching the reference posterior corners and
   near-diagonal TARP.

## Future migration notes

- Offline swap: replace `OnlineTaskDataset(...)`/`get_online_train_loader`
  with `TaskDataset("spherical_grf", ordering="nest", normalize=True)` and
  its `get_train_loader`/`get_val_loader`; Hub metadata then supplies stats.
- gensbi migration: the wrapper + pipeline usage already matches
  gensbi-examples conventions; the constants block would become a YAML
  config there.
