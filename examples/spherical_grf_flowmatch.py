# -*- coding: utf-8 -*-
"""Flow-matching NPE on the spherical GRF task with a HealSwin encoder.

A HEALPix-native Swin encoder compresses each nside-64 spherical map to 48
bottleneck tokens (nside 2, 512 features), which condition a gensbi Flux1
flow-matching model over the 3-dim posterior (logA, n, alpha) of the
sbibm-jax `spherical_grf` task. Training data is simulated online (healpy
backend, CPU spawn workers); the loader interface matches the future offline
TaskDataset so the swap is constructor-only. Design doc:
docs/superpowers/specs/2026-07-18-spherical-grf-flowmatch-example-design.md

Run headless. The script defaults to the GPU (``JAX_PLATFORMS=cuda``) and
will fail fast on a machine with no CUDA device.

    uv run python examples/spherical_grf_flowmatch.py

Or submit to a GPU node: ``condor_submit examples/sub/spherical_grf_flowmatch.sub``.

Debug modes (both CPU-safe):

    SMOKE=1 JAX_PLATFORMS=cpu uv run python examples/spherical_grf_flowmatch.py
        forward-shape check, no data, no training
    QUICK=1 JAX_PLATFORMS=cpu uv run python examples/spherical_grf_flowmatch.py
        tiny end-to-end run (few sims, few steps, few samples)
"""

from __future__ import annotations

import os
import sys

# Any spawned worker re-importing this module must never grab the GPU; the
# main process defaults to CUDA (an explicit JAX_PLATFORMS from the caller
# still wins via setdefault). Same pattern as mnist_healpix_classify.py.
if __name__ != "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ.setdefault("JAX_PLATFORMS", "cuda")

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from absl import flags

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from corner import corner

from heal_swin_nnx import HealSwinEncoder, HealSwinParams

from gensbi.core import FlowMatchingMethod
from gensbi.models import Flux1, Flux1Params
from gensbi.recipes import ConditionalPipeline
from gensbi.recipes.utils import init_ids_1d
from gensbi.utils.plotting import plot_marginals
from gensbi.diagnostics import run_tarp, plot_tarp

from sbibm_jax.tasks import get_task
from sbibm_jax.tasks.spherical_grf.task import PRIOR_LOW, PRIOR_HIGH
from sbibm_jax.data import OnlineTaskDataset

# grain's mp_prefetch reads absl flags; parse argv once so a plain
# `python ...` run doesn't hit UnparsedFlagAccessError on first prefetch.
if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv, known_only=True)

QUICK = os.environ.get("QUICK") == "1"

# --- config (tune here) --------------------------------------------------
NSIDE = 64
NPIX = 12 * NSIDE ** 2
DIM_THETA = 3
THETA_LABELS = (r"$\log A$", r"$n$", r"$\alpha$")

# HealSwin encoder: patch embed nside 64->32, then 4 mergings 32->16->8->4->2.
EMBED_DIM = 32
DEPTHS = (2, 2, 6, 2, 2)
ENC_NUM_HEADS = (4, 8, 16, 16, 16)
WINDOW_SIZE = 16
COND_TOKENS = 48                                    # 12 faces * bottleneck nside(=2)^2
COND_FEATURES = EMBED_DIM * 2 ** (len(DEPTHS) - 1)  # 512

# Flux1 posterior model
FLUX_DEPTH = 4                       # double-stream blocks
FLUX_DEPTH_SINGLE = 4                # single-stream blocks
FLUX_NUM_HEADS = 6
FLUX_AXES_DIM = (64,)                # hidden_size = sum(axes_dim) * heads = 384
ID_EMBEDDING = ("absolute", "pos1d") # learned theta-token ids, sinusoidal cond ids

# training / data
SEED = 0
BATCH_SIZE = 8 if QUICK else 128
VAL_BATCH_SIZE = 8 if QUICK else 256
NSTEPS = 5 if QUICK else 20_000
WARMUP_SIMS = 32 if QUICK else 512
NUM_WORKERS = 0 if QUICK else min(8, max(1, (os.cpu_count() or 2) - 2))
TRAIN_MODEL = True
RESTORE_MODEL = False

# evaluation
EVAL_OBSERVATIONS = (1,) if QUICK else (1, 2, 3)
NUM_POSTERIOR_SAMPLES = 64 if QUICK else 10_000
SAMPLE_CHUNK = 64 if QUICK else 500       # encoder reruns per ODE step: keep chunks GPU-sized
SAMPLE_STEP_SIZE = 0.25 if QUICK else 0.01
TARP_PAIRS = 2 if QUICK else 200
TARP_POSTERIOR_SAMPLES = 8 if QUICK else 1_000
TARP_CHUNK = 2 if QUICK else 100

EXPERIMENT_ID = "spherical_grf_fm_quick" if QUICK else "spherical_grf_fm"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", EXPERIMENT_ID)
IMGS_DIR = os.path.join(BASE_DIR, "imgs")
RESULTS_FILE = os.path.join(BASE_DIR, f"{EXPERIMENT_ID}_results.txt")
# ------------------------------------------------------------------------

# Uniform prior box -> analytic normalization stats.
THETA_MEAN = tuple((lo + hi) / 2.0 for lo, hi in zip(PRIOR_LOW, PRIOR_HIGH))
THETA_STD = tuple((hi - lo) / math.sqrt(12.0) for lo, hi in zip(PRIOR_LOW, PRIOR_HIGH))


def make_encoder_params() -> HealSwinParams:
    return HealSwinParams(
        nside=NSIDE,
        in_channels=1,
        out_channels=1,  # required by the dataclass; unused by the encoder
        embed_dim=EMBED_DIM,
        depths=DEPTHS,
        num_heads=ENC_NUM_HEADS,
        window_size=WINDOW_SIZE,
    )


def make_flux_params(rngs: nnx.Rngs) -> Flux1Params:
    return Flux1Params(
        rngs=rngs,
        in_channels=1,
        vec_in_dim=None,
        context_in_dim=COND_FEATURES,
        mlp_ratio=4.0,
        num_heads=FLUX_NUM_HEADS,
        depth=FLUX_DEPTH,
        depth_single_blocks=FLUX_DEPTH_SINGLE,
        qkv_bias=True,
        dim_obs=DIM_THETA,
        dim_cond=COND_TOKENS,
        axes_dim=list(FLUX_AXES_DIM),
        id_embedding_strategy=ID_EMBEDDING,
        param_dtype=jnp.float32,
    )


class SphericalGRFModel(nnx.Module):
    """HealSwin spherical encoder feeding Flux1's conditioning stream."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.encoder = HealSwinEncoder(make_encoder_params(), rngs=rngs)
        assert self.encoder.num_features == COND_FEATURES
        self.flux = Flux1(make_flux_params(rngs))

    def __call__(self, t, obs, obs_ids, cond, cond_ids,
                 conditioned=True, guidance=None, **kwargs):
        tokens, _skips = self.encoder(cond)  # (B, NPIX, 1) -> (B, 48, 512)
        return self.flux(t=t, obs=obs, obs_ids=obs_ids, cond=tokens,
                         cond_ids=cond_ids, conditioned=conditioned,
                         guidance=guidance)


def normalize_theta(theta):
    return (np.asarray(theta) - np.asarray(THETA_MEAN)) / np.asarray(THETA_STD)


def unnormalize_theta(theta):
    return np.asarray(theta) * np.asarray(THETA_STD) + np.asarray(THETA_MEAN)


def compute_x_stats(task, num_sims, seed):
    """Global scalar x mean/std from a warmup batch of prior simulations.

    The field is isotropic, so a single scalar pair suffices (matches the
    published metadata's x stats axes (0, 1)). Ordering-independent, so the
    RING simulator output can be used directly.
    """
    sim = task.get_simulator(jax.random.PRNGKey(seed))
    kt, ks = jax.random.split(jax.random.PRNGKey(seed + 1))
    theta = task.get_prior(kt, num_sims)
    x = np.asarray(sim(ks, theta))
    return float(x.mean()), float(x.std())


def make_datasets():
    """OnlineTaskDataset + train/val loaders of normalized NEST (theta, x) batches.

    Offline swap (once the HF dataset is published): replace this body with
    TaskDataset("spherical_grf", ordering="nest", normalize=True) and its
    get_train_loader/get_val_loader — stats then come from Hub metadata.
    """
    task = get_task("spherical_grf")
    x_mean, x_std = compute_x_stats(task, WARMUP_SIMS, SEED + 100)
    stats = {
        "theta_mean": list(THETA_MEAN), "theta_std": list(THETA_STD),
        "x_mean": x_mean, "x_std": x_std,
    }
    ds = OnlineTaskDataset(
        "spherical_grf", task_kwargs={}, ordering="nest",
        normalize=True, stats=stats, seed=SEED,
    )
    train_loader = ds.get_online_train_loader(
        BATCH_SIZE, seed=SEED, num_workers=NUM_WORKERS)
    # The pipeline draws one fixed val batch; simulate it in-process.
    val_loader = ds.get_online_train_loader(
        VAL_BATCH_SIZE, seed=SEED + 1, num_workers=0)
    return ds, stats, train_loader, val_loader


def prep_x(x_ring, ds):
    """Raw RING maps (B, NPIX) -> normalized NEST tokens (B, NPIX, 1).

    Mirrors the training collate exactly: permute, tokenize, normalize.
    """
    x = np.asarray(x_ring)[:, ds._x_perm][..., None]
    x = (x - ds.x_mean) / ds.x_std
    return jnp.asarray(x, dtype=jnp.float32)


# gensbi 0.4.0 passes the experiment_id *string* as the orbax checkpoint
# step, which orbax-checkpoint 0.12.1 rejects (TypeError during step
# directory formatting). Checkpoints are already namespaced by
# CHECKPOINT_DIR, so pin the step to 0 for both save and restore until
# gensbi fixes this upstream.
_orig_save_model = ConditionalPipeline.save_model
_orig_restore_model = ConditionalPipeline.restore_model


def _save_model_step0(self, experiment_id=None):
    return _orig_save_model(self, 0)


def _restore_model_step0(self, experiment_id=None):
    return _orig_restore_model(self, 0)


ConditionalPipeline.save_model = _save_model_step0
ConditionalPipeline.restore_model = _restore_model_step0


def make_training_config():
    cfg = ConditionalPipeline.get_default_training_config()
    cfg["nsteps"] = NSTEPS
    cfg["experiment_id"] = EXPERIMENT_ID
    cfg["checkpoint_dir"] = CHECKPOINT_DIR
    if QUICK:
        cfg["warmup_steps"] = 2
        cfg["val_every"] = 2
        cfg["decay_transition"] = 0
    return cfg


def make_pipeline(model, train_loader, val_loader):
    return ConditionalPipeline(
        model, train_loader, val_loader,
        dim_obs=DIM_THETA, dim_cond=COND_TOKENS,
        method=FlowMatchingMethod(),
        ch_obs=1, ch_cond=COND_FEATURES,
        id_embedding_strategy=ID_EMBEDDING,
        training_config=make_training_config(),
    )


def evaluate(pipeline, ds, log):
    """Posterior vs reference for canonical observations, then TARP."""
    task = ds.task
    key = jax.random.PRNGKey(SEED + 7)
    labels = list(THETA_LABELS)

    for i in EVAL_OBSERVATIONS:
        x_o = prep_x(np.asarray(task.get_observation(i)), ds)  # (1, NPIX, 1)
        theta_true = np.asarray(task.get_true_parameters(i))[0]
        ref = np.asarray(task.get_reference_posterior_samples(i))
        key, sk = jax.random.split(key)
        t0 = time.time()
        samples = pipeline.sample_batched(
            sk, x_o, NUM_POSTERIOR_SAMPLES,
            chunk_size=SAMPLE_CHUNK, step_size=SAMPLE_STEP_SIZE,
        )
        flow = unnormalize_theta(np.asarray(samples)[:, 0, :, 0])  # (S, 3)
        log(f"obs {i}: {flow.shape[0]} samples in {time.time() - t0:.0f}s | "
            f"true {np.array2string(theta_true, precision=3)} | "
            f"flow mean {np.array2string(flow.mean(0), precision=3)} "
            f"std {np.array2string(flow.std(0), precision=3)} | "
            f"ref mean {np.array2string(ref.mean(0), precision=3)} "
            f"std {np.array2string(ref.std(0), precision=3)}")

        # Overlay: reference (blue) under flow posterior (orange).
        fig = corner(ref, labels=labels, truths=list(theta_true), color="C0",
                     hist_kwargs={"density": True},
                     plot_contours=not QUICK, plot_density=not QUICK)
        corner(flow, fig=fig, color="C1", hist_kwargs={"density": True},
               plot_contours=not QUICK, plot_density=not QUICK)
        fig.suptitle(f"obs {i}: reference (blue) vs flow (orange)")
        fig.savefig(os.path.join(IMGS_DIR, f"{EXPERIMENT_ID}_overlay_obs{i}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Separate corners, in case the overlay hides one under the other.
        plot_marginals(ref, true_param=theta_true, labels=labels, gridsize=30)
        plt.savefig(os.path.join(IMGS_DIR, f"{EXPERIMENT_ID}_reference_obs{i}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close("all")
        plot_marginals(flow, true_param=theta_true, labels=labels, gridsize=30)
        plt.savefig(os.path.join(IMGS_DIR, f"{EXPERIMENT_ID}_flow_obs{i}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close("all")

    tarp_diagnostic(pipeline, ds, log, key)


def tarp_diagnostic(pipeline, ds, log, key):
    """TARP coverage on freshly simulated pairs (normalized theta space)."""
    task = ds.task
    kt, ks, kp = jax.random.split(key, 3)
    sim = task.get_simulator(jax.random.PRNGKey(SEED + 300))
    theta = np.asarray(task.get_prior(kt, TARP_PAIRS))       # (P, 3)
    t0 = time.time()
    x = np.asarray(sim(ks, jnp.asarray(theta)))               # (P, NPIX) RING
    x_tok = prep_x(x, ds)
    post = pipeline.sample_batched(
        kp, x_tok, TARP_POSTERIOR_SAMPLES,
        chunk_size=TARP_CHUNK, step_size=SAMPLE_STEP_SIZE,
    )
    post = np.asarray(post)[:, :, :, 0]                       # (S, P, 3)
    res = run_tarp(jnp.asarray(normalize_theta(theta)), jnp.asarray(post),
                   bootstrap=False)
    plot_tarp(res, mode="both")
    plt.savefig(os.path.join(IMGS_DIR, f"{EXPERIMENT_ID}_tarp.png"),
                dpi=100, bbox_inches="tight")
    plt.close("all")
    log(f"TARP: {TARP_PAIRS} pairs x {TARP_POSTERIOR_SAMPLES} samples "
        f"in {time.time() - t0:.0f}s -> {EXPERIMENT_ID}_tarp.png")


def main():
    os.makedirs(IMGS_DIR, exist_ok=True)
    results_file = open(RESULTS_FILE, "w")

    def log(line):
        print(line, flush=True)
        results_file.write(line + "\n")
        results_file.flush()

    log(f"quick={QUICK} batch={BATCH_SIZE} nsteps={NSTEPS} workers={NUM_WORKERS} "
        f"nside={NSIDE} embed_dim={EMBED_DIM} depths={DEPTHS} window={WINDOW_SIZE} "
        f"cond={COND_TOKENS}x{COND_FEATURES} flux={FLUX_DEPTH}d+{FLUX_DEPTH_SINGLE}s "
        f"heads={FLUX_NUM_HEADS} ids={ID_EMBEDDING}")

    t0 = time.time()
    ds, stats, train_loader, val_loader = make_datasets()
    log(f"x stats from {WARMUP_SIMS} warmup sims: mean={stats['x_mean']:.6g} "
        f"std={stats['x_std']:.6g} ({time.time() - t0:.1f}s)")

    model = SphericalGRFModel(rngs=nnx.Rngs(SEED))
    pipeline = make_pipeline(model, train_loader, val_loader)

    if TRAIN_MODEL:
        t0 = time.time()
        losses, val_losses = pipeline.train(nnx.Rngs(SEED + 2), save_model=True)
        log(f"training: {len(losses)} steps in {time.time() - t0:.0f}s, "
            f"final train loss {float(losses[-1]):.4f}, "
            f"final val loss {float(val_losses[-1]):.4f}")
    if RESTORE_MODEL:
        pipeline.restore_model()

    evaluate(pipeline, ds, log)
    results_file.close()


if __name__ == "__main__" and os.environ.get("SMOKE") != "1":
    main()

if __name__ == "__main__" and os.environ.get("SMOKE") == "1":
    # Forward-shape smoke check: no data, no training; runs on CPU.
    model = SphericalGRFModel(rngs=nnx.Rngs(0))
    model.eval()
    B = 2
    obs_ids, _ = init_ids_1d(DIM_THETA, 0)    # (1, 3, 2) — broadcast over batch
    cond_ids, _ = init_ids_1d(COND_TOKENS, 1)  # (1, 48, 2)
    v = model(
        t=jnp.full((B,), 0.5),
        obs=jnp.zeros((B, DIM_THETA, 1)),
        obs_ids=obs_ids,
        cond=jnp.zeros((B, NPIX, 1)),
        cond_ids=cond_ids,
    )
    print("vector field shape:", v.shape)
    assert v.shape == (B, DIM_THETA, 1)
    print("forward smoke check OK")
