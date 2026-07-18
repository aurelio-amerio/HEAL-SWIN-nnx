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
