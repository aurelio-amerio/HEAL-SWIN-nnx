"""Benchmark HEALPix window-shift strategies: build cost + forward/backward time.

Two benchmarks, both forward+backward, on the full sphere (12 base pixels):

  1. Shift op in isolation — construct each shifter, then time its shift and shift_back
     through ``jax.value_and_grad`` (gathers forward, scatters in the backward).
  2. Full HealSwin model — build a model per strategy and time one forward+backward
     (loss-gradient) step.

The point it makes visible: at *runtime* ``nest_grid_shift``, ``nest_grid_shift_exact``
and ``ring_shift`` are the SAME op — a single ``jnp.take`` gather over a precomputed
index buffer — while ``nest_roll`` is a ``jnp.roll``. Where the strategies genuinely
diverge is *build* cost (index construction, healpy round-trips, seam geometry) and
geometric fidelity, not per-step compute. Device (CPU/GPU) is auto-detected.

Run:  uv run python scripts/bench_strategies.py
"""
import argparse
import statistics
import time

import jax
import jax.numpy as jnp
from flax import nnx

from heal_swin_nnx import HealSwin, HealSwinParams
from heal_swin_nnx.hp import shifting

STRATEGIES = ("nest_roll", "nest_grid_shift", "nest_grid_shift_exact", "ring_shift")


def build_shifter(strategy, nside, base_pixels, window_size):
    """Construct the standalone shift module for a strategy (the real code path)."""
    shift_size = window_size // 2
    npix = len(base_pixels) * nside ** 2
    if strategy == "nest_roll":
        return shifting.NestRollShift(shift_size=shift_size, input_resolution=npix,
                                      window_size=window_size)
    if strategy == "nest_grid_shift":
        return shifting.NestGridShift(nside=nside, base_pixels=base_pixels,
                                      window_size=window_size)
    if strategy == "nest_grid_shift_exact":
        return shifting.NestGridShiftExact(nside=nside, base_pixels=base_pixels,
                                           window_size=window_size)
    return shifting.RingShift(nside=nside, base_pixels=base_pixels,
                              window_size=window_size, shift_size=shift_size)


def timeit(fn, *args, warmup, repeats):
    """Median wall time of ``fn(*args)`` after warmup, with device sync each call."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def bench_shift_op(nside, base_pixels, window_size, channels, batch, warmup, repeats):
    """Per strategy: constructor wall time, and jitted forward+backward of the shift."""
    npix = len(base_pixels) * nside ** 2
    x = jax.random.normal(jax.random.PRNGKey(0), (batch, npix, channels))
    # Absorb the one-time cost of importing healpy and priming the Buffer/asarray
    # path so it doesn't land on whichever strategy is measured first.
    for warm in STRATEGIES:
        build_shifter(warm, 4, base_pixels, window_size)
    rows = []
    for strat in STRATEGIES:
        t0 = time.perf_counter()
        shifter = build_shifter(strat, nside, base_pixels, window_size)
        build_ms = (time.perf_counter() - t0) * 1e3

        def loss(x, shifter=shifter):
            # A block shifts on the way in and shifts back on the way out: two gathers
            # forward, two scatters in the backward. The 0.5*sum(.**2) makes the loss
            # non-linear in x, so the gradient depends on x and XLA cannot constant-fold
            # the backward pass away (a plain linear loss silently measures a no-op).
            return 0.5 * (jnp.sum(shifter.shift(x) ** 2)
                          + jnp.sum(shifter.shift_back(x) ** 2))

        step = jax.jit(jax.value_and_grad(loss))
        fb_us = timeit(step, x, warmup=warmup, repeats=repeats) * 1e6
        rows.append((strat, build_ms, fb_us))
    return npix, rows


def bench_model(nside, base_pixels, in_ch, out_ch, batch, warmup, repeats, arch):
    """Per strategy: model build time, and jitted forward+backward of one step."""
    npix = len(base_pixels) * nside ** 2
    x = jax.random.normal(jax.random.PRNGKey(2), (batch, npix, in_ch))
    y = jax.random.normal(jax.random.PRNGKey(3), (batch, npix, out_ch))

    @nnx.jit
    def step(model, x, y):
        def loss_fn(model):
            return jnp.mean((model(x) - y) ** 2)
        return nnx.grad(loss_fn)(model)

    # Warm up nnx/JAX module-construction machinery on a throwaway tiny model so its
    # one-time cost doesn't get charged to the first strategy's build time.
    warm = HealSwinParams(nside=16, in_channels=in_ch, out_channels=out_ch,
                          base_pixels=base_pixels, drop_path_rate=0.0, **arch)
    HealSwin(warm, rngs=nnx.Rngs(0))

    rows = []
    for strat in STRATEGIES:
        params = HealSwinParams(nside=nside, in_channels=in_ch, out_channels=out_ch,
                                base_pixels=tuple(base_pixels), shift_strategy=strat,
                                drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                                **arch)
        t0 = time.perf_counter()
        model = HealSwin(params, rngs=nnx.Rngs(0))
        build_ms = (time.perf_counter() - t0) * 1e3
        fb_ms = timeit(step, model, x, y, warmup=warmup, repeats=repeats) * 1e3
        rows.append((strat, build_ms, fb_ms))
    return npix, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nsides", type=int, nargs="+", default=[16, 64],
                    help="HEALPix resolutions to sweep (full sphere)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--window-size", type=int, default=4)
    ap.add_argument("--channels", type=int, default=96,
                    help="channel width for the shift-op micro-benchmark")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--shift-repeats", type=int, default=50)
    ap.add_argument("--model-repeats", type=int, default=10)
    args = ap.parse_args()

    base_pixels = tuple(range(12))  # full sphere
    # 3 stages: with patch_size=4 the per-stage nside is nside/2, /4, /8, so the
    # deepest stage keeps nside>=2 for both sweep points (16 -> 2, 64 -> 8). That
    # matters because nest_grid_shift's index math needs nside>=2 at every shifted
    # stage; a 4th stage would drive nside=16 down to 1 and divide by zero.
    arch = dict(embed_dim=48, depths=(2, 2, 2), num_heads=(2, 4, 8))

    dev = jax.devices()[0]
    print("=" * 74)
    print("HEALPix shift-strategy benchmark  (forward + backward)")
    print("backend=%s  device=%s  jax=%s" % (jax.default_backend(), dev, jax.__version__))
    print("full sphere (12 base pixels), window_size=%d, batch=%d" %
          (args.window_size, args.batch))
    print("=" * 74)

    for nside in args.nsides:
        _, rows = bench_shift_op(nside, base_pixels, args.window_size, args.channels,
                                 args.batch, args.warmup, args.shift_repeats)
        npix = len(base_pixels) * nside ** 2
        print("\n[1] shift op   nside=%d  npix=%d  channels=%d" %
              (nside, npix, args.channels))
        print("    %-24s %12s %14s" % ("strategy", "build (ms)", "fwd+bwd (us)"))
        for strat, build_ms, fb_us in rows:
            print("    %-24s %12.2f %14.1f" % (strat, build_ms, fb_us))

    for nside in args.nsides:
        _, rows = bench_model(nside, base_pixels, 3, 5, args.batch,
                              args.warmup, args.model_repeats, arch)
        npix = len(base_pixels) * nside ** 2
        print("\n[2] full model   nside=%d  npix=%d  %s" % (nside, npix, arch))
        print("    %-24s %12s %14s" % ("strategy", "build (ms)", "fwd+bwd (ms)"))
        for strat, build_ms, fb_ms in rows:
            print("    %-24s %12.2f %14.1f" % (strat, build_ms, fb_ms))
    print()


if __name__ == "__main__":
    main()
