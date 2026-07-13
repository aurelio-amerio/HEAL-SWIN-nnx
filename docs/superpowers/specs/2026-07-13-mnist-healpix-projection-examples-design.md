# MNIST → HEALPix projection examples — design

**Date:** 2026-07-13
**Status:** approved

## Goal

Provide a clean, dependency-light `examples/` module that reimplements the two
image→HEALPix projection functions from the upstream NNhealpix reference
(`references/NNhealpix/nnhealpix/projections/__init__.py`), plus a notebook that
mirrors the upstream `MNIST2healpix.ipynb` demo but loads data through the
HuggingFace `datasets` library instead of `tensorflow.keras`.

The functions are data-preparation helpers (run once, off the model's hot path),
so construction-time numpy is fine — mirroring the `hp/topology.py` philosophy.

## Deliverables

### 1. `examples/projections.py`

A self-contained module — **no PyTorch, no numba** — defining:

- `img2healpix(img, nside, delta_theta, delta_phi, rot=np.eye(3), nest=True) -> (hp_map, hits)`
  Ray-tracing binner. Fires a dense grid of rays around the equator
  (θ=π/2, φ=0), rotates them by `rot` (a 3×3 matrix or `hp.rotator.Rotator`),
  bins the interpolated image values into HEALPix pixels, and returns the
  averaged map **and** the per-pixel hit count. Input is a single 2D image
  `(H, W)`.
- `img2healpix_planar(img, nside, thetac, phic, delta_theta, delta_phi, nest=True) -> hp_map`
  Planar projection centered at `(thetac, phic)` (HEALPix angle convention),
  using `scipy.interpolate.griddata(method="nearest")`. Accepts a single 2D
  image `(H, W)` or a batch `(N, H, W)`; returns a map of matching leading
  shape (`(npix,)` or `(N, npix)`).
- Private helpers `_img2map`, `_binned_map`.

Both public functions default to **NEST ordering** (`nest=True`) to match the
HealSwin model convention. Pass `nest=False` for healpy-default RING ordering
(e.g. to drop straight into `hp.mollview` without `nest=True`).

### 2. `examples/mnist_to_healpix.ipynb`

Mirrors the upstream notebook's project-and-visualize flow:

1. Load MNIST via `datasets.load_dataset("mnist")`; pull one digit as a numpy
   array (HF returns PIL images → `np.array(...)`).
2. Planar projection of the digit → `hp.mollview(..., nest=True)` + `plt.imshow`
   of the source image.
3. Ray-traced projection → `hp.mollview` of the map **and** the hit map (to show
   the projection leaves no gaps within the frame).
4. Rotated example (`hp.rotator.Rotator(rot=(120, 45, 30))`) showing the image
   wrapping around the pole.

Visualization uses `nest=True` throughout since the functions now emit NEST.

### 3. `pyproject.toml`

Add `matplotlib` and `scipy` to `[project.optional-dependencies].examples`
(currently `notebook`, `datasets`). `matplotlib` is needed by `hp.mollview` /
`plt.imshow`; `scipy` by the planar `griddata` path. Both happen to be present
transitively today, but the extra should declare them explicitly.

## Key implementation notes

- **NEST-native construction:** thread `nest=nest` through `hp.ang2pix` /
  `hp.pix2ang` so maps are built directly in the requested ordering — no
  post-hoc `ring2nest` reindexing.
- **De-numba:** replace the numba `binned_map` loop with vectorized
  `np.add.at(mappixels, pixidx, signal)` and `np.add.at(hits, pixidx, 1)`, then
  average where `hits > 0`. Same numerical result, one fewer dependency.
- **Modernize scipy:** the deprecated `scipy.ndimage.interpolation.zoom` →
  `scipy.ndimage.zoom`. Keep `scipy.interpolate.griddata` for the planar path.
- **YAGNI:** drop the unused `projectimages` iterator class from the port; it is
  not needed by the notebook and was not requested.
- Type hints, cleaned docstrings, and an explicit note on the RING/NEST
  convention on both public functions.

## Alternatives rejected

- **Keep numba** for the binner loop — numba is not installed, and these are
  one-off data-prep helpers where vectorized numpy is plenty fast.
- **Hand-roll the planar interpolation** to drop scipy — scipy is already
  available; `griddata(method="nearest")` is clearer than a bespoke KD-tree.
- **RING-default / NEST-default-with-no-flag** — settled during brainstorming:
  NEST by default (model-ready) with a `nest` flag retained for RING.

## Verification

Examples are not part of the pytest suite. Before completion, execute the module
and notebook end-to-end:

- Project a digit at a known center; assert the hit region is non-empty and
  spatially confined (a manual invariant check, not a committed pytest).
- Run the notebook top-to-bottom and confirm it produces figures without error.

Report the observed result, not just that the code imports.
