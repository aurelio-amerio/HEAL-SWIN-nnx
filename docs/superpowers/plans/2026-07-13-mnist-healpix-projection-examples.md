# MNIST → HEALPix Projection Examples Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a dependency-light `examples/projections.py` reimplementing the two NNhealpix image→HEALPix projection functions (NEST-default, no numba), plus a notebook that mirrors the upstream MNIST demo using the HuggingFace `datasets` library.

**Architecture:** Pure construction-time numpy + healpy + scipy, mirroring the `hp/topology.py` philosophy (these are one-off data-prep helpers, off the model's hot path). Two public functions (`img2healpix` ray-tracing binner, `img2healpix_planar` griddata projection) plus two private helpers. A `.ipynb` imports them and visualizes with `hp.mollview(..., nest=True)`.

**Tech Stack:** numpy, healpy, scipy (`ndimage.zoom`, `interpolate.griddata`), matplotlib, HuggingFace `datasets`, Jupyter.

## Global Constraints

- Python >= 3.12; project is uv-managed (`uv sync`, `uv run`).
- **No PyTorch and no numba** in the new module.
- HEALPix maps are emitted in **NEST ordering by default** (`nest=True`); `nest=False` gives healpy-default RING.
- Examples are **not** part of the pytest suite — verification is via manual invariant scripts kept in the scratchpad, not committed pytests.
- `references/` is read-only — consult but never import from it at runtime.
- Do not edit `src/`; all new code lives under `examples/` and the `pyproject.toml` examples extra.
- Scratchpad for temporary scripts: `/tmp/aamerio/claude-6356/-lustre-ific-uv-es-ml-ific088-github-HEAL-SWIN-nnx/c680a2b4-85dd-4d0a-a74a-d6173d6c4745/scratchpad`

---

### Task 1: Declare example dependencies

**Files:**
- Modify: `pyproject.toml` (the `[project.optional-dependencies].examples` list)

**Interfaces:**
- Consumes: nothing.
- Produces: a synced `examples` extra providing `matplotlib`, `scipy`, `notebook`, `datasets` — later tasks assume `import matplotlib`, `import scipy`, `import datasets`, and `jupyter nbconvert` all work.

- [ ] **Step 1: Confirm matplotlib is currently missing (the failing check)**

Run:
```bash
cd /lustre/ific.uv.es/ml/ific088/github/HEAL-SWIN-nnx
.venv/bin/python -c "import matplotlib" ; echo "exit=$?"
```
Expected: `ModuleNotFoundError: No module named 'matplotlib'` and `exit=1`.

- [ ] **Step 2: Add matplotlib and scipy to the examples extra**

Edit `pyproject.toml` so the extra reads:
```toml
[project.optional-dependencies]
examples = [
    "notebook>=7.6.0",
    "datasets>=5.0.0",
    "matplotlib>=3.9.0",
    "scipy>=1.15.0",
]
```

- [ ] **Step 3: Sync the examples extra**

Run:
```bash
cd /lustre/ific.uv.es/ml/ific088/github/HEAL-SWIN-nnx && uv sync --extra examples
```
Expected: resolves and installs matplotlib (and scipy if not already); exit 0.

- [ ] **Step 4: Verify the imports the notebook needs**

Run:
```bash
cd /lustre/ific.uv.es/ml/ific088/github/HEAL-SWIN-nnx && uv run --extra examples python -c "import matplotlib, scipy, datasets, healpy, numpy; print('ok')"
```
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/HEAL-SWIN-nnx
git add pyproject.toml uv.lock
git commit -m "build: add matplotlib and scipy to examples extra"
```

---

### Task 2: Implement `examples/projections.py`

**Files:**
- Create: `examples/projections.py`
- Verify (scratch, not committed): `<scratchpad>/verify_projections.py`

**Interfaces:**
- Consumes: numpy, healpy, `scipy.ndimage.zoom`, `scipy.interpolate.griddata` (from Task 1).
- Produces:
  - `img2healpix(img, nside, delta_theta, delta_phi, rot=np.eye(3), nest=True) -> tuple[np.ndarray, np.ndarray]` — returns `(hp_map, hits)`, each shape `(hp.nside2npix(nside),)`; `img` is 2D `(H, W)`.
  - `img2healpix_planar(img, nside, thetac, phic, delta_theta, delta_phi, nest=True) -> np.ndarray` — `img` is 2D `(H, W)` → returns `(npix,)`, or 3D `(N, H, W)` → returns `(N, npix)`.
  - Private `_binned_map(signal, pixidx, mappixels, hits) -> None` and `_img2map(img, resultmap, resulthits, delta_theta, delta_phi, rot=np.eye(3), nest=True) -> None`.

- [ ] **Step 1: Write the failing verification script**

Create `<scratchpad>/verify_projections.py` (uses a synthetic image — no network):
```python
import sys
sys.path.insert(0, "examples")
import numpy as np
import healpy as hp
from projections import img2healpix, img2healpix_planar

nside = 32

# A bright square in the middle of a 28x28 frame.
img = np.zeros((28, 28))
img[8:20, 8:20] = 1.0

# --- ray-traced binner: map + hits, NEST ordering ---
m, hits = img2healpix(img, nside, 60.0, 60.0)
assert m.shape == (hp.nside2npix(nside),), m.shape
assert hits.shape == m.shape
assert hits.sum() > 0, "no rays landed"
assert (hits > 0).sum() < hp.nside2npix(nside), "projection covers whole sphere (should be confined)"
# averaged, so within [0, 1]
assert m.max() <= 1.0 + 1e-9 and m.min() >= 0.0

# --- NEST vs RING really differ ---
m_ring, _ = img2healpix(img, nside, 60.0, 60.0, nest=False)
assert not np.array_equal(m, m_ring), "NEST and RING maps identical -> nest flag ignored"

# --- planar: 2D in -> 1D out; 3D in -> 2D out ---
p2 = img2healpix_planar(img, nside, 90.0, 0.0, 60.0, 60.0)
assert p2.shape == (hp.nside2npix(nside),), p2.shape
p3 = img2healpix_planar(img[np.newaxis], nside, 90.0, 0.0, 60.0, 60.0)
assert p3.shape == (1, hp.nside2npix(nside)), p3.shape
assert np.array_equal(p3[0], p2), "2D and 3D planar paths disagree"
assert (p2 != 0).sum() > 0, "planar projection empty"

print("ALL OK")
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
cd /lustre/ific.uv.es/ml/ific088/github/HEAL-SWIN-nnx && uv run --extra examples python "<scratchpad>/verify_projections.py"
```
Expected: FAIL with `ModuleNotFoundError: No module named 'projections'`.

- [ ] **Step 3: Write the module**

Create `examples/projections.py`:
```python
# -*- coding: utf-8 -*-
"""Project 2D images onto HEALPix maps.

A dependency-light reimplementation of the two projection helpers from the
upstream NNhealpix reference (``references/NNhealpix/nnhealpix/projections``),
adapted for HEAL-SWIN-nnx:

* the numba ``binned_map`` loop is replaced by vectorized ``numpy.add.at``;
* maps are emitted in **NEST** ordering by default (``nest=True``) to match the
  HealSwin model convention — pass ``nest=False`` for healpy-default RING, e.g.
  to feed ``hp.mollview`` without ``nest=True``.

These run once at data-prep time (off the model's hot path), so construction-time
numpy is entirely adequate.
"""

import numpy as np
import healpy as hp
from scipy.ndimage import zoom
from scipy.interpolate import griddata


def _binned_map(signal, pixidx, mappixels, hits):
    """Bin a 1D signal into a HEALPix map in place, averaging per pixel.

    ``mappixels`` and ``hits`` must be pre-zeroed arrays of length npix. After
    the call, ``mappixels`` holds the mean signal per hit pixel and ``hits`` the
    number of samples that landed in each pixel.
    """
    assert len(mappixels) == len(hits)
    assert len(signal) == len(pixidx)
    np.add.at(mappixels, pixidx, signal)
    np.add.at(hits, pixidx, 1)
    seen = hits > 0
    mappixels[seen] /= hits[seen]


def _img2map(img, resultmap, resulthits, delta_theta, delta_phi,
             rot=np.eye(3), nest=True):
    """Ray-trace a 2D image onto a preallocated HEALPix map (in place).

    Fires a dense grid of rays around (theta=pi/2, phi=0), rotates them by
    ``rot`` (a 3x3 matrix or ``hp.rotator.Rotator``), and bins the interpolated
    image values into ``resultmap`` / ``resulthits``.
    """
    assert img.ndim == 2
    assert len(resultmap) == len(resulthits)
    assert delta_theta > 0.0
    assert delta_phi > 0.0

    nside = hp.npix2nside(len(resultmap))
    delta_theta, delta_phi = np.deg2rad(delta_theta), np.deg2rad(delta_phi)

    rotmatr = rot.mat if isinstance(rot, hp.rotator.Rotator) else rot

    # Fire rays spaced at half the map resolution so no frame pixel is missed.
    map_resolution = 0.5 * hp.nside2resol(nside, arcmin=False)
    nx, ny = [max(1, int(span / map_resolution))
              for span in (delta_theta, delta_phi)]
    theta_proj = np.linspace((np.pi - delta_theta) / 2,
                             (np.pi + delta_theta) / 2, nx)
    phi_proj = np.linspace(delta_phi / 2, -delta_phi / 2, ny)

    # Upsample the image to the ray grid (nearest, order=0).
    proj_img = zoom(img, (nx / img.shape[1], ny / img.shape[0]), order=0)

    theta_proj, phi_proj = np.meshgrid(theta_proj, phi_proj)
    dirs = hp.ang2vec(theta_proj, phi_proj)              # nx x ny x 3
    rotdirs = np.tensordot(dirs, rotmatr, (2, 1))
    theta, phi = hp.vec2ang(np.reshape(rotdirs, (-1, 3)))
    pixidx = hp.ang2pix(nside, theta, phi, nest=nest)

    _binned_map(np.ravel(proj_img), pixidx, resultmap, resulthits)


def img2healpix(img, nside, delta_theta, delta_phi, rot=np.eye(3), nest=True):
    """Project a single 2D image onto a HEALPix map by ray tracing.

    Args:
        img: 2D array ``(H, W)`` to project.
        nside: HEALPix ``NSIDE`` of the output map.
        delta_theta: angular width of the image along the meridian (degrees).
        delta_phi: angular height of the image along the meridian (degrees).
        rot: a 3x3 rotation matrix or ``hp.rotator.Rotator`` placing the image.
        nest: if True (default) the map is in NEST ordering, else RING.

    Returns:
        ``(hp_map, hits)`` — the averaged map and the per-pixel hit count, each
        of shape ``(hp.nside2npix(nside),)``. Unseen pixels are zero.
    """
    assert hp.isnsideok(nside)
    assert delta_theta < 180.0
    assert delta_phi < 180.0

    result = np.zeros(hp.nside2npix(nside))
    hits = np.zeros(result.size, dtype=int)
    _img2map(img, result, hits, delta_theta, delta_phi, rot, nest=nest)
    return result, hits


def img2healpix_planar(img, nside, thetac, phic, delta_theta, delta_phi,
                       nest=True):
    """Project image(s) onto a HEALPix map centered at ``(thetac, phic)``.

    Uses nearest-neighbour interpolation (``scipy.interpolate.griddata``) from
    the image grid onto the HEALPix pixels falling inside the frame.

    Args:
        img: 2D array ``(H, W)`` or a batch ``(N, H, W)``.
        nside: HEALPix ``NSIDE`` of the output map.
        thetac, phic: center of the projection (degrees), HEALPix convention
            (``0 <= thetac <= 180`` from N to S pole; ``0 <= phic <= 360``).
        delta_theta, delta_phi: angular size of the projected image (degrees).
        nest: if True (default) the map is in NEST ordering, else RING.

    Returns:
        A map of shape ``(npix,)`` for a 2D input, or ``(N, npix)`` for a batch.
        Unseen pixels are zero.
    """
    img = np.asarray(img)
    squeeze = img.ndim == 2
    if squeeze:
        img = img[np.newaxis]
    assert img.ndim == 3

    imgf = np.flip(img, axis=2)
    data = imgf.reshape(img.shape[0], img.shape[1] * img.shape[2])
    xsize = img.shape[1]
    ysize = img.shape[2]

    theta_min = np.radians(thetac - delta_theta / 2.0)
    theta_max = np.radians(thetac + delta_theta / 2.0)
    phi_min = np.radians(phic - delta_phi / 2.0)
    phi_max = np.radians(phic + delta_phi / 2.0)

    img_theta_temp = np.linspace(theta_min, theta_max, ysize)
    img_phi_temp = np.linspace(phi_min, phi_max, xsize)

    ipix = np.arange(hp.nside2npix(nside))
    theta_r, phi_r = hp.pix2ang(nside, ipix, nest=nest)

    # Keep only pixels inside the image frame.
    flg = np.where(theta_r < theta_min, 0, 1)
    flg *= np.where(theta_r > theta_max, 0, 1)
    if phi_min >= 0:
        flg *= np.where(phi_r < phi_min, 0, 1)
        flg *= np.where(phi_r > phi_max, 0, 1)
    else:
        # Frame straddles phi=0: handle the wrap-around.
        phi1 = 2.0 * np.pi + phi_min
        phi2 = phi_max
        flg *= np.where((phi2 < phi_r) & (phi_r < phi1), 0, 1)
        img_phi_temp[img_phi_temp < 0] += 2 * np.pi

    img_phi, img_theta = np.meshgrid(img_phi_temp, img_theta_temp)
    img_theta = img_theta.flatten()
    img_phi = img_phi.flatten()

    ipix = np.compress(flg, ipix)
    pl_theta = np.compress(flg, theta_r)
    pl_phi = np.compress(flg, phi_r)

    points = np.zeros((len(img_theta), 2))
    points[:, 0] = img_theta
    points[:, 1] = img_phi

    npix = hp.nside2npix(nside)
    hp_map = np.zeros((data.shape[0], npix))
    for i in range(data.shape[0]):
        hp_map[i, ipix] = griddata(
            points, data[i, :], (pl_theta, pl_phi), method="nearest"
        )

    return hp_map[0] if squeeze else hp_map
```

- [ ] **Step 4: Run the verification script to verify it passes**

Run:
```bash
cd /lustre/ific.uv.es/ml/ific088/github/HEAL-SWIN-nnx && uv run --extra examples python "<scratchpad>/verify_projections.py"
```
Expected: `ALL OK`.

- [ ] **Step 5: Commit**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/HEAL-SWIN-nnx
git add examples/projections.py
git commit -m "feat(examples): NEST-default img2healpix / img2healpix_planar"
```

---

### Task 3: Create `examples/mnist_to_healpix.ipynb`

**Files:**
- Create: `examples/mnist_to_healpix.ipynb`
- Build script (scratch, not committed): `<scratchpad>/build_notebook.py`

**Interfaces:**
- Consumes: `img2healpix`, `img2healpix_planar` from Task 2; `datasets`, `matplotlib`, `healpy` from Task 1.
- Produces: a runnable notebook (final deliverable; nothing depends on it downstream).

- [ ] **Step 1: Write the notebook builder script**

Create `<scratchpad>/build_notebook.py` (uses `nbformat`, installed via the `notebook` package in Task 1):
```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md(
    "# Project 2D flat MNIST images onto the sphere\n\n"
    "This notebook mirrors the upstream `MNIST2healpix` demo, but uses the "
    "HuggingFace `datasets` library to load MNIST and the local "
    "`projections` module (NEST-ordered output) instead of NNhealpix.\n\n"
    "Run with the `examples` extra installed: `uv sync --extra examples`."
))

cells.append(code(
    "import numpy as np\n"
    "import healpy as hp\n"
    "import matplotlib.pyplot as plt\n"
    "%matplotlib inline\n\n"
    "from datasets import load_dataset\n\n"
    "from projections import img2healpix, img2healpix_planar"
))

cells.append(md(
    "## Load one digit from MNIST\n\n"
    "HuggingFace returns PIL images, so we convert to a numpy array. We only "
    "need a single image to demonstrate the projection."
))
cells.append(code(
    "mnist = load_dataset('mnist')\n"
    "sample = mnist['train'][1140]\n"
    "img = np.array(sample['image'], dtype=float)\n"
    "print('label:', sample['label'], 'shape:', img.shape)"
))

cells.append(md(
    "## Planar projection\n\n"
    "`img2healpix_planar` centers the image at `(thetac, phic)` and fills the "
    "covered HEALPix pixels by nearest-neighbour interpolation. The map is in "
    "NEST ordering, so we pass `nest=True` to `hp.mollview`."
))
cells.append(code(
    "img_hp = img2healpix_planar(img, nside=128, thetac=90, phic=0,\n"
    "                            delta_theta=100, delta_phi=100)\n\n"
    "plt.figure(); plt.imshow(img); plt.title('source digit'); plt.axis('off')\n"
    "hp.mollview(img_hp, nest=True, title='planar projection')\n"
    "hp.graticule()"
))

cells.append(md(
    "## Ray-traced projection and hit map\n\n"
    "`img2healpix` fires rays through the image and bins them, returning both "
    "the map and a hit map. Plotting the hit map confirms the projection leaves "
    "no gaps within the image frame."
))
cells.append(code(
    "img_hp, img_hits = img2healpix(img, nside=128,\n"
    "                               delta_theta=100, delta_phi=100)\n\n"
    "hp.mollview(img_hp, nest=True, title='ray-traced projection')\n"
    "hp.graticule()\n"
    "hp.mollview(img_hits, nest=True, title='hit map')\n"
    "hp.graticule()"
))

cells.append(md(
    "## Wrapping around the pole\n\n"
    "The same projection with a rotation, so the image drapes over the pole."
))
cells.append(code(
    "rot = hp.rotator.Rotator(rot=(120.0, 45.0, 30.0))\n"
    "img_hp, img_hits = img2healpix(img, nside=128,\n"
    "                               delta_theta=100, delta_phi=100, rot=rot)\n\n"
    "hp.mollview(img_hp, nest=True, title='rotated projection')\n"
    "hp.graticule()\n"
    "hp.mollview(img_hits, nest=True, title='rotated hit map')\n"
    "hp.graticule()"
))

nb['cells'] = cells
nb.metadata.kernelspec = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
with open("examples/mnist_to_healpix.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote examples/mnist_to_healpix.ipynb")
```

- [ ] **Step 2: Build the notebook**

Run:
```bash
cd /lustre/ific.uv.es/ml/ific088/github/HEAL-SWIN-nnx && uv run --extra examples python "<scratchpad>/build_notebook.py"
```
Expected: `wrote examples/mnist_to_healpix.ipynb`.

- [ ] **Step 3: Execute the notebook end-to-end**

Run:
```bash
cd /lustre/ific.uv.es/ml/ific088/github/HEAL-SWIN-nnx && uv run --extra examples jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 examples/mnist_to_healpix.ipynb ; echo "exit=$?"
```
Expected: `exit=0` and no traceback. This downloads MNIST via HuggingFace on first run.

**If this step fails only because the HuggingFace download has no network access:** fall back to verifying the notebook logic against a synthetic image, and record in the completion note that live execution was blocked by the sandbox network. Run:
```bash
cd /lustre/ific.uv.es/ml/ific088/github/HEAL-SWIN-nnx && uv run --extra examples python -c "
import numpy as np, healpy as hp, matplotlib; matplotlib.use('Agg')
import sys; sys.path.insert(0, 'examples')
from projections import img2healpix, img2healpix_planar
img = np.zeros((28,28)); img[8:20,8:20]=1.0
p = img2healpix_planar(img,128,90,0,100,100); print('planar', p.shape)
m,h = img2healpix(img,128,100,100); print('ray', m.shape, 'hits', int(h.sum()))
r = hp.rotator.Rotator(rot=(120.,45.,30.))
m,h = img2healpix(img,128,100,100,rot=r); print('rot hits', int(h.sum()))
print('NOTEBOOK-LOGIC-OK')
"
```
Expected: prints shapes and `NOTEBOOK-LOGIC-OK`.

- [ ] **Step 4: Confirm the executed notebook has no error outputs**

Run:
```bash
cd /lustre/ific.uv.es/ml/ific088/github/HEAL-SWIN-nnx && uv run --extra examples python -c "
import nbformat
nb = nbformat.read('examples/mnist_to_healpix.ipynb', as_version=4)
errs = [o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.get('output_type')=='error']
print('error outputs:', len(errs))
assert not errs
print('CLEAN')
"
```
Expected: `error outputs: 0` and `CLEAN`. (Skip if Step 3 fell back to synthetic verification because the notebook was never executed.)

- [ ] **Step 5: Commit**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/HEAL-SWIN-nnx
git add examples/mnist_to_healpix.ipynb
git commit -m "docs(examples): MNIST->HEALPix notebook using HuggingFace datasets"
```

---

## Notes for the implementer

- Replace every `<scratchpad>` placeholder with the absolute scratchpad path listed under Global Constraints.
- The `zoom(img, (nx/img.shape[1], ny/img.shape[0]), order=0)` scaling in `_img2map` is intentionally kept identical to upstream — do not "fix" the axis pairing; it preserves the ray-count/pixel-count match with the meshgrid.
- `img2healpix_planar` drops the upstream `rot` parameter, which was never implemented there ("not implemented yet!"); the signature in the spec has no `rot`.
