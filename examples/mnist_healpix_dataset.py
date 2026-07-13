# -*- coding: utf-8 -*-
"""In-memory grain dataset of MNIST digits projected onto HEALPix maps.

Each record is a single MNIST digit ray-traced onto a full-sphere HEALPix map
(NEST ordering, via the local :mod:`projections` module) under an independent
random rotation and random angular extension. The number of samples is
decoupled from the MNIST split size by sampling **with replacement**, so
``num_samples`` may exceed the ~60k training images.

All maps are materialized up front into memory, then wrapped with
``grain.MapDataset.source``. The caller composes the training pipeline on top
(shuffle / repeat / batch / prefetch), mirroring the sbibm-jax grain usage.

Memory footprint is roughly ``num_samples * 12 * nside**2 * 4`` bytes
(e.g. 10k samples at nside 64 is about 2 GB). Pick ``nside`` / ``num_samples``
accordingly.

Run headless as a smoke check:

    uv run --extra examples python examples/mnist_healpix_dataset.py
"""

from __future__ import annotations

import numpy as np
import healpy as hp
from datasets import load_dataset

from projections import img2healpix


def make_mnist_healpix_dataset(
    num_samples: int,
    nside: int = 64,
    split: str = "train",
    delta_range: tuple[float, float] = (50.0, 100.0),
    seed: int = 0,
):
    """Build an in-memory grain dataset of HEALPix-projected MNIST digits.

    Args:
        num_samples: Number of records to generate. May exceed the MNIST split
            size; digits are drawn with replacement.
        nside: HEALPix ``NSIDE`` of the output maps (``npix = 12 * nside**2``).
        split: MNIST split to draw from (``"train"`` or ``"test"``).
        delta_range: ``(low, high)`` degrees; ``delta_theta`` and ``delta_phi``
            are each drawn independently and uniformly from this range, setting
            the angular extent of the projected digit.
        seed: Seed for the single numpy RNG driving indices, extensions, and
            rotation angles (fully reproducible per seed).

    Returns:
        A ``grain.MapDataset`` whose records are dicts
        ``{"image": np.ndarray (npix,) float32, "label": int}`` in NEST order.
        Compose ``.shuffle(seed).repeat().batch(...).to_iter_dataset()`` on the
        result to feed a training loop.
    """
    import grain

    rng = np.random.default_rng(seed)

    mnist = load_dataset("ylecun/mnist")[split]
    images = np.asarray(mnist["image"], dtype=np.float64)  # (M, 28, 28)
    labels = np.asarray(mnist["label"], dtype=np.int64)    # (M,)
    n_mnist = images.shape[0]

    # Sample with replacement so num_samples can exceed the split size.
    idx = rng.integers(0, n_mnist, size=num_samples)
    delta_theta = rng.uniform(delta_range[0], delta_range[1], size=num_samples)
    delta_phi = rng.uniform(delta_range[0], delta_range[1], size=num_samples)
    # Three Euler angles per sample, matching the notebook's
    # ``rot=(120., 45., 10.)`` form. Uniform in [0, 360) is simple and gives
    # varied placements (including pole wrap-around); it is not measure-uniform
    # on SO(3), which is adequate for a demo dataset.
    angles = rng.uniform(0.0, 360.0, size=(num_samples, 3))

    records = []
    for i, mnist_i in enumerate(idx):
        rot = hp.rotator.Rotator(rot=tuple(angles[i]))
        hp_map, _hits = img2healpix(
            images[mnist_i],
            nside=nside,
            delta_theta=float(delta_theta[i]),
            delta_phi=float(delta_phi[i]),
            rot=rot,
            nest=True,
        )
        records.append(
            {"image": hp_map.astype(np.float32), "label": int(labels[mnist_i])}
        )

    return grain.MapDataset.source(records)

