# -*- coding: utf-8 -*-
"""Lazy grain dataset of MNIST digits projected onto HEALPix maps.

Each record is a single MNIST digit ray-traced onto a full-sphere HEALPix map
(NEST ordering, via the local :mod:`projections` module) under a fixed random
rotation and angular extent. A fixed list of ``num_samples`` projection *specs*
(digit index, Euler angles, angular extents) is pre-drawn once from a seeded
RNG; the expensive ray-trace runs **lazily** inside grain workers, so the same
deterministic maps are produced every epoch without materializing them in RAM.

Digits are sampled with replacement, so ``num_samples`` may exceed the MNIST
split size.

Run headless as a smoke check:

    uv run --extra examples python examples/mnist_healpix_dataset.py
"""

from __future__ import annotations

import numpy as np
import healpy as hp
from datasets import load_dataset

from projections import img2healpix


class _Projector:
    """Picklable per-record projection: spec index -> {image, label}.

    Holds the shared MNIST arrays and the pre-drawn spec arrays. grain pickles
    this into each ``mp_prefetch`` worker (copying ``images`` ~37 MB/worker).
    Uses only numpy + healpy, so workers touch no accelerator.
    """

    def __init__(self, images, labels, idx, angles, delta_theta, delta_phi, nside):
        self.images = images
        self.labels = labels
        self.idx = idx
        self.angles = angles
        self.delta_theta = delta_theta
        self.delta_phi = delta_phi
        self.nside = nside

    def __call__(self, i: int):
        j = int(self.idx[i])
        rot = hp.rotator.Rotator(rot=tuple(self.angles[i]))
        hp_map, _hits = img2healpix(
            self.images[j],
            nside=self.nside,
            delta_theta=float(self.delta_theta[i]),
            delta_phi=float(self.delta_phi[i]),
            rot=rot,
            nest=True,
        )
        return {"image": hp_map.astype(np.float32), "label": int(self.labels[j])}


def make_mnist_healpix_dataset(
    num_samples: int,
    nside: int = 64,
    split: str = "train",
    delta_range: tuple[float, float] = (50.0, 100.0),
    seed: int = 0,
):
    """Build a lazy grain dataset of HEALPix-projected MNIST digits.

    Args:
        num_samples: Number of records. May exceed the split size (digits are
            drawn with replacement).
        nside: HEALPix ``NSIDE`` of the output maps (``npix = 12 * nside**2``).
        split: MNIST split to draw from (``"train"`` or ``"test"``).
        delta_range: ``(low, high)`` degrees; ``delta_theta`` and ``delta_phi``
            are each drawn independently and uniformly from this range.
        seed: Seed for the single numpy RNG driving indices, extents, and
            rotation angles (fully reproducible per seed).

    Returns:
        A ``grain.MapDataset`` of ``{"image": (npix,) float32, "label": int}``
        records in NEST order, unbatched and unshuffled. Compose
        ``.shuffle(seed).to_iter_dataset().batch(...).mp_prefetch(...)`` to feed
        a training loop.
    """
    import grain

    rng = np.random.default_rng(seed)

    mnist = load_dataset("ylecun/mnist")[split]
    images = np.asarray(mnist["image"], dtype=np.float64)  # (M, 28, 28)
    labels = np.asarray(mnist["label"], dtype=np.int64)    # (M,)
    n_mnist = images.shape[0]

    # Fixed specs, pre-drawn once (tiny in RAM; the maps stay lazy).
    idx = rng.integers(0, n_mnist, size=num_samples)
    delta_theta = rng.uniform(delta_range[0], delta_range[1], size=num_samples)
    delta_phi = rng.uniform(delta_range[0], delta_range[1], size=num_samples)
    # Three Euler angles per sample (same form as the notebook's
    # ``rot=(120., 45., 10.)``); uniform in [0, 360) is not measure-uniform on
    # SO(3) but is adequate for a demo.
    angles = rng.uniform(0.0, 360.0, size=(num_samples, 3))

    projector = _Projector(images, labels, idx, angles, delta_theta, delta_phi, nside)
    return grain.MapDataset.range(num_samples).map(projector)


if __name__ == "__main__":
    ds = make_mnist_healpix_dataset(num_samples=8, nside=64, seed=0)
    rec = ds[0]
    print("dataset length:", len(ds))
    print("image shape:", rec["image"].shape, rec["image"].dtype)
    print("label:", rec["label"])
    assert rec["image"].shape == (12 * 64 ** 2,)
    assert rec["image"].dtype == np.float32
    assert 0 <= rec["label"] <= 9
    print("smoke check OK")
