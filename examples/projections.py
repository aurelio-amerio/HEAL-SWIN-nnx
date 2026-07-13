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
