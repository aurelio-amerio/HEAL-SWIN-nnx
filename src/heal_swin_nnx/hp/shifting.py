"""HEALPix window shifting. Port of models_torch/hp_shifting.py.

All index/mask computation is numpy at construction time; the nnx wrapper
classes store the results as Buffers and apply pure gathers/rolls at runtime.

Topology tables (base-pixel adjacency, shift offsets, masks) are derived at
construction time from the HEALPix face-adjacency data in hp_topology, for the
full sphere or any strictly-increasing subset of the 12 base pixels.
"""
import math

import jax.numpy as jnp
import numpy as np
from flax import nnx

from heal_swin_nnx.hp import topology as hp_topology
from heal_swin_nnx.variables import Buffer


def get_attn_mask_from_mask(mask, window_size):
    """(N,) int-valued region mask -> (nW, ws, ws) float32 attention mask in {0, -100}."""
    mask_windows = np.asarray(mask, dtype=np.float32).reshape(-1, window_size)
    attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
    return np.where(attn_mask != 0, np.float32(-100.0), np.float32(0.0))


def nest_roll_mask(input_resolution, window_size, shift_size):
    img_mask = np.zeros(input_resolution, dtype=np.float32)
    slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    for cnt, s in enumerate(slices):
        img_mask[s] = cnt
    return get_attn_mask_from_mask(img_mask, window_size)


class NoShift(nnx.Module):
    def __init__(self):
        self.attn_mask = None

    def shift(self, x):
        return x

    def shift_back(self, x):
        return x


class NestRollShift(nnx.Module):
    def __init__(self, shift_size, input_resolution, window_size):
        self.shift_size = shift_size
        self.attn_mask = Buffer(jnp.asarray(
            nest_roll_mask(input_resolution, window_size, shift_size)))

    def shift(self, x):
        return jnp.roll(x, -self.shift_size, axis=1)

    def shift_back(self, x):
        return jnp.roll(x, self.shift_size, axis=1)


def _log4(x):
    return int(math.log(x) / math.log(4))


def _get_scale(idx, ws, base_pix_len):
    assert idx % ws == 0
    w_idx = idx // ws
    scale = base_pix_len
    while w_idx % scale != 0:
        scale //= 4
    return _log4(scale)


def _get_offset_dir1(idx, ws, base_pix_len, base_pix_offsets):
    assert idx % ws == 0
    while True:
        scale = _get_scale(idx, ws, base_pix_len)
        idx -= ws * 4 ** scale
        if scale >= _get_scale(idx, ws, base_pix_len):
            break
    offset = sum(ws * 4 ** power for power in range(0, scale + 1))
    if scale == _log4(base_pix_len):
        idx += ws * 4 ** scale
        offset -= base_pix_len * ws
        base_pix = idx // (base_pix_len * ws)
        offset += base_pix_offsets[base_pix] * base_pix_len * ws
    return offset


def _get_offset_dir2(idx, ws, base_pix_len, base_pix_offsets):
    assert idx % ws == 0
    scale = _get_scale(idx, ws, base_pix_len)
    while (idx % (ws * 4 ** (scale + 1))) // (ws * 4 ** scale) == 2:
        idx -= 2 * ws * 4 ** scale
        scale = _get_scale(idx, ws, base_pix_len)
    offset = sum(2 * ws * 4 ** power for power in range(0, scale))
    if scale == _log4(base_pix_len):
        base_pix = idx // (base_pix_len * ws)
        offset += base_pix_offsets[base_pix] * base_pix_len * ws
    return offset


def nest_grid_shift_idcs(nside, base_pixels, window_size):
    base_pixels = list(base_pixels)
    base_pix = len(base_pixels)
    ws = window_size
    npix = base_pix * nside ** 2
    n_windows = npix // ws
    base_pix_len = (npix // base_pix) // ws
    hws, qws = ws // 2, ws // 4
    off1, off2 = hp_topology.derive_offset_tables(base_pixels)

    dir1 = np.zeros(npix, dtype=np.int64)
    for w in range(n_windows):
        first = w * ws
        os_ = _get_offset_dir1(first, ws, base_pix_len, off1)
        dir1[first:first + hws] = np.arange(first - os_ - hws, first - os_)
        dir1[first + hws:first + ws] = np.arange(first, first + hws)
    dir1 %= npix

    dir2 = np.zeros(npix, dtype=np.int64)
    for w in range(n_windows):
        first = w * ws
        os_ = _get_offset_dir2(first, ws, base_pix_len, off2)
        dir2[first:first + qws] = np.arange(first - os_ - hws - qws, first - os_ - hws)
        dir2[first + qws:first + hws] = np.arange(first, first + qws)
        dir2[first + hws:first + hws + qws] = np.arange(first - os_ - qws, first - os_)
        dir2[first + hws + qws:first + ws] = np.arange(first + hws, first + hws + qws)
    dir2 %= npix

    result = dir1[dir2]
    assert np.array_equal(np.sort(result), np.arange(npix)), (
        "shift validation failed for nside=%d, window_size=%d" % (nside, ws))
    return result


def nest_grid_mask(nside, base_pixels, window_size):
    base_pixels = list(base_pixels)
    base_pix = len(base_pixels)
    ws = window_size
    hws, qws = ws // 2, ws // 4
    npix = base_pix * nside ** 2
    base_pix_len = (npix // base_pix) // ws
    masked, carry = hp_topology.derive_mask_faces(
        base_pixels, nside, ws, nest_grid_shift_idcs(nside, base_pixels, ws))
    mask = np.zeros(npix)

    def right_mask_subset(first, size, mask_value):
        if size == ws:
            mask[first:first + qws] = mask_value
            mask[first + hws:first + hws + qws] = mask_value
        else:
            right_mask_subset(first, size // 4, mask_value)
            right_mask_subset(first + 2 * size // 4, size // 4, mask_value)

    def left_mask_subset(first, size, mask_value):
        if size == ws:
            mask[first:first + hws] = mask_value
        else:
            left_mask_subset(first, size // 4, mask_value)
            left_mask_subset(first + size // 4, size // 4, mask_value)

    # Two-phase assembly: all region labels first, then all carry writes. A face can
    # be both masked and a carry target (never true for the reference [0..7], where
    # masked=[4..7] and carry=[0..3] are disjoint; e.g. full sphere: face 11 is masked
    # and also carries for masked face 8), and the carry label on its first qws pixels
    # must survive the face's own left/right region labeling — so carries go last.
    for b in masked:
        left_mask_subset(b * base_pix_len * ws, base_pix_len * ws, b + 1)
        right_mask_subset(b * base_pix_len * ws, base_pix_len * ws, b + 1 + len(masked))
    for b, co in zip(masked, carry):
        if co is not None:
            first_co = co * base_pix_len * ws
            # A self-carry (co == b) means a self-mapped fictitious seam glues
            # face b's own tail into its own head: the carried slots must be
            # isolated from BOTH of b's own region labels (b+1 left, b+1+len
            # (masked) right), so use a label outside that pair.
            label = b + 1 if co != b else b + 1 + 2 * len(masked)
            mask[first_co:first_co + qws] = label
    return mask


class NestGridShift(nnx.Module):
    def __init__(self, nside, base_pixels, window_size):
        base_pixels = list(base_pixels)
        idcs = nest_grid_shift_idcs(nside, base_pixels, window_size)
        self.shift_idcs = Buffer(jnp.asarray(idcs))
        self.back_shift_idcs = Buffer(jnp.asarray(np.argsort(idcs)))
        self.attn_mask = Buffer(jnp.asarray(get_attn_mask_from_mask(
            nest_grid_mask(nside, base_pixels, window_size), window_size)))

    def shift(self, x):
        return jnp.take(x, self.shift_idcs[...], axis=1)

    def shift_back(self, x):
        return jnp.take(x, self.back_shift_idcs[...], axis=1)


class NestGridShiftExact(nnx.Module):
    """Seam-exact grid shift: seam-exact wherever the two faces' local frames
    align (all polar-to-equatorial seams). Attention masking remains at the 8
    pinch points, at the 90-degree-rotated south-south seams, and at coverage
    borders for partial-sky models. See spec §4."""

    def __init__(self, nside, base_pixels, window_size):
        idcs, raw_mask = hp_topology.exact_shift_idcs_and_mask(
            list(base_pixels), nside, window_size)
        self.shift_idcs = Buffer(jnp.asarray(idcs))
        self.back_shift_idcs = Buffer(jnp.asarray(np.argsort(idcs)))
        self.attn_mask = Buffer(jnp.asarray(get_attn_mask_from_mask(raw_mask, window_size)))

    def shift(self, x):
        return jnp.take(x, self.shift_idcs[...], axis=1)

    def shift_back(self, x):
        return jnp.take(x, self.back_shift_idcs[...], axis=1)


# --- ring topology (healpy ring coordinate, cross-base-pixel wrapping) ---
def ring_shift_idcs_and_mask(nside, base_pixels, window_size, shift_size):
    import healpy as hp  # local import: healpy pulls matplotlib; keep module import light

    base_pixels = list(base_pixels)
    n = len(base_pixels)
    pixel_size = nside ** 2
    npix = n * pixel_size

    # roll on the full 12-face sphere in ring order, then restrict to the subset
    ring_idcs = np.arange(12 * pixel_size)
    shifted_ring_idcs_in_nest = hp.pixelfunc.ring2nest(nside, np.roll(ring_idcs, shift_size))
    full_result = shifted_ring_idcs_in_nest[
        hp.pixelfunc.nest2ring(nside, np.arange(12 * pixel_size))]
    sel = hp_topology.local_to_global(base_pixels, nside, np.arange(npix))
    result = hp_topology.global_to_local(base_pixels, nside, full_result[sel])

    mask = np.zeros(npix)
    for i in range(n):
        sl = slice(i * pixel_size, (i + 1) * pixel_size)
        mask[sl][result[sl] < 0] = i + 1

    lost_pix = [np.setdiff1d(np.arange(i * pixel_size, (i + 1) * pixel_size), result)
                for i in range(n)]
    get_lost_from = hp_topology.derive_ring_lost_from(base_pixels)

    # first pass: donor-fed faces (reference behaviour for its faces 4..7)
    leftover, donated = [], set()
    for i in range(n):
        if i not in get_lost_from:
            continue
        sl = slice(i * pixel_size, (i + 1) * pixel_size)
        result_subset = result[sl]
        holes = np.where(result_subset < 0)[0]
        source_pix = lost_pix[get_lost_from[i]]
        donated.add(get_lost_from[i])
        take = min(holes.shape[0], source_pix.shape[0])
        result_subset[holes[:take]] = source_pix[:take]
        leftover.append(source_pix[take:])
    # pool: donor remainders (reference order), then lost pixels of never-donor faces
    pool = np.concatenate(
        leftover + [lost_pix[i] for i in range(n) if i not in donated]
        or [np.array([], dtype=np.int64)]).astype(np.int64)

    # second pass: remaining holes in face order (reference behaviour for faces 0..3)
    first = 0
    for i in range(n):
        sl = slice(i * pixel_size, (i + 1) * pixel_size)
        result_subset = result[sl]
        holes = np.where(result_subset < 0)[0]
        result_subset[holes] = pool[first:first + holes.shape[0]]
        first += holes.shape[0]

    result = result.astype(np.int64)
    assert np.array_equal(np.sort(result), np.arange(npix)), (
        "shift validation failed for nside=%d, window_size=%d" % (nside, window_size))
    return result, mask.astype(np.int64)


class RingShift(nnx.Module):
    def __init__(self, nside, base_pixels, window_size, shift_size):
        idcs, raw_mask = ring_shift_idcs_and_mask(nside, list(base_pixels),
                                                  window_size, shift_size)
        self.shift_idcs = Buffer(jnp.asarray(idcs))
        self.back_shift_idcs = Buffer(jnp.asarray(np.argsort(idcs)))
        self.attn_mask = Buffer(jnp.asarray(get_attn_mask_from_mask(raw_mask, window_size)))

    def shift(self, x):
        return jnp.take(x, self.shift_idcs[...], axis=1)

    def shift_back(self, x):
        return jnp.take(x, self.back_shift_idcs[...], axis=1)
