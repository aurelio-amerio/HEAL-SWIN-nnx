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

from heal_swin_nnx import hp_topology
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


# --- 8-base-pixel fisheye-subset topology (reference HEAL-SWIN). Keyed by base_pix
# so that full-sphere tables can be added without touching traversal logic. ---
NEST_GRID_MASKED_BASE_PIX = {8: [4, 5, 6, 7]}
NEST_GRID_LEFT_CARRY_OVER_BASE_PIX = {8: [0, 1, 2, 3]}


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


def nest_grid_mask(nside, base_pix, window_size):
    ws = window_size
    hws, qws = ws // 2, ws // 4
    npix = base_pix * nside ** 2
    base_pix_len = (npix // base_pix) // ws
    masked = NEST_GRID_MASKED_BASE_PIX[base_pix]
    carry = NEST_GRID_LEFT_CARRY_OVER_BASE_PIX[base_pix]
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

    for b, co in zip(masked, carry):
        left_mask_subset(b * base_pix_len * ws, base_pix_len * ws, b + 1)
        right_mask_subset(b * base_pix_len * ws, base_pix_len * ws, b + 1 + len(masked))
        first_co = co * base_pix_len * ws
        mask[first_co:first_co + qws] = b + 1
    return mask


class NestGridShift(nnx.Module):
    def __init__(self, nside, base_pixels, window_size):
        base_pixels = list(base_pixels)
        idcs = nest_grid_shift_idcs(nside, base_pixels, window_size)
        self.shift_idcs = Buffer(jnp.asarray(idcs))
        self.back_shift_idcs = Buffer(jnp.asarray(np.argsort(idcs)))
        self.attn_mask = Buffer(jnp.asarray(get_attn_mask_from_mask(
            nest_grid_mask(nside, len(base_pixels), window_size), window_size)))

    def shift(self, x):
        return jnp.take(x, self.shift_idcs[...], axis=1)

    def shift_back(self, x):
        return jnp.take(x, self.back_shift_idcs[...], axis=1)


# --- ring topology (healpy ring coordinate, cross-base-pixel wrapping) ---
RING_GET_LOST_FROM = {8: {4: 7, 5: 4, 6: 5, 7: 6}}


def ring_shift_idcs_and_mask(nside, base_pix, window_size, shift_size):
    import healpy as hp  # local import: healpy pulls matplotlib; keep module import light

    npix = base_pix * nside ** 2
    get_lost_from = RING_GET_LOST_FROM[base_pix]

    ring_idcs = np.arange(12 * nside ** 2)
    shifted_ring_idcs = np.roll(ring_idcs, shift_size)
    shifted_ring_idcs_in_nest = hp.pixelfunc.ring2nest(nside, shifted_ring_idcs)

    nest_idcs = np.arange(npix)
    nest_idcs_in_ring = hp.pixelfunc.nest2ring(nside, nest_idcs)
    result = shifted_ring_idcs_in_nest[nest_idcs_in_ring]

    max_idx = nest_idcs.max()
    pixel_size = nside ** 2
    mask = np.zeros(npix)
    for i in range(base_pix):
        subset_slice = slice(i * pixel_size, (i + 1) * pixel_size)
        mask_subset = mask[subset_slice]
        result_subset = result[subset_slice]
        mask_subset[result_subset > max_idx] = i + 1

    lost_pix = []
    for i in range(base_pix):
        lost_pix.append(np.setdiff1d(np.arange(i * pixel_size, (i + 1) * pixel_size), result))

    # base pixels 4..7 are the masked/backfilled subset, 0..3 carry over unchanged; this is the
    # 8-base-pixel subset topology guarded by the NotImplementedError above. Full-sphere (phase 2)
    # must derive these ranges from the topology tables instead of hardcoding them.
    unused_source_pix = []
    for i in range(4, base_pix):
        subset_slice = slice(i * pixel_size, (i + 1) * pixel_size)
        result_subset = result[subset_slice]
        source_pix = lost_pix[get_lost_from[i]]
        pix_to_be_filled = result_subset[result_subset > max_idx]
        assert pix_to_be_filled.shape[0] <= source_pix.shape[0], (
            "for base pixel %d, there were not enough source pixel" % i)
        result_subset[result_subset > max_idx] = source_pix[:pix_to_be_filled.shape[0]]
        unused_source_pix.append(source_pix[pix_to_be_filled.shape[0]:])
    unused_pix = np.concatenate(unused_source_pix).flatten()

    assert unused_pix.shape[0] == (result > max_idx).sum(), (
        "the number of unused source pixels does not match the number of pixels to be filled")
    # Same 8-base-pixel subset topology as above: only base pixels 0..3 receive the leftover
    # unused source pixels. Full-sphere (phase 2) must derive this range from the topology tables.
    first = 0
    for i in range(4):
        subset_slice = slice(i * pixel_size, (i + 1) * pixel_size)
        result_subset = result[subset_slice]
        no_to_be_filled = result_subset[result_subset > max_idx].shape[0]
        result_subset[result_subset > max_idx] = unused_pix[first:first + no_to_be_filled]
        first += no_to_be_filled

    result = result.astype(np.int64)
    assert np.array_equal(np.sort(result), np.arange(npix)), (
        "shift validation failed for nside=%d, window_size=%d" % (nside, window_size))
    return result, mask.astype(np.int64)


class RingShift(nnx.Module):
    def __init__(self, nside, base_pix, window_size, shift_size):
        # The reference silently assumes base_pix == 8 here; we assert loudly (spec).
        if base_pix not in RING_GET_LOST_FROM:
            raise NotImplementedError(
                "RingShift backfill tables only exist for base_pix in %s; "
                "full-sphere support is a planned extension (see design spec)"
                % sorted(RING_GET_LOST_FROM))
        idcs, raw_mask = ring_shift_idcs_and_mask(nside, base_pix, window_size, shift_size)
        self.shift_idcs = Buffer(jnp.asarray(idcs))
        self.back_shift_idcs = Buffer(jnp.asarray(np.argsort(idcs)))
        self.attn_mask = Buffer(jnp.asarray(get_attn_mask_from_mask(raw_mask, window_size)))

    def shift(self, x):
        return jnp.take(x, self.shift_idcs[...], axis=1)

    def shift_back(self, x):
        return jnp.take(x, self.back_shift_idcs[...], axis=1)
