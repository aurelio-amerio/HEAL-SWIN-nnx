"""HEALPix window shifting. Port of models_torch/hp_shifting.py.

All index/mask computation is numpy at construction time; the nnx wrapper
classes store the results as Buffers and apply pure gathers/rolls at runtime.

The *_TABLES constants encode base-pixel adjacency for the 8-base-pixel
fisheye subset used by the reference. Full-sphere (base_pix=12) tables are a
planned extension (see the design spec); unsupported base_pix raises
NotImplementedError loudly rather than computing garbage.
"""
import math

import jax.numpy as jnp
import numpy as np
from flax import nnx

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
