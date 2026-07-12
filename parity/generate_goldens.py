#!/usr/bin/env python3
"""Generate golden parity fixtures from the reference HEAL-SWIN implementation.

Run inside the parity environment:

    cd parity && uv run python generate_goldens.py [--only indices|leaves|models]

Everything is deterministic (fixed seeds). Output: ../tests/goldens/*.npz
"""
import argparse
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless: matplotlib backend autodetect crashes on broken _tkinter

import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "references", "HEAL-SWIN"))

from heal_swin.data.segmentation.data_spec import DataSpec
from heal_swin.models_torch import hp_shifting, hp_windowing
from heal_swin.models_torch import swin_hp_transformer as hp
from heal_swin.models_torch import swin_transformer as flat

OUT_DIR = os.path.join(HERE, "..", "tests", "goldens")
SCHEMA_VERSION = 1


def to_np(t):
    return t.detach().cpu().numpy()


def save_case(name, arrays, meta):
    arrays = dict(arrays)
    arrays["schema_version"] = np.array(SCHEMA_VERSION)
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    path = os.path.join(OUT_DIR, name + ".npz")
    np.savez_compressed(path, **arrays)
    print("wrote %s (%d arrays)" % (path, len(arrays)))


def gen_indices():
    arrays = {}
    for ws in (4, 16):
        arrays["nest_win_idcs/ws%d" % ws] = to_np(hp_windowing.get_nest_win_idcs(ws))
        wa = hp.WindowAttention(dim=4, window_size=ws, num_heads=2, rel_pos_bias="flat")
        arrays["hp_rel_pos_index/ws%d" % ws] = to_np(wa.relative_position_index)
    for nside in (4, 8, 16):
        npix = 8 * nside ** 2
        for ws in (4, 16):
            tag = "ns%d_ws%d" % (nside, ws)
            if npix // ws < 8:
                continue
            nr = hp_shifting.NestRollShift(shift_size=ws // 2, input_resolution=npix, window_size=ws)
            arrays["nest_roll/mask/%s" % tag] = to_np(nr.get_mask())
            if (npix // 8) // ws < 4:
                continue  # too small for grid/ring window traversal
            ng = hp_shifting.NestGridShift(nside=nside, base_pix=8, window_size=ws)
            arrays["nest_grid/idcs/%s" % tag] = to_np(ng.shift_idcs)
            arrays["nest_grid/back/%s" % tag] = to_np(ng.back_shift_idcs)
            arrays["nest_grid/attn_mask/%s" % tag] = to_np(ng.get_mask())
            arrays["nest_grid/mask_raw/%s" % tag] = to_np(ng.get_mask(get_attn_mask=False))
            rs = hp_shifting.RingShift(nside=nside, base_pix=8, window_size=ws, shift_size=ws // 2)
            arrays["ring/idcs/%s" % tag] = to_np(rs.shift_idcs)
            arrays["ring/back/%s" % tag] = to_np(rs.back_shift_idcs)
            arrays["ring/attn_mask/%s" % tag] = to_np(rs.get_mask())
            arrays["ring/mask_raw/%s" % tag] = to_np(rs.get_mask(get_attn_mask=False))
    save_case("indices", arrays, {"base_pix": 8, "nsides": [4, 8, 16], "window_sizes": [4, 16],
                                  "shift_size": "ws//2"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["indices", "leaves", "models"], default=None)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.set_grad_enabled(True)
    if args.only in (None, "indices"):
        gen_indices()
    if args.only in (None, "leaves"):
        gen_leaves()   # Task 4
    if args.only in (None, "models"):
        gen_models()   # Task 5


def gen_leaves():
    pass  # filled in Task 4


def gen_models():
    pass  # filled in Task 5


if __name__ == "__main__":
    main()
