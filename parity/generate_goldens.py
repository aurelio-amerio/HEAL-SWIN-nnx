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


def run_leaf(name, module, x, meta, call=None):
    module.eval()
    x = x.clone().detach().requires_grad_(True)
    y = call(module, x) if call is not None else module(x)
    y.sum().backward()
    arrays = {"input": to_np(x), "output": to_np(y), "input_grad": to_np(x.grad)}
    for k, v in module.state_dict().items():
        arrays["sd/%s" % k] = to_np(v)
    for k, p in module.named_parameters():
        if p.grad is not None:
            arrays["grad/%s" % k] = to_np(p.grad)
    save_case(name, arrays, meta)


def randn(*shape):
    g = torch.Generator().manual_seed(1234)
    return torch.randn(*shape, generator=g)


def gen_leaves():
    HP_DS = DataSpec(dim_in=2048, f_in=3, f_out=5, base_pix=8,
                     class_names=["c%d" % i for i in range(5)])
    FLAT_DS = DataSpec(dim_in=(32, 64), f_in=3, f_out=5, base_pix=None,
                       class_names=["c%d" % i for i in range(5)])

    torch.manual_seed(0)
    run_leaf("leaf_mlp", hp.Mlp(12, 48), randn(4, 32, 12), {"in_features": 12, "hidden_features": 48})

    for name, kw in [("leaf_hp_attn", {}),
                     ("leaf_hp_attn_relbias", {"rel_pos_bias": "flat"}),
                     ("leaf_hp_attn_cos", {"use_cos_attn": True})]:
        torch.manual_seed(0)
        m = hp.WindowAttention(dim=12, window_size=4, num_heads=2, **kw)
        if name == "leaf_hp_attn_relbias":
            with torch.no_grad():  # table is zeros at init; make the bias path non-trivial
                m.relative_position_bias_table.normal_(0, 0.02)
        run_leaf(name, m, randn(16, 4, 12), {"dim": 12, "window_size": 4, "num_heads": 2, **kw})

    torch.manual_seed(0)
    run_leaf("leaf_hp_patch_merging", hp.PatchMerging(dim=12), randn(2, 128, 12), {"dim": 12})
    torch.manual_seed(0)
    run_leaf("leaf_hp_patch_expand", hp.PatchExpand(dim=24), randn(2, 32, 24), {"dim": 24})
    torch.manual_seed(0)
    run_leaf("leaf_hp_final_expand", hp.FinalPatchExpand_X4(patch_size=4, dim=12),
             randn(2, 512, 12), {"patch_size": 4, "dim": 12})

    torch.manual_seed(0)
    hp_cfg = hp.SwinHPTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4],
                                        drop_path_rate=0.0)
    run_leaf("leaf_hp_patch_embed", hp.PatchEmbed(hp_cfg, HP_DS), randn(2, 3, 2048),
             {"patch_size": 4, "embed_dim": 12, "f_in": 3, "dim_in": 2048})

    for name, kw in [("leaf_hp_block_noshift", {"shift_size": 0}),
                     ("leaf_hp_block_nestroll", {"shift_size": 2, "shift_strategy": "nest_roll"}),
                     ("leaf_hp_block_grid", {"shift_size": 2, "shift_strategy": "nest_grid_shift"}),
                     ("leaf_hp_block_ring", {"shift_size": 2, "shift_strategy": "ring_shift"})]:
        torch.manual_seed(0)
        m = hp.SwinTransformerBlock(dim=12, input_resolution=512, base_pix=8, num_heads=2,
                                    window_size=4, **kw)
        run_leaf(name, m, randn(2, 512, 12),
                 {"dim": 12, "input_resolution": 512, "base_pix": 8, "num_heads": 2,
                  "window_size": 4, **kw})

    for name, kw in [("leaf_flat_attn", {}),
                     ("leaf_flat_attn_norelbias", {"use_rel_pos_bias": False}),
                     ("leaf_flat_attn_cos", {"use_cos_attn": True})]:
        torch.manual_seed(0)
        m = flat.WindowAttention(dim=12, window_size=(4, 4), num_heads=2, **kw)
        run_leaf(name, m, randn(8, 16, 12), {"dim": 12, "window_size": [4, 4], "num_heads": 2, **kw})

    torch.manual_seed(0)
    run_leaf("leaf_flat_patch_merging", flat.PatchMerging(input_resolution=(8, 16), dim=12),
             randn(2, 128, 12), {"input_resolution": [8, 16], "dim": 12})
    torch.manual_seed(0)
    run_leaf("leaf_flat_patch_expand", flat.PatchExpand(input_resolution=(4, 8), dim=24),
             randn(2, 32, 24), {"input_resolution": [4, 8], "dim": 24})
    torch.manual_seed(0)
    run_leaf("leaf_flat_final_expand",
             flat.FinalPatchExpand_X4(input_resolution=(8, 16), patch_size=(4, 4), dim=12),
             randn(2, 128, 12), {"input_resolution": [8, 16], "patch_size": [4, 4], "dim": 12})

    torch.manual_seed(0)
    flat_cfg = flat.SwinTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4],
                                          drop_path_rate=0.0)
    flat.SwinTransformerSys(flat_cfg, FLAT_DS)  # normalizes cfg tuple fields in-place
    torch.manual_seed(0)
    run_leaf("leaf_flat_patch_embed", flat.PatchEmbed(flat_cfg, FLAT_DS), randn(2, 3, 32, 64),
             {"patch_size": [4, 4], "embed_dim": 12, "f_in": 3, "dim_in": [32, 64]})

    for name, kw in [("leaf_flat_block_noshift", {"shift_size": [0, 0]}),
                     ("leaf_flat_block_shift", {"shift_size": [2, 2]}),
                     ("leaf_flat_block_nomask", {"shift_size": [2, 2], "use_masking": False})]:
        torch.manual_seed(0)
        m = flat.SwinTransformerBlock(dim=12, input_resolution=(8, 16), num_heads=2,
                                      window_size=[4, 4], **kw)
        run_leaf(name, m, randn(2, 128, 12),
                 {"dim": 12, "input_resolution": [8, 16], "num_heads": 2,
                  "window_size": [4, 4], **kw})


HP_CASES = {
    "hp_base": {},
    "hp_grid": {"shift_strategy": "nest_grid_shift"},
    "hp_ring": {"shift_strategy": "ring_shift"},
    "hp_cos_v2": {"use_cos_attn": True, "use_v2_norm_placement": True},
    "hp_relbias": {"rel_pos_bias": "flat"},
    "hp_ape": {"ape": True},
}
FLAT_CASES = {
    "flat_base": {},
    "flat_cos_v2": {"use_cos_attn": True, "use_v2_norm_placement": True},
    "flat_norelbias": {"use_rel_pos_bias": False},
    "flat_nomask": {"use_masking": False},
    "flat_ape": {"ape": True},
}


def run_model(name, model, x, hooks, meta):
    model.eval()
    inter = {}

    def mk(key):
        def hook(_mod, _inp, out):
            inter[key] = to_np(out)
        return hook

    handles = [m.register_forward_hook(mk(k)) for k, m in hooks]
    x = x.clone().detach().requires_grad_(True)
    y = model(x)
    y.sum().backward()
    for h in handles:
        h.remove()
    arrays = {"input": to_np(x), "output": to_np(y), "input_grad": to_np(x.grad)}
    for k, v in model.state_dict().items():
        arrays["sd/%s" % k] = to_np(v)
    for k, p in model.named_parameters():
        if p.grad is not None:
            arrays["grad/%s" % k] = to_np(p.grad)
    for k, v in inter.items():
        arrays["int/%s" % k] = v
    save_case(name, arrays, meta)


def gen_models():
    HP_DS = DataSpec(dim_in=2048, f_in=3, f_out=5, base_pix=8,
                     class_names=["c%d" % i for i in range(5)])
    FLAT_DS = DataSpec(dim_in=(32, 64), f_in=3, f_out=5, base_pix=None,
                       class_names=["c%d" % i for i in range(5)])

    for name, over in HP_CASES.items():
        torch.manual_seed(0)
        cfg = hp.SwinHPTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4],
                                         drop_path_rate=0.0, **over)
        model = hp.SwinHPTransformerSys(cfg, HP_DS)
        if over.get("rel_pos_bias") == "flat":
            with torch.no_grad():  # tables init to zeros; randomize so the bias path is exercised
                for mod in model.modules():
                    if hasattr(mod, "relative_position_bias_table"):
                        mod.relative_position_bias_table.normal_(0, 0.02)
        hooks = [("patch_embed", model.patch_embed)]
        hooks += [("enc_layer_%d" % i, l) for i, l in enumerate(model.layers)]
        hooks += [("enc_norm", model.norm)]
        hooks += [("dec_layer_up_%d" % i, l) for i, l in enumerate(model.decoder.layers_up)]
        hooks += [("dec_norm_up", model.decoder.norm_up), ("dec_up", model.decoder.up)]
        run_model(name, model, randn(2, 3, 2048), hooks,
                  {"overrides": over, "embed_dim": 12, "depths": [2, 2], "num_heads": [2, 4],
                   "data_spec": {"dim_in": 2048, "f_in": 3, "f_out": 5, "base_pix": 8}})

    for name, over in FLAT_CASES.items():
        torch.manual_seed(0)
        cfg = flat.SwinTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4],
                                         drop_path_rate=0.0, **over)
        model = flat.SwinTransformerSys(cfg, FLAT_DS)
        hooks = [("patch_embed", model.patch_embed)]
        hooks += [("enc_layer_%d" % i, l) for i, l in enumerate(model.layers)]
        hooks += [("enc_norm", model.norm)]
        hooks += [("dec_layer_up_%d" % i, l) for i, l in enumerate(model.layers_up)]
        hooks += [("dec_norm_up", model.norm_up), ("dec_up", model.up)]
        run_model(name, model, randn(2, 3, 32, 64), hooks,
                  {"overrides": over, "embed_dim": 12, "depths": [2, 2], "num_heads": [2, 4],
                   "data_spec": {"dim_in": [32, 64], "f_in": 3, "f_out": 5}})


if __name__ == "__main__":
    main()
