import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references", "HEAL-SWIN"))

import healpy
import numpy
import timm
import torch

from heal_swin.data.segmentation.data_spec import DataSpec
from heal_swin.models_torch import hp_shifting, hp_windowing  # noqa: F401
from heal_swin.models_torch.swin_hp_transformer import SwinHPTransformerConfig, SwinHPTransformerSys
from heal_swin.models_torch.swin_transformer import SwinTransformerConfig, SwinTransformerSys  # noqa: F401

assert torch.__version__.startswith("1.8.0"), torch.__version__
assert numpy.__version__ == "1.19.2", numpy.__version__
assert healpy.__version__ == "1.15.2", healpy.__version__
assert timm.__version__ == "0.4.12", timm.__version__

cfg = SwinHPTransformerConfig(embed_dim=12, depths=[2, 2], num_heads=[2, 4], drop_path_rate=0.0)
ds = DataSpec(dim_in=2048, f_in=3, f_out=5, base_pix=8, class_names=["c%d" % i for i in range(5)])
model = SwinHPTransformerSys(cfg, ds).eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 2048))
assert out.shape == (1, 5, 2048), out.shape
print("parity env OK")
