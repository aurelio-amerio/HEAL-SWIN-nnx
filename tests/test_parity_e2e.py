import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from heal_swin_nnx import swin_hp_transformer as hp
from heal_swin_nnx.config import DataSpec, SwinHPTransformerConfig
from heal_swin_nnx.weight_transfer import HP_PREFIX_MAP, load_torch_state
from tests.parity_utils import grads_of, load_case, state_dict_of

E2E_FWD = dict(rtol=1e-4, atol=1e-4)
E2E_GRD = dict(rtol=1e-3, atol=1e-4)

HP_CASES = ["hp_base", "hp_grid", "hp_ring", "hp_cos_v2", "hp_relbias", "hp_ape"]


def build_hp_model(meta):
    cfg = SwinHPTransformerConfig(embed_dim=meta["embed_dim"], depths=meta["depths"],
                                  num_heads=meta["num_heads"], drop_path_rate=0.0,
                                  **meta["overrides"])
    ds = DataSpec(**meta["data_spec"])
    model = hp.SwinHPTransformerSys(cfg, ds, rngs=nnx.Rngs(0))
    return model


@pytest.mark.parametrize("case", HP_CASES)
def test_hp_e2e_forward_parity(case):
    npz, meta = load_case(case)
    model = build_hp_model(meta)
    load_torch_state(model, state_dict_of(npz), prefix_map=HP_PREFIX_MAP)
    model.eval()
    x = jnp.asarray(npz["input"]).transpose(0, 2, 1)          # (B,C,N) -> (B,N,C)
    y = np.asarray(model(x)).transpose(0, 2, 1)               # back to torch layout
    np.testing.assert_allclose(y, npz["output"], **E2E_FWD)


@pytest.mark.parametrize("case", HP_CASES)
def test_hp_encoder_boundary_parity(case):
    npz, meta = load_case(case)
    model = build_hp_model(meta)
    load_torch_state(model, state_dict_of(npz), prefix_map=HP_PREFIX_MAP)
    model.eval()
    x = jnp.asarray(npz["input"]).transpose(0, 2, 1)
    tokens, skips = model.encoder(x)
    np.testing.assert_allclose(np.asarray(tokens), npz["int/enc_norm"], **E2E_FWD)
    np.testing.assert_allclose(np.asarray(skips[1]), npz["int/enc_layer_0"], **E2E_FWD)
    inters = model.decoder(tokens, skips, return_intermediates=True)[1]
    for i, inter in enumerate(inters):
        np.testing.assert_allclose(np.asarray(inter), npz["int/dec_layer_up_%d" % i],
                                   err_msg="dec_layer_up_%d" % i, **E2E_FWD)
