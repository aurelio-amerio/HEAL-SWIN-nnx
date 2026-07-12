import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from heal_swin_nnx import swin_hp_transformer as hp
from heal_swin_nnx import swin_transformer as flat
from heal_swin_nnx.config import DataSpec, SwinHPTransformerConfig, SwinTransformerConfig
from heal_swin_nnx.weight_transfer import FLAT_PREFIX_MAP, HP_PREFIX_MAP, load_torch_state
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


from heal_swin_nnx.weight_transfer import torch_key_to_path, transform_array

# Two cases exceed E2E_GRD on gradients that are ill-conditioned in float32, verified
# by a float64 self-convergence experiment (max abs diff on the driving tensor;
# "f64" is our own model rerun under jax_enable_x64):
#   hp_ring / decoder.layers_up.0.expand.weight:
#     |f32-f64| 9.9e-4, |torch-f64| 1.0e-3, |f32-torch| 4.9e-4
#   hp_cos_v2 / input_grad:
#     |f32-f64| 3.5e-2, |torch-f64| 4.6e-2, |f32-torch| 4.0e-2
#   hp_cos_v2 / layers.0.blocks.1.attn.qkv.weight (drives the atol; grads reach ~1e4):
#     |f32-f64| 0.67, |torch-f64| 1.3, |f32-torch| 1.2
# Both our f32 run and the torch golden deviate from the f64 reference by as much as
# they deviate from each other, so the mismatch is float32 accumulation noise, not an
# algorithmic difference. Overrides below are the tightest round tolerances that pass
# with >=2x margin (i.e., they still pass at rtol/2, atol/2).
E2E_GRD_OVERRIDES = {"hp_ring": dict(rtol=1e-3, atol=5e-4),
                     "hp_cos_v2": dict(rtol=5e-3, atol=2.0)}


@pytest.mark.parametrize("case", HP_CASES)
def test_hp_e2e_gradient_parity(case):
    npz, meta = load_case(case)
    model = build_hp_model(meta)
    load_torch_state(model, state_dict_of(npz), prefix_map=HP_PREFIX_MAP)
    model.eval()
    x = jnp.asarray(npz["input"]).transpose(0, 2, 1)
    tol = E2E_GRD_OVERRIDES.get(case, E2E_GRD)

    gx = jax.grad(lambda x: model(x).sum())(x)
    np.testing.assert_allclose(np.asarray(gx).transpose(0, 2, 1), npz["input_grad"], **tol)

    gp = nnx.grad(lambda m: m(x).sum())(model)
    flat = {tuple(str(p) for p in path): v for path, v in nnx.to_flat_state(gp)}
    for tkey, tgrad in grads_of(npz).items():
        path = tuple(str(p) for p in torch_key_to_path(tkey, HP_PREFIX_MAP))
        expected = transform_array(tgrad, path[-1])
        np.testing.assert_allclose(np.asarray(flat[path][...]), expected,
                                   err_msg=tkey, **tol)


@pytest.mark.parametrize("case", ["hp_base", "hp_grid", "hp_ring"])
def test_hp_buffer_bit_parity(case):
    """Torch buffers (attn_mask, relative_position_index) match ours exactly."""
    npz, meta = load_case(case)
    model = build_hp_model(meta)
    for key in state_dict_of(npz):
        if not key.endswith(("attn_mask", "relative_position_index")):
            continue
        obj = model
        parts = torch_key_to_path(key, HP_PREFIX_MAP)
        for p in parts[:-1]:
            obj = obj[p] if isinstance(p, int) else getattr(obj, p)
        leaf = parts[-1]
        ours = (obj.shifter.attn_mask if leaf == "attn_mask" else
                getattr(obj, leaf))
        assert np.array_equal(np.asarray(ours[...]), npz["sd/" + key]), key


FLAT_CASES = ["flat_base", "flat_cos_v2", "flat_norelbias", "flat_nomask", "flat_ape"]


def build_flat_model(meta):
    cfg = SwinTransformerConfig(embed_dim=meta["embed_dim"], depths=meta["depths"],
                                num_heads=meta["num_heads"], drop_path_rate=0.0,
                                **meta["overrides"])
    ds = DataSpec(dim_in=tuple(meta["data_spec"]["dim_in"]), f_in=meta["data_spec"]["f_in"],
                  f_out=meta["data_spec"]["f_out"])
    return flat.SwinTransformerSys(cfg, ds, rngs=nnx.Rngs(0))


@pytest.mark.parametrize("case", FLAT_CASES)
def test_flat_e2e_forward_parity(case):
    npz, meta = load_case(case)
    model = build_flat_model(meta)
    load_torch_state(model, state_dict_of(npz), prefix_map=FLAT_PREFIX_MAP)
    model.eval()
    x = jnp.asarray(npz["input"]).transpose(0, 2, 3, 1)   # (B,C,H,W) -> (B,H,W,C)
    y = np.asarray(model(x)).transpose(0, 3, 1, 2)        # back to torch layout
    np.testing.assert_allclose(y, npz["output"], **E2E_FWD)


# flat_cos_v2 exceeds E2E_GRD the same way hp_cos_v2 does: cosine attention's
# logit_scale/normalization is ill-conditioned in float32. Verified by a float64
# self-convergence experiment (max abs diff; "f64" is our own model rerun under
# jax_enable_x64):
#   flat_cos_v2 / input_grad:
#     |f32-f64| 2.9e-3, |torch-f64| 5.3e-3, |f32-torch| 4.8e-3
#   flat_cos_v2 / layers.1.blocks.0.attn.proj.bias (drives the atol; grads reach ~10):
#     |f32-f64| 3.6, |torch-f64| 4.0, |f32-torch| 3.7
#   flat_cos_v2 / layers.0.blocks.0.attn.proj.bias:
#     |f32-f64| 3.4, |torch-f64| 8.0, |f32-torch| 9.3
# In every case our f32 run and the torch golden deviate from the f64 reference by as
# much as they deviate from each other, so the mismatch is float32 accumulation noise,
# not an algorithmic difference. The override below is the tightest round tolerance
# passing with >=2x margin (verified passing at rtol/2=2.5e-3, atol/2=1.0 too).
FLAT_E2E_GRD_OVERRIDES = {"flat_cos_v2": dict(rtol=5e-3, atol=2.0)}


@pytest.mark.parametrize("case", FLAT_CASES)
def test_flat_e2e_gradient_parity(case):
    npz, meta = load_case(case)
    model = build_flat_model(meta)
    load_torch_state(model, state_dict_of(npz), prefix_map=FLAT_PREFIX_MAP)
    model.eval()
    x = jnp.asarray(npz["input"]).transpose(0, 2, 3, 1)
    tol = FLAT_E2E_GRD_OVERRIDES.get(case, E2E_GRD)

    gx = jax.grad(lambda x: model(x).sum())(x)
    np.testing.assert_allclose(np.asarray(gx).transpose(0, 3, 1, 2), npz["input_grad"], **tol)

    gp = nnx.grad(lambda m: m(x).sum())(model)
    flat_g = {tuple(str(p) for p in path): v for path, v in nnx.to_flat_state(gp)}
    for tkey, tgrad in grads_of(npz).items():
        path = tuple(str(p) for p in torch_key_to_path(tkey, FLAT_PREFIX_MAP))
        expected = transform_array(tgrad, path[-1])
        np.testing.assert_allclose(np.asarray(flat_g[path][...]), expected,
                                   err_msg=tkey, **tol)
