import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from heal_swin_nnx import layers
from heal_swin_nnx.weight_transfer import load_torch_state, torch_key_to_path, transform_array
from tests.parity_utils import grads_of, load_case, state_dict_of

FWD = dict(rtol=1e-5, atol=1e-6)
GRD = dict(rtol=1e-4, atol=1e-6)


def check_param_grads(nnx_grads, torch_grads, prefix_map=None):
    flat = {tuple(str(p) for p in path): v for path, v in nnx_grads.flat_state()}
    for tkey, tgrad in torch_grads.items():
        path = tuple(str(p) for p in torch_key_to_path(tkey, prefix_map))
        leaf = path[-1]
        expected = transform_array(tgrad, leaf)
        got = np.asarray(flat[path].value)
        np.testing.assert_allclose(got, expected, err_msg=tkey, **GRD)


def test_mlp_parity():
    npz, meta = load_case("leaf_mlp")
    m = layers.Mlp(meta["in_features"], meta["hidden_features"], rngs=nnx.Rngs(0))
    load_torch_state(m, state_dict_of(npz))
    m.eval()
    x = jnp.asarray(npz["input"])
    np.testing.assert_allclose(np.asarray(m(x)), npz["output"], **FWD)
    gx = jax.grad(lambda x: m(x).sum())(x)
    np.testing.assert_allclose(np.asarray(gx), npz["input_grad"], **GRD)
    gp = nnx.grad(lambda m: m(x).sum())(m)
    check_param_grads(gp, grads_of(npz))


def test_transfer_completeness_raises_on_missing():
    import pytest
    npz, meta = load_case("leaf_mlp")
    m = layers.Mlp(meta["in_features"], meta["hidden_features"], rngs=nnx.Rngs(0))
    sd = state_dict_of(npz)
    sd.pop("fc1.bias")
    with pytest.raises(ValueError):
        load_torch_state(m, sd)
