import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from heal_swin_nnx import layers, swin_hp_transformer as hp
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


def _leaf_forward_and_grads(m, npz, mask=None):
    m.eval()
    x = jnp.asarray(npz["input"])
    call = (lambda mod, x: mod(x, mask=mask)) if mask is not None else (lambda mod, x: mod(x))
    np.testing.assert_allclose(np.asarray(call(m, x)), npz["output"], **FWD)
    gx = jax.grad(lambda x: call(m, x).sum())(x)
    np.testing.assert_allclose(np.asarray(gx), npz["input_grad"], **GRD)
    gp = nnx.grad(lambda m: call(m, x).sum())(m)
    check_param_grads(gp, grads_of(npz))


def test_hp_window_attention_parity():
    for case in ("leaf_hp_attn", "leaf_hp_attn_relbias", "leaf_hp_attn_cos"):
        npz, meta = load_case(case)
        m = hp.WindowAttention(dim=meta["dim"], window_size=meta["window_size"],
                               num_heads=meta["num_heads"],
                               rel_pos_bias=meta.get("rel_pos_bias"),
                               use_cos_attn=meta.get("use_cos_attn", False),
                               rngs=nnx.Rngs(0))
        load_torch_state(m, state_dict_of(npz))
        _leaf_forward_and_grads(m, npz)


def test_hp_rel_pos_index_buffer_matches_reference():
    npz, _ = load_case("leaf_hp_attn_relbias")
    m = hp.WindowAttention(dim=12, window_size=4, num_heads=2, rel_pos_bias="flat",
                           rngs=nnx.Rngs(0))
    assert np.array_equal(np.asarray(m.relative_position_index.value),
                          npz["sd/relative_position_index"])


def test_hp_patch_merge_expand_parity():
    npz, meta = load_case("leaf_hp_patch_merging")
    m = hp.PatchMerging(dim=meta["dim"], rngs=nnx.Rngs(0))
    load_torch_state(m, state_dict_of(npz))
    _leaf_forward_and_grads(m, npz)

    npz, meta = load_case("leaf_hp_patch_expand")
    m = hp.PatchExpand(dim=meta["dim"], rngs=nnx.Rngs(0))
    load_torch_state(m, state_dict_of(npz))
    _leaf_forward_and_grads(m, npz)

    npz, meta = load_case("leaf_hp_final_expand")
    m = hp.FinalPatchExpand_X4(patch_size=meta["patch_size"], dim=meta["dim"], rngs=nnx.Rngs(0))
    load_torch_state(m, state_dict_of(npz))
    _leaf_forward_and_grads(m, npz)
