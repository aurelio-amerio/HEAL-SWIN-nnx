"""Float64 gradient parity: proves the loosened f32 tolerances on the three
ill-conditioned e2e cases (hp_ring, hp_cos_v2, flat_cos_v2 — see
E2E_GRD_OVERRIDES/FLAT_E2E_GRD_OVERRIDES in tests/test_parity_e2e.py) are
float32 precision noise, not an algorithmic gap: rerunning the same models and
torch goldens in float64 should agree to ~1e-9, far tighter than the f32
overrides. Two controls (hp_base, flat_base) are included for contrast.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from heal_swin_nnx.weight_transfer import (FLAT_PREFIX_MAP, HP_PREFIX_MAP, load_torch_state,
                                           torch_key_to_path, transform_array)
from tests.parity_utils import grads_of, load_case, state_dict_of
from tests.test_parity_e2e import build_flat_model, build_hp_model

# Measured tightest-passing tolerance per case (rtol, atol; max abs diff shown too):
#   hp_base:     (2e-11, 2e-13)  max|diff|=8.6e-12
#   hp_ring:     (5e-12, 5e-14)  max|diff|=5.5e-12
#   hp_cos_v2:   (3e-9,  3e-11)  max|diff|=2.2e-08   <- drives the tolerance below
#   flat_base:   (2e-12, 2e-14)  max|diff|=1.4e-12
#   flat_cos_v2: (5e-10, 5e-12)  max|diff|=1.8e-08
# All comfortably at or below the ~1e-9 rel / ~1e-11 abs expectation and far under
# the rtol=1e-6/atol=1e-8 hard gate. F64_TOL below is the tightest round value with
# >=10x margin over the worst case (hp_cos_v2): it also passes at F64_TOL/10.
F64_TOL = dict(rtol=1e-7, atol=1e-9)

HP_F64_CASES = ["hp_base", "hp_ring", "hp_cos_v2"]
FLAT_F64_CASES = ["flat_base", "flat_cos_v2"]


@pytest.fixture
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _to_f64(model):
    """nnx initializers bind their default float dtype at import time (before the
    x64 fixture flips the global flag), so freshly built models still carry
    float32 Params even inside the x64 scope. Recast explicitly before loading
    float64 torch weights, otherwise load_torch_state's `.at[...].set()` silently
    truncates the incoming float64 values back to float32 (with a scatter
    FutureWarning) and the test would trivially pass at float32 precision.
    """
    def cast(path, v):
        if isinstance(v, nnx.Param):
            val = v[...]
            if jnp.issubdtype(val.dtype, jnp.floating):
                return v.replace(val.astype(jnp.float64))
        return v
    return nnx.map(cast, model)


@pytest.mark.parametrize("case", HP_F64_CASES)
def test_hp_f64_gradient_parity(case, x64):
    npz, meta = load_case(case + "_f64")
    assert meta["dtype"] == "float64"
    model = _to_f64(build_hp_model(meta))
    load_torch_state(model, state_dict_of(npz), prefix_map=HP_PREFIX_MAP)
    model.eval()
    x = jnp.asarray(npz["input"]).transpose(0, 2, 1)          # (B,C,N) -> (B,N,C)
    assert x.dtype == jnp.float64

    y = model(x)
    assert y.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(y).transpose(0, 2, 1), npz["output"], **F64_TOL)

    gx = jax.grad(lambda x: model(x).sum())(x)
    np.testing.assert_allclose(np.asarray(gx).transpose(0, 2, 1), npz["input_grad"], **F64_TOL)

    gp = nnx.grad(lambda m: m(x).sum())(model)
    flat = {tuple(str(p) for p in path): v for path, v in nnx.to_flat_state(gp)}
    grads = grads_of(npz)
    assert len(grads) > 0
    for tkey, tgrad in grads.items():
        path = tuple(str(p) for p in torch_key_to_path(tkey, HP_PREFIX_MAP))
        expected = transform_array(tgrad, path[-1])
        np.testing.assert_allclose(np.asarray(flat[path][...]), expected,
                                   err_msg=tkey, **F64_TOL)


@pytest.mark.parametrize("case", FLAT_F64_CASES)
def test_flat_f64_gradient_parity(case, x64):
    npz, meta = load_case(case + "_f64")
    assert meta["dtype"] == "float64"
    model = _to_f64(build_flat_model(meta))
    load_torch_state(model, state_dict_of(npz), prefix_map=FLAT_PREFIX_MAP)
    model.eval()
    x = jnp.asarray(npz["input"]).transpose(0, 2, 3, 1)        # (B,C,H,W) -> (B,H,W,C)
    assert x.dtype == jnp.float64

    y = model(x)
    assert y.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(y).transpose(0, 3, 1, 2), npz["output"], **F64_TOL)

    gx = jax.grad(lambda x: model(x).sum())(x)
    np.testing.assert_allclose(np.asarray(gx).transpose(0, 3, 1, 2), npz["input_grad"], **F64_TOL)

    gp = nnx.grad(lambda m: m(x).sum())(model)
    flat_g = {tuple(str(p) for p in path): v for path, v in nnx.to_flat_state(gp)}
    grads = grads_of(npz)
    assert len(grads) > 0
    for tkey, tgrad in grads.items():
        path = tuple(str(p) for p in torch_key_to_path(tkey, FLAT_PREFIX_MAP))
        expected = transform_array(tgrad, path[-1])
        np.testing.assert_allclose(np.asarray(flat_g[path][...]), expected,
                                   err_msg=tkey, **F64_TOL)
