import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

from heal_swin_nnx import (HealConv, HealConvParams, HealSwin, HealSwinEncoder,
                           HealSwinParams, SwinParams, SwinUnet)
from heal_swin_nnx import sharding as sh
from heal_swin_nnx.variables import Buffer

pytestmark = pytest.mark.sharding

requires_8_devices = pytest.mark.skipif(
    jax.device_count() < 8,
    reason="needs XLA_FLAGS=--xla_force_host_platform_device_count=8")


def test_eight_cpu_devices_present():
    assert jax.device_count() == 8
    assert all(d.platform == "cpu" for d in jax.devices())


class Toy(nnx.Module):
    def __init__(self, rngs):
        self.l1 = nnx.Linear(16, 32, rngs=rngs, kernel_metadata=sh.axes("embed", "mlp"), bias_metadata=sh.axes("mlp"))
        self.l2 = nnx.Linear(32, 16, rngs=rngs, kernel_metadata=sh.axes("mlp", "embed"), bias_metadata=sh.axes("embed"))
        self.odd = nnx.Param(jnp.ones((6, 3)), **sh.axes("embed", None))     # 6 % 8 != 0 -> fsdp fallback
        self.bad = nnx.Param(jnp.ones((4,)), **sh.axes("nonsense"))          # unknown name -> fallback
        self.buf = Buffer(jnp.arange(5))
        self.drop = nnx.Dropout(0.1, rngs=rngs)


def test_rule_tables_cover_every_logical_name():
    for rule, table in sh.RULES.items():
        assert tuple(n for n, _ in table) == sh.LOGICAL_AXES, rule
    assert sh.RULES["replicated"] == tuple((n, None) for n in sh.LOGICAL_AXES)


def test_axes_and_partitioned_set_the_opt_out():
    assert sh.axes("embed", None) == {"out_sharding": ("embed", None), "eager_sharding": False}
    m = Toy(nnx.Rngs(0))                      # must construct with NO mesh anywhere
    assert isinstance(m.l1.kernel[...].sharding, jax.sharding.SingleDeviceSharding)
    assert m.l1.kernel.get_metadata()["eager_sharding"] is False


@requires_8_devices
def test_make_mesh_shape_and_axis_types():
    mesh = sh.make_mesh()
    assert mesh.axis_names == ("data", "model") and dict(mesh.shape) == {"data": 8, "model": 1}
    assert all(t == AxisType.Auto for t in mesh.axis_types)
    mesh2 = sh.make_mesh(model_axis=2)
    assert dict(mesh2.shape) == {"data": 4, "model": 2}
    assert sh.batch_axes(mesh2) == ("data",) and sh.batch_size_divisor(mesh2) == 4
    with pytest.raises(ValueError):
        sh.make_mesh(model_axis=3)


@requires_8_devices
def test_shard_model_replicated_places_everything():
    m = Toy(nnx.Rngs(0)); mesh = sh.make_mesh()
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # replicated must not warn
        sh.shard_model(m, mesh, "replicated")
    for path, s in sh.sharding_of(m).items():
        assert isinstance(s, NamedSharding) and s.mesh == mesh, path
        assert s.is_equivalent_to(NamedSharding(mesh, P()), 1), path   # includes buf and dropout RngKey/RngCount


@requires_8_devices
def test_shard_model_fsdp_shards_embed_and_falls_back():
    m = Toy(nnx.Rngs(0)); mesh = sh.make_mesh()
    with pytest.warns(UserWarning) as rec:
        sh.shard_model(m, mesh, "fsdp")
    msgs = " ".join(str(w.message) for w in rec)
    assert "odd" in msgs and "bad" in msgs
    s = sh.sharding_of(m)
    assert s["l1/kernel"].is_equivalent_to(NamedSharding(mesh, P("data", None)), 2)
    assert s["l2/kernel"].is_equivalent_to(NamedSharding(mesh, P(None, "data")), 2)
    assert s["l2/bias"].is_equivalent_to(NamedSharding(mesh, P("data")), 1)
    assert s["odd"].is_equivalent_to(NamedSharding(mesh, P()), 2)
    assert s["bad"].is_equivalent_to(NamedSharding(mesh, P()), 1)
    with pytest.warns(UserWarning):
        sh.shard_model(m, mesh, "fsdp")       # idempotent: second call is a no-op (no error)


@requires_8_devices
def test_shard_model_fsdp_warns_once_for_unannotated_param_only():
    class Plain(nnx.Module):
        def __init__(self, rngs):
            self.l = nnx.Linear(8, 8, rngs=rngs)      # no annotation
            self.buf = Buffer(jnp.arange(3))
    m = Plain(nnx.Rngs(0))
    with pytest.warns(UserWarning) as rec:
        sh.shard_model(m, sh.make_mesh(), "fsdp")
    fb = [str(w.message) for w in rec if "fallback" in str(w.message)]
    assert len(fb) == 1 and "l/kernel" in fb[0] and "buf" not in fb[0]


@requires_8_devices
def test_shard_model_fsdp_places_optimizer_state_by_param_metadata():
    m = Toy(nnx.Rngs(0)); mesh = sh.make_mesh()
    opt = nnx.Optimizer(m, optax.adamw(1e-3), wrt=nnx.Param)
    with pytest.warns(UserWarning):
        sh.shard_model(opt, mesh, "fsdp")
    s = sh.sharding_of(opt)
    mu = [k for k in s if k.endswith("mu/l1/kernel")]
    assert mu and s[mu[0]].is_equivalent_to(NamedSharding(mesh, P("data", None)), 2)


@requires_8_devices
def test_shard_model_tensor_shards_mlp_over_model_axis():
    m = Toy(nnx.Rngs(0)); mesh = sh.make_mesh(model_axis=2)
    with pytest.warns(UserWarning):
        sh.shard_model(m, mesh, "tensor")
    assert sh.sharding_of(m)["l1/kernel"].is_equivalent_to(NamedSharding(mesh, P(None, "model")), 2)


def test_shard_model_rejects_unknown_rule():
    with pytest.raises(ValueError):
        sh.shard_model(Toy(nnx.Rngs(0)), sh.make_mesh(), "nope")


@requires_8_devices
def test_shard_batch_numpy_jax_tuple_and_dict_leaves():
    mesh = sh.make_mesh()
    x = np.zeros((16, 5, 1), np.float32); y = jnp.zeros((16,), jnp.int32)
    out = sh.shard_batch({"image": x, "label": y, "meta": 3}, mesh)
    assert out["image"].sharding.is_equivalent_to(NamedSharding(mesh, P("data")), 3)
    assert out["label"].sharding.is_equivalent_to(NamedSharding(mesh, P("data")), 1)
    assert out["meta"] == 3
    a, b = sh.shard_batch((x, y), mesh)
    assert a.shape == (16, 5, 1) and b.shape == (16,)


@requires_8_devices
def test_check_batch_divisible_message():
    mesh = sh.make_mesh()
    sh.check_batch_divisible((np.zeros((16, 4)),), mesh, "train")   # ok
    with pytest.raises(ValueError, match=r"train batch size 12 is not divisible by the data-parallel device count 8"):
        sh.check_batch_divisible((np.zeros((12, 4)),), mesh, "train")
    with pytest.raises(ValueError, match="differing leading sizes"):
        sh.check_batch_divisible((np.zeros((16, 4)), np.zeros((8,))), mesh, "train")


def _tiny_params(**over):
    kw = dict(nside=16, in_channels=3, out_channels=5, base_pixels=tuple(range(8)),
              embed_dim=16, depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0)
    kw.update(over)
    return HealSwinParams(**kw)


def assert_every_param_annotated(model):
    for path, var in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        meta = var.get_metadata()
        name = "/".join(map(str, path))
        assert "out_sharding" in meta, f"{name} has no annotation"
        assert meta["eager_sharding"] is False, name
        names = meta["out_sharding"]
        assert len(names) == var[...].ndim, f"{name}: rank {var[...].ndim} vs {names}"
        assert set(n for n in names if n is not None) <= set(sh.LOGICAL_AXES), name
        used = [n for n in names if n is not None]
        assert len(used) == len(set(used)), f"{name}: repeated logical name in {names}"
    for path, var in nnx.to_flat_state(nnx.state(model, Buffer)):
        assert "out_sharding" not in var.get_metadata(), path


@pytest.mark.parametrize("pos_embed", ["rope_mixed", "rel_bias", "rope_axial"])
def test_every_healswin_param_annotated(pos_embed):
    m = HealSwin(_tiny_params(pos_embed=pos_embed), rngs=nnx.Rngs(0))
    assert_every_param_annotated(m)
    first_kernel = next(v for p, v in nnx.to_flat_state(nnx.state(m, nnx.Param))
                        if str(p[-1]) == "kernel")
    assert isinstance(first_kernel[...].sharding, jax.sharding.SingleDeviceSharding)


def test_healswin_values_unchanged_by_annotations():
    # snapshot of the first kernel's statistics taken on the unannotated code
    m = HealSwin(_tiny_params(), rngs=nnx.Rngs(0))
    first_kernel = next(v for p, v in nnx.to_flat_state(nnx.state(m, nnx.Param))
                        if str(p[-1]) == "kernel")
    k = np.asarray(first_kernel[...], np.float32)
    np.testing.assert_allclose(float(k.mean()), 0.000746572797652334, rtol=1e-6)
    np.testing.assert_allclose(float(k.std()), 0.017588790506124496, rtol=1e-6)


@pytest.mark.parametrize("pos_embed", ["none", "rel_bias", "rope_axial", "rope_mixed"])
def test_every_swin_param_annotated(pos_embed):
    m = SwinUnet(SwinParams(img_size=(32, 64), in_channels=2, out_channels=3,
                            embed_dim=16, depths=(2, 2), num_heads=(2, 4),
                            drop_path_rate=0.0, pos_embed=pos_embed),
                 rngs=nnx.Rngs(0))
    assert_every_param_annotated(m)


@pytest.mark.parametrize("conv_bias", [True, False])
def test_every_healconv_param_annotated(conv_bias):
    m = HealConv(HealConvParams(nside=16, in_channels=3, out_channels=5,
                                base_pixels=tuple(range(8)), embed_dim=16,
                                depths=(2, 2), drop_path_rate=0.0,
                                conv_bias=conv_bias),
                 rngs=nnx.Rngs(0))
    assert_every_param_annotated(m)


def _batch(B, p, key):
    return jax.random.normal(key, (B, p.npix, p.in_channels), jnp.float32)


def _params_allclose(a, b, **tol):
    fa = nnx.to_flat_state(nnx.state(a, nnx.Param)); fb = nnx.to_flat_state(nnx.state(b, nnx.Param))
    for (pa, va), (pb, vb) in zip(fa, fb):
        assert pa == pb
        np.testing.assert_allclose(np.asarray(vb[...]), np.asarray(va[...]), err_msg=str(pa), **tol)


_experimental = pytest.mark.filterwarnings("ignore:sharding rule")

# Sharding the batch changes XLA's CPU reduction/tiling order: fp32 forwards differ by ~1e-4
# (measured max abs 1.6e-4 fwd, 3.3e-4 with dropout, 1.2e-4 tensor; output scale ~3.5).
# A real placement bug gives O(1) differences, so 2e-3 still discriminates.
FWD_TOL = dict(rtol=2e-3, atol=2e-3)


@requires_8_devices
@_experimental
@pytest.mark.parametrize("rule", ["replicated", "fsdp"])
def test_forward_matches_unsharded(rule):
    p = _tiny_params(dtype="float32")                      # fp32 compute -> tight tolerance
    ref = HealSwin(p, rngs=nnx.Rngs(0)); ref.eval()
    m = HealSwin(p, rngs=nnx.Rngs(0)); m.eval()
    mesh = sh.make_mesh(); sh.shard_model(m, mesh, rule)
    x = _batch(16, p, jax.random.key(1))
    y_ref = nnx.jit(lambda m, x: m(x))(ref, x)
    y = nnx.jit(lambda m, x: m(x))(m, sh.shard_batch(x, mesh))
    assert y.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), y.ndim)
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_ref), **FWD_TOL)


@requires_8_devices
@_experimental
def test_forward_matches_unsharded_bf16():
    # bf16 compute is noisy on its own (unsharded bf16 vs fp32: rms 0.081, max 3.2 on output rms 0.97),
    # so a sharded-vs-unsharded allclose is meaningless. Instead the sharded model must be about as
    # close to the fp32 truth as the unsharded bf16 model is (measured ratio 1.11).
    p = _tiny_params()                                     # default dtype="bfloat16"
    fwd = nnx.jit(lambda m, x: m(x))
    truth = HealSwin(_tiny_params(dtype="float32"), rngs=nnx.Rngs(0)); truth.eval()
    ref = HealSwin(p, rngs=nnx.Rngs(0)); ref.eval()
    m = HealSwin(p, rngs=nnx.Rngs(0)); m.eval()
    mesh = sh.make_mesh(); sh.shard_model(m, mesh, "fsdp")
    x = _batch(16, p, jax.random.key(1))
    y_true = np.asarray(fwd(truth, x))
    rms = lambda d: float(np.sqrt(np.mean(np.square(d))))
    err_ref = rms(np.asarray(fwd(ref, x), np.float32) - y_true)
    err = rms(np.asarray(fwd(m, sh.shard_batch(x, mesh)), np.float32) - y_true)
    assert err_ref < 0.2 * rms(y_true)
    assert err <= 1.5 * err_ref, (err, err_ref)


@requires_8_devices
def test_train_mode_forward_matches_unsharded():
    # active Dropout + DropPath: the per-sample masks must agree for the same RNG state
    p = _tiny_params(dtype="float32", drop_rate=0.2, drop_path_rate=0.2)
    ref = HealSwin(p, rngs=nnx.Rngs(0)); ref.train()
    m = HealSwin(p, rngs=nnx.Rngs(0)); m.train()
    mesh = sh.make_mesh(); sh.shard_model(m, mesh, "replicated")
    x = _batch(16, p, jax.random.key(1))
    y_ref = nnx.jit(lambda m, x: m(x))(ref, x)
    y = nnx.jit(lambda m, x: m(x))(m, sh.shard_batch(x, mesh))
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_ref), **FWD_TOL)


@requires_8_devices
@_experimental
def test_fsdp_places_qkv_and_mlp_kernels_over_data_and_buffers_replicated():
    p = _tiny_params(); m = HealSwin(p, rngs=nnx.Rngs(0)); mesh = sh.make_mesh()
    sh.shard_model(m, mesh, "fsdp")
    s = sh.sharding_of(m)
    qkv = [k for k in s if k.endswith("qkv/kernel")]; fc1 = [k for k in s if k.endswith("fc1/kernel")]
    assert qkv and fc1
    for k in qkv + fc1:
        assert s[k].is_equivalent_to(NamedSharding(mesh, P("data", None)), 2), k
    bufs = {"/".join(map(str, pth)): v[...].ndim for pth, v in nnx.to_flat_state(nnx.state(m, Buffer))}
    assert bufs
    for k, nd in bufs.items():
        assert s[k].is_equivalent_to(NamedSharding(mesh, P()), nd), k


@requires_8_devices
@_experimental
@pytest.mark.parametrize("rule", ["replicated", "fsdp"])
def test_one_gradient_step_matches_unsharded(rule):
    p = _tiny_params(dtype="float32")
    def make():
        m = HealSwin(p, rngs=nnx.Rngs(0)); m.train()
        return m, nnx.Optimizer(m, optax.adamw(1e-3), wrt=nnx.Param)
    ref, opt_ref = make(); m, opt = make()
    mesh = sh.make_mesh(); sh.shard_model(m, mesh, rule); sh.shard_model(opt, mesh, rule)
    x = _batch(16, p, jax.random.key(1))
    @nnx.jit
    def step(model, opt, x):
        loss, g = nnx.value_and_grad(lambda mm: (mm(x) ** 2).mean())(model)
        opt.update(model, g); return loss
    l_ref = step(ref, opt_ref, x); l = step(m, opt, sh.shard_batch(x, mesh))
    np.testing.assert_allclose(float(l), float(l_ref), rtol=1e-5)
    _params_allclose(ref, m, rtol=1e-4, atol=1e-6)
    if rule == "fsdp":   # optimizer moments keep the data sharding after the step
        mu = [v for pth, v in nnx.to_flat_state(nnx.state(opt)) if "mu" in map(str, pth) and v[...].ndim == 2]
        assert mu and any(v[...].sharding.is_equivalent_to(NamedSharding(mesh, P("data", None)), 2) for v in mu)


@requires_8_devices
@_experimental
def test_encoder_standalone_fsdp():
    p = _tiny_params(); enc = HealSwinEncoder(p, rngs=nnx.Rngs(0)); enc.eval()
    mesh = sh.make_mesh(); sh.shard_model(enc, mesh, "fsdp")
    tokens, skips = nnx.jit(lambda m, x: m(x))(enc, sh.shard_batch(_batch(8, p, jax.random.key(0)), mesh))
    assert tokens.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), tokens.ndim)


@requires_8_devices
@_experimental
def test_tensor_rule_smoke_4x2():
    p = _tiny_params(dtype="float32")
    ref = HealSwin(p, rngs=nnx.Rngs(0)); ref.eval(); m = HealSwin(p, rngs=nnx.Rngs(0)); m.eval()
    mesh = sh.make_mesh(model_axis=2); sh.shard_model(m, mesh, "tensor")
    s = sh.sharding_of(m); k = next(k for k in s if k.endswith("fc1/kernel"))
    assert s[k].is_equivalent_to(NamedSharding(mesh, P(None, "model")), 2)
    x = _batch(8, p, jax.random.key(1))
    y = nnx.jit(lambda m, x: m(x))(m, sh.shard_batch(x, mesh))
    np.testing.assert_allclose(np.asarray(y), np.asarray(nnx.jit(lambda m, x: m(x))(ref, x)), **FWD_TOL)
