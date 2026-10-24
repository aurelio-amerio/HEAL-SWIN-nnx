import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

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
