"""Multi-device sharding helpers: logical axis names, rule tables, mesh and placement.

Cross-repo contract: GenSBI's ``sharding.py`` holds the same ``LOGICAL_AXES``,
``RULES`` and ``MODEL_AXIS`` values and the same ``axes``/``partitioned`` bodies
(byte-identical code; docstrings differ). ``make_mesh``, ``batch_axes``,
``batch_size_divisor``, ``shard_model``, ``shard_batch``, ``check_batch_divisible``
and ``sharding_of`` share semantics (including warning and error conditions) but
not text. Change both files together.

Models annotate parameters with *logical* axis names (``axes("embed", "mlp")``)
and stay mesh-free at construction (``eager_sharding=False``). A mesh and a rule
table are applied afterwards by ``shard_model``, which places every state leaf.
"""
import warnings

import jax
import numpy as np
from flax import nnx
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

LOGICAL_AXES: tuple[str, ...] = ("embed", "mlp", "heads", "vocab", "in", "out")
RULES: dict[str, tuple[tuple[str, str | None], ...]] = {
    "replicated": (("embed", None), ("mlp", None), ("heads", None), ("vocab", None), ("in", None), ("out", None)),
    "fsdp":       (("embed", "data"), ("mlp", None), ("heads", None), ("vocab", None), ("in", None), ("out", None)),
    "tensor":     (("embed", None), ("mlp", "model"), ("heads", "model"), ("vocab", "model"), ("in", None), ("out", None)),
}
MODEL_AXIS = "model"


def axes(*names: str | None) -> dict:
    """Variable metadata annotating logical axis names, without a mesh at construction."""
    return {"out_sharding": names, "eager_sharding": False}


def partitioned(init, names: tuple[str | None, ...]):
    """``nnx.with_partitioning`` for initializers, opted out of eager sharding."""
    return nnx.with_partitioning(init, names, eager_sharding=False)


def make_mesh(devices=None, model_axis: int = 1) -> jax.sharding.Mesh:
    """Mesh with axes ("data", "model"); ``data = len(devices) // model_axis``."""
    devices = list(jax.devices() if devices is None else devices)
    if model_axis < 1 or len(devices) % model_axis:
        raise ValueError(
            f"model_axis={model_axis} must be a positive divisor of the device count {len(devices)}.")
    return jax.sharding.Mesh(
        np.asarray(devices).reshape(len(devices) // model_axis, model_axis),
        ("data", MODEL_AXIS), axis_types=(AxisType.Auto, AxisType.Auto))


def batch_axes(mesh) -> tuple[str, ...]:
    """Mesh axes the batch is split over: every axis except the model axis."""
    return tuple(a for a in mesh.axis_names if a != MODEL_AXIS)


def batch_size_divisor(mesh) -> int:
    return int(np.prod([mesh.shape[a] for a in batch_axes(mesh)]))


def _spec_for(var, rule_map, mesh):
    """Return (PartitionSpec, fallback reason or None) for one variable."""
    value = var[...]
    names = var.get_metadata().get("out_sharding")
    if names is None:
        reason = "no annotation" if isinstance(var, nnx.Param) else None
        return P(), reason
    if len(names) != np.ndim(value):
        return P(), f"annotation rank {len(names)} != array rank {np.ndim(value)}"
    mapped = []
    for name, dim in zip(names, np.shape(value)):
        if name is None:
            mapped.append(None)
            continue
        if name not in rule_map:
            return P(), f"unknown logical axis '{name}'"
        axis = rule_map[name]
        if axis is not None and dim % mesh.shape[axis]:
            return P(), (f"dim {dim} of logical axis '{name}' is not divisible by "
                         f"mesh axis '{axis}' of size {mesh.shape[axis]}")
        mapped.append(axis)
    return P(*mapped), None


def shard_model(node, mesh, rule: str = "replicated") -> None:
    """Place every state leaf of ``node`` on ``mesh`` according to ``rule``, in place.

    Idempotent. Under "replicated" everything is replicated and nothing is
    recorded. Otherwise unknown names, indivisible dims and unannotated
    ``nnx.Param`` leaves fall back to replicated, with one warning listing them.
    """
    if rule not in RULES:
        raise ValueError(f"unknown sharding rule '{rule}'; choose from {tuple(RULES)}")
    if any(t == AxisType.Explicit for t in mesh.axis_types):
        warnings.warn("mesh has Explicit axis types; shard_model is designed for Auto meshes",
                      stacklevel=2)
    if rule != "replicated":
        warnings.warn(f"sharding rule '{rule}' is experimental and not covered by parity "
                      "or convergence guarantees", UserWarning, stacklevel=2)
    rule_map = dict(RULES[rule])
    fallbacks = []
    flat = []
    for path, var in nnx.to_flat_state(nnx.state(node)):
        if rule == "replicated":
            spec = P()
        else:
            spec, reason = _spec_for(var, rule_map, mesh)
            if reason is not None:
                fallbacks.append(f"{'/'.join(map(str, path))} ({reason})")
        new = var.copy()
        new.set_value(jax.device_put(var[...], NamedSharding(mesh, spec)))
        flat.append((path, new))
    nnx.update(node, nnx.from_flat_state(flat))
    if fallbacks:
        warnings.warn("replicated fallback for: " + "; ".join(fallbacks), stacklevel=2)


def shard_batch(batch, mesh):
    """Shard the leading axis of every array leaf over the batch axes; others untouched."""
    sharding = NamedSharding(mesh, P(batch_axes(mesh)))
    return jax.tree.map(
        lambda x: jax.device_put(x, sharding) if isinstance(x, (np.ndarray, jax.Array)) else x,
        batch)


def check_batch_divisible(batch, mesh, what: str) -> None:
    sizes = [x.shape[0] for x in jax.tree.leaves(batch)
             if isinstance(x, (np.ndarray, jax.Array)) and x.ndim > 0]
    if not sizes:
        return
    if len(set(sizes)) > 1:
        raise ValueError(f"{what} batch leaves have differing leading sizes: {sizes}")
    d = batch_size_divisor(mesh)
    if sizes[0] % d:
        raise ValueError(
            f"{what} batch size {sizes[0]} is not divisible by the data-parallel device count "
            f"{d} (mesh axes {batch_axes(mesh)}). Use a batch size that is a multiple of {d}.")


def sharding_of(node) -> dict[str, jax.sharding.Sharding]:
    """Map "a/b/c" state paths to the sharding of each leaf."""
    return {"/".join(map(str, path)): var[...].sharding
            for path, var in nnx.to_flat_state(nnx.state(node))}
