"""Load reference torch state_dicts (saved as npz arrays) into nnx models.

The nnx module tree mirrors the torch tree, so mapping is mechanical:
prefix rewrite (encoder/decoder seam) + leaf rename + layout transform.
"""
import jax.numpy as jnp
import numpy as np
from flax import nnx

HP_PREFIX_MAP = {"patch_embed": ("encoder", "patch_embed"), "layers": ("encoder", "layers"),
                 "norm": ("encoder", "norm"), "absolute_pos_embed": ("encoder", "absolute_pos_embed"),
                 "decoder": ("decoder",)}
FLAT_PREFIX_MAP = {"patch_embed": ("encoder", "patch_embed"), "layers": ("encoder", "layers"),
                   "norm": ("encoder", "norm"), "absolute_pos_embed": ("encoder", "absolute_pos_embed"),
                   "layers_up": ("decoder", "layers_up"), "concat_back_dim": ("decoder", "concat_back_dim"),
                   "norm_up": ("decoder", "norm_up"), "up": ("decoder", "up"), "output": ("decoder", "output")}

SKIP_SUFFIXES = ("attn_mask", "relative_position_index")


def torch_key_to_path(key, prefix_map=None):
    parts = key.split(".")
    if prefix_map is not None:
        parts = list(prefix_map[parts[0]]) + parts[1:]
    if parts[-1] == "weight":
        parent = parts[-2] if len(parts) >= 2 else ""
        parts[-1] = "scale" if "norm" in parent else "kernel"
    return tuple(int(p) if p.isdigit() else p for p in parts)


def transform_array(arr, renamed_leaf):
    if renamed_leaf != "kernel":
        return arr
    if arr.ndim == 2:            # nn.Linear (out, in) -> (in, out)
        return arr.T
    if arr.ndim == 3:            # nn.Conv1d (out, in, k) -> (k, in, out)
        return arr.transpose(2, 1, 0)
    if arr.ndim == 4:            # nn.Conv2d (out, in, kh, kw) -> (kh, kw, in, out)
        return arr.transpose(2, 3, 1, 0)
    return arr


def load_torch_state(model, sd, prefix_map=None):
    state = nnx.state(model)
    flat = dict(nnx.to_flat_state(state))
    param_paths = {path for path, v in flat.items() if isinstance(v, nnx.Param)}
    assigned = set()
    for key, arr in sd.items():
        if key.endswith(SKIP_SUFFIXES):
            continue
        path = torch_key_to_path(key, prefix_map)
        if path not in flat:
            raise ValueError("torch key %r maps to %r which is not in the nnx state" % (key, path))
        value = transform_array(np.asarray(arr), path[-1])
        if tuple(flat[path][...].shape) != tuple(value.shape):
            raise ValueError("shape mismatch for %r: nnx %s vs torch %s"
                             % (key, flat[path][...].shape, value.shape))
        # in-place set casts to the target param's dtype — callers doing non-float32 work must cast the model's state first (see tests/test_parity_f64.py)
        flat[path][...] = jnp.asarray(value)
        assigned.add(path)
    missing = param_paths - assigned
    if missing:
        raise ValueError("nnx Params not assigned by transfer: %s" % sorted(missing))
    nnx.update(model, nnx.from_flat_state(flat))
