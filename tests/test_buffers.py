import jax.numpy as jnp
from flax import nnx

from heal_swin_nnx import HealSwinParams, SwinParams
from heal_swin_nnx.variables import Buffer


class Toy(nnx.Module):
    def __init__(self):
        self.w = nnx.Param(jnp.ones((3,)))
        self.idx = Buffer(jnp.arange(3))

    def __call__(self):
        return (self.w * self.idx).sum()


def test_buffer_excluded_from_params():
    m = Toy()
    params = nnx.state(m, nnx.Param)
    flat = dict(nnx.to_flat_state(params))
    assert ("w",) in flat
    assert not any("idx" in path for path in flat)


def test_buffer_not_differentiated():
    m = Toy()
    grads = nnx.grad(lambda m: m())(m)  # default wrt=nnx.Param
    flat = dict(nnx.to_flat_state(grads))
    assert ("w",) in flat
    assert not any("idx" in path for path in flat)


def test_params_construct_with_defaults():
    hp = HealSwinParams(nside=16, in_channels=3, out_channels=5)
    assert hp.patch_size == 4 and hp.window_size == 4 and hp.shift_size == 2
    assert hp.shift_strategy == "nest_grid_shift_exact" and hp.pos_embed == "rope_mixed"
    flat = SwinParams(img_size=(128, 128), in_channels=3, out_channels=5)
    assert flat.patch_size == (4, 4) and flat.window_size == (4, 4)
    assert flat.shift_size == (2, 2)
