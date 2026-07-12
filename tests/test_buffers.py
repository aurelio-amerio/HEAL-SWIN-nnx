import jax.numpy as jnp
from flax import nnx

from heal_swin_nnx.variables import Buffer
from heal_swin_nnx.config import DataSpec, SwinHPTransformerConfig, SwinTransformerConfig


class Toy(nnx.Module):
    def __init__(self):
        self.w = nnx.Param(jnp.ones((3,)))
        self.idx = Buffer(jnp.arange(3))

    def __call__(self):
        return (self.w * self.idx).sum()


def test_buffer_excluded_from_params():
    m = Toy()
    params = nnx.state(m, nnx.Param)
    flat = dict(params.flat_state())
    assert ("w",) in flat
    assert not any("idx" in path for path in flat)


def test_buffer_not_differentiated():
    m = Toy()
    grads = nnx.grad(lambda m: m())(m)  # default wrt=nnx.Param
    flat = dict(grads.flat_state())
    assert ("w",) in flat
    assert not any("idx" in path for path in flat)


def test_configs_construct_with_reference_defaults():
    hp = SwinHPTransformerConfig()
    assert hp.patch_size == 4 and hp.window_size == 4 and hp.shift_size == 2
    assert hp.shift_strategy == "nest_roll" and hp.rel_pos_bias is None
    assert hp.depths == [2, 2, 2, 2] and hp.num_heads == [3, 6, 12, 24]
    flat = SwinTransformerConfig()
    assert flat.patch_size == (4, 4) and flat.window_size == (4, 4)
    assert flat.shift_size == (2, 2)  # -1 sentinel resolved to window//2
    flat2 = SwinTransformerConfig(window_size=8, shift_size=3)
    assert flat2.window_size == (8, 8) and flat2.shift_size == (3, 3)
    ds = DataSpec(dim_in=2048, f_in=3, f_out=5, base_pix=8, class_names=["a"] * 5)
    assert ds.dim_in == 2048
