import dataclasses
import json

import jax
import jax.numpy as jnp
import pytest

from heal_swin_nnx.models.healswin import HealSwinParams


def test_defaults_full_sphere():
    p = HealSwinParams(nside=16, in_channels=3, out_channels=5)
    assert p.base_pixels == tuple(range(12))
    assert p.npix == 12 * 16 ** 2
    assert p.shift_size == 2
    assert p.pos_embed == "rope_mixed"
    assert p.shift_strategy == "nest_grid_shift_exact"
    assert p.depths == (2, 2, 2, 2) and p.num_heads == (3, 6, 12, 24)


def test_partial_coverage_and_tuple_coercion():
    p = HealSwinParams(nside=16, in_channels=1, out_channels=1,
                       base_pixels=[8, 9, 10, 11], depths=[2, 2], num_heads=[2, 4],
                       embed_dim=16)
    assert p.base_pixels == (8, 9, 10, 11)
    assert p.npix == 4 * 16 ** 2
    assert p.depths == (2, 2)


@pytest.mark.parametrize("bad", [[0, 0, 1], [3, 2], [-1, 0], [11, 12]])
def test_invalid_base_pixels_rejected(bad):
    with pytest.raises(ValueError):
        HealSwinParams(nside=16, in_channels=1, out_channels=1, base_pixels=bad)


def test_params_is_pure_serializable_data():
    p = HealSwinParams(nside=16, in_channels=3, out_channels=5)
    json.dumps(dataclasses.asdict(p))  # must not raise


def test_nside_must_be_power_of_two():
    with pytest.raises(ValueError):
        HealSwinParams(nside=12, in_channels=1, out_channels=1)


def test_not_enough_resolution_for_stages():
    # nside=4, patch_size=4 -> 4 patches/face; 3 stages need divisibility by 4^2
    with pytest.raises(ValueError):
        HealSwinParams(nside=4, in_channels=1, out_channels=1,
                       depths=(2, 2, 2), num_heads=(2, 2, 2), embed_dim=8)


def test_depths_num_heads_length_mismatch_rejected():
    with pytest.raises(ValueError):
        HealSwinParams(nside=16, in_channels=1, out_channels=1,
                       depths=(2, 2), num_heads=(2, 4, 8))


def test_rope_requires_head_dim_divisible_by_4():
    # embed_dim=12, heads (2, 4): head dims 6, 6 -> invalid for RoPE...
    with pytest.raises(ValueError):
        HealSwinParams(nside=16, in_channels=1, out_channels=1,
                       embed_dim=12, depths=(2, 2), num_heads=(2, 4))
    # ...but fine without positional encoding
    HealSwinParams(nside=16, in_channels=1, out_channels=1,
                   embed_dim=12, depths=(2, 2), num_heads=(2, 4), pos_embed="none")
    # and fine with head dims divisible by 4
    HealSwinParams(nside=16, in_channels=1, out_channels=1,
                   embed_dim=16, depths=(2, 2), num_heads=(2, 4))


def test_embed_dim_must_divide_by_heads():
    with pytest.raises(ValueError):
        HealSwinParams(nside=16, in_channels=1, out_channels=1,
                       embed_dim=16, depths=(2, 2), num_heads=(3, 4))


def test_unknown_enum_values_rejected():
    with pytest.raises(ValueError):
        HealSwinParams(nside=16, in_channels=1, out_channels=1, pos_embed="learned")
    with pytest.raises(ValueError):
        HealSwinParams(nside=16, in_channels=1, out_channels=1, shift_strategy="roll")


def test_nest_grid_shift_requires_window_sized_bottleneck():
    # nside=16, patch_size=4, 4 stages -> per-face nside 8 -> 4 -> 2 -> 1 at the
    # bottleneck. nest_grid_shift's index math needs the deepest stage to hold a
    # full window (nside**2 >= window_size), so nside=1 must be rejected up front
    # with a clear message instead of a ZeroDivisionError during model construction.
    with pytest.raises(ValueError, match="nest_grid_shift"):
        HealSwinParams(nside=16, in_channels=1, out_channels=1,
                       depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                       shift_strategy="nest_grid_shift")
    # The same unit-bottleneck geometry is fine for strategies that support it.
    for strat in ("nest_roll", "nest_grid_shift_exact", "ring_shift"):
        HealSwinParams(nside=16, in_channels=1, out_channels=1,
                       depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                       shift_strategy=strat)
    # And a bottleneck that does hold a window (nside=2 here) is accepted.
    HealSwinParams(nside=16, in_channels=1, out_channels=1,
                   depths=(2, 2, 2), num_heads=(3, 6, 12),
                   shift_strategy="nest_grid_shift")


def test_patch_size_must_be_power_of_four():
    # powers of 4 keep the patched grid a HEALPix grid (integer nside);
    # patch_size=1 means no regrouping: one pixel = one token
    for ps in (1, 4, 16):
        HealSwinParams(nside=16, in_channels=1, out_channels=1, patch_size=ps,
                       depths=(2, 2), num_heads=(2, 4), embed_dim=16)
    # multiples of 4 that are not powers of 4 (e.g. 8) give a non-integer nside
    for ps in (0, -4, 3, 8, 12):
        with pytest.raises(ValueError, match="patch_size"):
            HealSwinParams(nside=16, in_channels=1, out_channels=1, patch_size=ps,
                           depths=(2, 2), num_heads=(2, 4), embed_dim=16)


def test_window_size_must_be_power_of_four():
    with pytest.raises(ValueError):
        HealSwinParams(nside=16, in_channels=1, out_channels=1, window_size=8)
    HealSwinParams(nside=16, in_channels=1, out_channels=1, window_size=4)
    HealSwinParams(nside=16, in_channels=1, out_channels=1, window_size=16)


from heal_swin_nnx.models.swin import SwinParams


def test_swin_params_defaults_and_coercion():
    # default depths (2, 2, 2, 2) -> img must divide patch*window*2^(L-1) = 128
    p = SwinParams(img_size=(128, 256), in_channels=3, out_channels=5)
    assert p.patch_size == (4, 4) and p.window_size == (4, 4)
    assert p.shift_size == (2, 2)
    assert p.patches_resolution == (32, 64)
    assert p.pos_embed == "rel_bias"
    p2 = SwinParams(img_size=64, in_channels=1, out_channels=1, window_size=8,
                    depths=(2, 2), num_heads=(2, 4), embed_dim=16)
    assert p2.img_size == (64, 64) and p2.window_size == (8, 8)
    assert p2.shift_size == (4, 4)


def test_swin_params_serializable():
    import dataclasses, json
    json.dumps(dataclasses.asdict(SwinParams(img_size=(128, 128), in_channels=3,
                                             out_channels=5)))


def test_swin_params_rejects_indivisible_geometry():
    import pytest
    # H=60 not divisible by patch*window*2^(L-1) = 4*4*8
    with pytest.raises(ValueError):
        SwinParams(img_size=(60, 64), in_channels=1, out_channels=1)


def test_swin_params_rope_head_dim_check():
    import pytest
    with pytest.raises(ValueError):
        SwinParams(img_size=(32, 32), in_channels=1, out_channels=1, embed_dim=12,
                   depths=(2, 2), num_heads=(2, 4), pos_embed="rope_mixed")
    SwinParams(img_size=(32, 32), in_channels=1, out_channels=1, embed_dim=16,
               depths=(2, 2), num_heads=(2, 4), pos_embed="rope_mixed")


# --- HealConvParams ---------------------------------------------------------
from heal_swin_nnx.models.healconv import HealConvParams


def test_healconv_defaults_and_derived_window():
    p = HealConvParams(nside=32, in_channels=3, out_channels=5)
    assert p.base_pixels == tuple(range(12))
    assert p.kernel_size == 4 and p.window_size == 16 and p.shift_size == 8
    assert p.shift_strategy == "nest_grid_shift_exact"
    assert p.depths == (2, 2, 2, 2)
    assert p.npix == 12 * 32 ** 2


def test_healconv_params_serializable():
    json.dumps(dataclasses.asdict(HealConvParams(nside=32, in_channels=3, out_channels=5)))


@pytest.mark.parametrize("bad_k", [0, 1, 3, 6, -4])
def test_healconv_kernel_size_must_be_power_of_two(bad_k):
    with pytest.raises(ValueError, match="kernel_size"):
        HealConvParams(nside=32, in_channels=1, out_channels=1, kernel_size=bad_k)


def test_healconv_valid_kernel_sizes_accepted():
    for k in (2, 4, 8):
        p = HealConvParams(nside=64, in_channels=1, out_channels=1, kernel_size=k,
                           depths=(2, 2), embed_dim=16)
        assert p.window_size == k ** 2


def test_healconv_bottleneck_must_hold_whole_windows():
    # nside=16, patch_size=4, 4 stages -> bottleneck = 12*256/4/64 = 12 pixels,
    # not divisible by window_size=16: the clamped window would not be a power
    # of four, so params must reject this up front.
    with pytest.raises(ValueError, match="divisible"):
        HealConvParams(nside=16, in_channels=1, out_channels=1)
    # nside=32 bottleneck = 48, divisible by 16 -> fine
    HealConvParams(nside=32, in_channels=1, out_channels=1)
    # kernel_size=2 (window 4) at nside=16 -> bottleneck 12 divisible by 4 -> fine
    HealConvParams(nside=16, in_channels=1, out_channels=1, kernel_size=2)


def test_healconv_inherited_geometry_rules():
    with pytest.raises(ValueError):  # nside not a power of two
        HealConvParams(nside=12, in_channels=1, out_channels=1)
    with pytest.raises(ValueError):  # bad base_pixels ordering
        HealConvParams(nside=32, in_channels=1, out_channels=1, base_pixels=[3, 2])
    with pytest.raises(ValueError):  # unknown shift strategy
        HealConvParams(nside=32, in_channels=1, out_channels=1, shift_strategy="roll")
    with pytest.raises(ValueError):  # patch_size not a power of 4
        HealConvParams(nside=32, in_channels=1, out_channels=1, patch_size=3)
    # patch_size=1 (no regrouping) is a valid power of 4
    HealConvParams(nside=32, in_channels=1, out_channels=1, patch_size=1)


def test_healconv_nest_grid_shift_requires_window_sized_bottleneck():
    # nside=32, patch_size=4, 4 stages -> per-face bottleneck nside^2 = 4 < 16
    with pytest.raises(ValueError, match="nest_grid_shift"):
        HealConvParams(nside=32, in_channels=1, out_channels=1,
                       shift_strategy="nest_grid_shift")
    # 3 stages -> per-face bottleneck 16 >= 16 -> fine
    HealConvParams(nside=32, in_channels=1, out_channels=1, depths=(2, 2, 2),
                   shift_strategy="nest_grid_shift")


# --- param_dtype ------------------------------------------------------------

PARAMS_FACTORIES = [
    lambda **over: HealSwinParams(nside=16, in_channels=1, out_channels=1, **over),
    lambda **over: HealConvParams(nside=16, in_channels=1, out_channels=1,
                                  depths=(2, 2), **over),
    lambda **over: SwinParams(img_size=(32, 64), in_channels=1, out_channels=1,
                              embed_dim=16, depths=(2, 2), num_heads=(2, 4), **over),
]


@pytest.mark.parametrize("mk", PARAMS_FACTORIES)
def test_param_dtype_defaults_and_canonicalization(mk):
    assert mk().param_dtype == "float32"
    for spec in ("bfloat16", jnp.bfloat16, jnp.dtype(jnp.bfloat16)):
        p = mk(param_dtype=spec)
        assert p.param_dtype == "bfloat16"
        json.dumps(dataclasses.asdict(p))  # must stay serializable


@pytest.mark.parametrize("mk", PARAMS_FACTORIES)
@pytest.mark.parametrize("bad", ["int32", "bool", "not_a_dtype", 7, None])
def test_param_dtype_rejects_non_floats(mk, bad):
    with pytest.raises(ValueError):
        mk(param_dtype=bad)


@pytest.mark.parametrize("mk", PARAMS_FACTORIES)
def test_param_dtype_rejects_float64_without_x64(mk):
    # test env runs with jax_enable_x64 disabled; float64 params would silently
    # be created as float32 arrays, so this must raise rather than pass through.
    assert not jax.config.jax_enable_x64
    with pytest.raises(ValueError):
        mk(param_dtype="float64")


# --- compute dtype ----------------------------------------------------------


@pytest.mark.parametrize("mk", PARAMS_FACTORIES)
def test_dtype_canonicalization(mk):
    for spec in ("bfloat16", jnp.bfloat16, jnp.dtype(jnp.bfloat16)):
        p = mk(dtype=spec)
        assert p.dtype == "bfloat16"
        json.dumps(dataclasses.asdict(p))  # must stay serializable


@pytest.mark.parametrize("mk", PARAMS_FACTORIES)
@pytest.mark.parametrize("bad", ["int32", "bool", "not_a_dtype", 7, None])
def test_dtype_rejects_non_floats(mk, bad):
    # \bdtype does NOT match "param_dtype" ('_' is a word char): this verifies
    # the error message names the *compute* knob, i.e. the field_name plumbing.
    with pytest.raises(ValueError, match=r"\bdtype"):
        mk(dtype=bad)


@pytest.mark.parametrize("mk", PARAMS_FACTORIES)
def test_dtype_rejects_float64_without_x64(mk):
    assert not jax.config.jax_enable_x64
    with pytest.raises(ValueError):
        mk(dtype="float64")


@pytest.mark.parametrize("mk", PARAMS_FACTORIES)
def test_precision_knobs_independent(mk):
    p = mk(param_dtype="bfloat16", dtype="float32")
    assert p.param_dtype == "bfloat16" and p.dtype == "float32"
