import dataclasses
import json

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
