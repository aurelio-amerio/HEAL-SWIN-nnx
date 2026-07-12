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
