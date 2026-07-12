import pytest

from heal_swin_nnx import DataSpec


def test_default_is_full_sphere():
    ds = DataSpec(dim_in=12 * 16 ** 2, f_in=1, f_out=1)
    assert ds.base_pixels == list(range(12))
    assert ds.base_pix == 12


def test_legacy_base_pix_resolves_to_prefix():
    ds = DataSpec(dim_in=8 * 16 ** 2, f_in=3, f_out=5, base_pix=8)
    assert ds.base_pixels == list(range(8))
    assert ds.base_pix == 8


def test_explicit_base_pixels():
    ds = DataSpec(dim_in=4 * 16 ** 2, f_in=1, f_out=1, base_pixels=[8, 9, 10, 11])
    assert ds.base_pixels == [8, 9, 10, 11]
    assert ds.base_pix == 4


def test_base_pix_and_base_pixels_must_agree():
    ds = DataSpec(dim_in=4 * 16 ** 2, f_in=1, f_out=1, base_pix=4, base_pixels=[8, 9, 10, 11])
    assert ds.base_pix == 4
    with pytest.raises(ValueError):
        DataSpec(dim_in=4 * 16 ** 2, f_in=1, f_out=1, base_pix=8, base_pixels=[8, 9, 10, 11])


@pytest.mark.parametrize("bad", [[0, 0, 1], [3, 2], [-1, 0], [11, 12]])
def test_invalid_base_pixels_rejected(bad):
    with pytest.raises(ValueError):
        DataSpec(dim_in=len(bad) * 16 ** 2, f_in=1, f_out=1, base_pixels=bad)
