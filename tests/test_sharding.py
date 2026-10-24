import jax
import pytest

pytestmark = pytest.mark.sharding

requires_8_devices = pytest.mark.skipif(
    jax.device_count() < 8,
    reason="needs XLA_FLAGS=--xla_force_host_platform_device_count=8")


def test_eight_cpu_devices_present():
    assert jax.device_count() == 8
    assert all(d.platform == "cpu" for d in jax.devices())
