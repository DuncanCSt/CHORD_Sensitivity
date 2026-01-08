"""
Tests for sensitivity calculations.
"""

import pytest
import numpy as np
from chord_sensitivity.sensitivity import calculate_sensitivity


def test_calculate_sensitivity_basic():
    """Test basic sensitivity calculation."""
    result = calculate_sensitivity(
        system_temperature=50.0,
        bandwidth=1e6,
        integration_time=3600.0,
        n_antennas=512
    )
    assert result > 0
    assert isinstance(result, float)


def test_calculate_sensitivity_single_antenna():
    """Test sensitivity with single antenna."""
    result = calculate_sensitivity(
        system_temperature=100.0,
        bandwidth=1e6,
        integration_time=1.0,
        n_antennas=1
    )
    expected = 100.0 / np.sqrt(1e6 * 1.0 * 1)
    assert np.isclose(result, expected)


def test_calculate_sensitivity_invalid_temperature():
    """Test that invalid system temperature raises error."""
    with pytest.raises(ValueError, match="System temperature must be positive"):
        calculate_sensitivity(
            system_temperature=-10.0,
            bandwidth=1e6,
            integration_time=1.0
        )


def test_calculate_sensitivity_invalid_bandwidth():
    """Test that invalid bandwidth raises error."""
    with pytest.raises(ValueError, match="Bandwidth must be positive"):
        calculate_sensitivity(
            system_temperature=50.0,
            bandwidth=-1e6,
            integration_time=1.0
        )


def test_calculate_sensitivity_invalid_time():
    """Test that invalid integration time raises error."""
    with pytest.raises(ValueError, match="Integration time must be positive"):
        calculate_sensitivity(
            system_temperature=50.0,
            bandwidth=1e6,
            integration_time=-1.0
        )


def test_calculate_sensitivity_invalid_antennas():
    """Test that invalid number of antennas raises error."""
    with pytest.raises(ValueError, match="Number of antennas must be at least 1"):
        calculate_sensitivity(
            system_temperature=50.0,
            bandwidth=1e6,
            integration_time=1.0,
            n_antennas=0
        )
