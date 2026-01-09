import pytest
import numpy as np
from src.CHORD_Sensitivity.main import Telescope
from src.CHORD_Sensitivity.constants import C

@pytest.fixture
def chord():
    return Telescope("CHORD")

@pytest.fixture
def pathfinder():
    return Telescope("PATHFINDER")

def test_init_valid_names(chord, pathfinder):
    assert chord.name == "CHORD"
    assert chord.params['ndish_ew'] == 22
    assert pathfinder.name == "PATHFINDER"
    assert pathfinder.params['ndish_ns'] == 6

def test_init_invalid_name():
    with pytest.raises(ValueError):
        Telescope("INVALID")

def test_convert_freq_to_Hz_scalar(chord):
    freq_hz = chord.convert_freq_to_Hz(500e6)  # Hz input
    assert freq_hz == 500e6
    
    freq_mhz = chord.convert_freq_to_Hz(500)  # MHz input
    assert freq_mhz == 500e6
    
    freq_ghz = chord.convert_freq_to_Hz(0.5)  # GHz input
    assert freq_ghz == 500e6

def test_convert_freq_to_Hz_list(chord):
    freqs = chord.convert_freq_to_Hz([300, 1.0, 500e6])  # MHz, GHz, Hz
    assert freqs == [300e6, 1e9, 500e6]

def test_convert_freq_to_Hz_invalid(chord):
    with pytest.raises(ValueError):
        chord.convert_freq_to_Hz(200)  # Below min MHz
    with pytest.raises(ValueError):
        chord.convert_freq_to_Hz(2.0)  # Above max GHz

def test_P_FWHM_scalar(chord):
    # Note: Code returns radians despite docstring claiming degrees
    fwhm = chord.P_FWHM(1e9)  # 1 GHz = 1000 MHz
    wavelength = C / 1e9
    expected = 1.029 * (wavelength / 6.0)
    assert fwhm == pytest.approx(expected, abs=1e-5)

def test_P_FWHM_list(chord):
    fwhms = chord.P_FWHM([0.3, 1.5])  # GHz inputs
    wavelength1 = C / 0.3e9
    wavelength2 = C / 1.5e9
    expected = [1.029 * (wavelength1 / 6.0), 1.029 * (wavelength2 / 6.0)]
    np.testing.assert_array_almost_equal(fwhms, expected, decimal=5)

def test_S_FWHM(chord):
    # Note: Returns array of two values (EW and NS), in radians despite docstring
    fwhm = chord.S_FWHM(1e9)  # 1 GHz
    wavelength = C / 1e9
    max_ew = 6.3 * 22
    max_ns = 8.5 * 24
    expected = np.array([wavelength / max_ew, wavelength / max_ns])
    np.testing.assert_array_almost_equal(fwhm, expected, decimal=5)

def test_S_solid_angle(chord):
    solid_angle = chord.S_solid_angle(1e9)
    fwhm = chord.S_FWHM(1e9)
    expected = np.pi * np.prod(fwhm) / (4 * np.log(2))
    assert solid_angle == pytest.approx(expected, abs=1e-5)