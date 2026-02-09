import pytest
import numpy as np
from CHORD_Sensitivity import Telescope
from CHORD_Sensitivity import C
from CHORD_Sensitivity.helpers import convert_freq_to_Hz

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
    freq_hz = convert_freq_to_Hz(500e6, chord.params)  # Hz input
    assert freq_hz == 500e6
    
    freq_mhz = convert_freq_to_Hz(500, chord.params)  # MHz input
    assert freq_mhz == 500e6
    
    freq_ghz = convert_freq_to_Hz(0.5, chord.params)  # GHz input
    assert freq_ghz == 500e6

def test_convert_freq_to_Hz_list(chord):
    freqs = convert_freq_to_Hz([300, 1.0, 500e6], chord.params)  # MHz, GHz, Hz
    assert freqs == [300e6, 1e9, 500e6]

def test_convert_freq_to_Hz_invalid(chord):
    with pytest.raises(ValueError):
        convert_freq_to_Hz(200, chord.params)  # Below min MHz
    with pytest.raises(ValueError):
        convert_freq_to_Hz(2.0, chord.params)  # Above max GHz

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
    # Returns tuple of two values (EW and NS), in radians
    fwhm = chord.S_FWHM(1e9)  # 1 GHz
    wavelength = C / 1e9
    max_ew = 6.3 * 22
    max_ns = 8.5 * 24
    expected = (wavelength / max_ew, wavelength / max_ns)
    assert fwhm == pytest.approx(expected, abs=1e-5)

def test_S_FWHM_list_returns_list_of_tuples(chord):
    freqs = [0.8, 1.0]  # GHz
    fwhm = chord.S_FWHM(freqs)
    assert isinstance(fwhm, list)
    assert len(fwhm) == 2
    assert all(isinstance(v, tuple) and len(v) == 2 for v in fwhm)

def test_S_solid_angle(chord):
    solid_angle = chord.S_solid_angle(1e9)
    fwhm = chord.S_FWHM(1e9)
    expected = np.pi * np.prod(fwhm) / (4 * np.log(2))
    assert solid_angle == pytest.approx(expected, abs=1e-5)

def test_D_phi_scalar(chord):
    freq = 1e9
    phi_offset = 0.01
    p_fwhm = chord.P_FWHM(freq)
    expected = np.exp(-8 * np.log(2) * (phi_offset / p_fwhm) ** 2)
    assert chord.D_phi(phi_offset, freq) == pytest.approx(expected, abs=1e-8)

def test_D_phi_list(chord):
    freqs = [0.8, 1.0]  # GHz
    phi_offsets = [0.01, 0.02]
    p_fwhm = np.asarray(chord.P_FWHM(freqs))
    expected = np.exp(-8 * np.log(2) * (np.asarray(phi_offsets) / p_fwhm) ** 2)
    np.testing.assert_allclose(chord.D_phi(phi_offsets, freqs), expected, rtol=1e-10, atol=0.0)

def test_sigma_rms_scalar(chord):
    chord.params["dec_deg"] = 49.320750
    sigma = chord.sigma_rms(delta_nu=390625.0, central_freq=1e9, phi_offset=0.0)
    assert sigma > 0
    assert np.isfinite(sigma)

def test_sigma_rms_list(chord):
    chord.params["dec_deg"] = 49.320750
    sigmas = chord.sigma_rms(
        delta_nu=390625.0,
        central_freq=[0.8, 1.0],
        phi_offset=[0.0, 0.01],
    )
    assert isinstance(sigmas, list)
    assert len(sigmas) == 2
    assert all(s > 0 and np.isfinite(s) for s in sigmas)

def test_sigma_rms_raises_on_mismatched_inputs(chord):
    chord.params["dec_deg"] = 49.320750
    with pytest.raises(ValueError):
        chord.sigma_rms(delta_nu=390625.0, central_freq=[0.8, 1.0], phi_offset=0.0)
    with pytest.raises(ValueError):
        chord.sigma_rms(delta_nu=390625.0, central_freq=[0.8, 1.0], phi_offset=[0.0])

def test_surface_temperature_scalar(chord):
    sigma_rms = 1.0
    freq = 1e9
    omega_s = chord.S_solid_angle(freq)
    expected = (sigma_rms * C**2) / (omega_s * 2 * 1.38e3 * freq**2)
    assert chord.surface_temperature(sigma_rms=sigma_rms, freq=freq) == pytest.approx(expected, rel=1e-10)

def test_surface_temperature_list(chord):
    sigma_rms = [1.0, 1.2]
    freqs = [0.8, 1.0]  # GHz
    freqs_hz = np.asarray(convert_freq_to_Hz(freqs, chord.params))
    omega_s = np.atleast_1d(chord.S_solid_angle(freqs))
    expected = (np.asarray(sigma_rms) * C**2) / (omega_s * 2 * 1.38e3 * freqs_hz**2)
    np.testing.assert_allclose(chord.surface_temperature(sigma_rms=sigma_rms, freq=freqs), expected, rtol=1e-10, atol=0.0)
