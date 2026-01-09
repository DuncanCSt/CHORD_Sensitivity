from constants import CHORD, PATHFINDER, K_B, OMEGA, C
from typing import Literal
from numpy.typing import NDArray
import numpy as np

class Telescope:
    """A simple telescope class."""
    
    def __init__(self, name: Literal["CHORD", "PATHFINDER"]):
        """
        Initialize a Telescope object.
        
        Parameters
        ----------
        name : str, optional
            Name of the telescope
        """
        self.name = name
        if name == "CHORD":
            self.params = CHORD
        elif name == "PATHFINDER":
            self.params = PATHFINDER
        else:
            raise ValueError("Telescope name not recognized. Choose either 'CHORD' or 'PATHFINDER'.")
        
    def P_FWHM(self, freq: float | list[float]) -> float | list[float]:
        """
        Calculate the primary beam full-width half-maximum (FWHM).
        
        Parameters
        ----------
        freq : float
            Observing frequency in Hz.
        
        Returns
        -------
        float
            Primary beam FWHM in degrees.
        """
        freq = self.convert_freq_to_Hz(freq)
        is_scalar = isinstance(freq, (int, float))
        freq_array = np.atleast_1d(freq)
        
        dish_diameter = self.params['dish_diameter']  # in meters
        wavelength = C / freq_array  # in meters
        fwhm_rad = 1.029 * (wavelength / dish_diameter)  # in radians
        fwhm_deg = fwhm_rad * (180.0 / np.pi)  # Convert to degrees
        
        return float(fwhm_rad[0]) if is_scalar else fwhm_rad.tolist()
    
    def S_FWHM(self, freq: float | list[float]) -> NDArray[float]:
        """
        Calculate the synthesized beam full-width half-maximum (FWHM).
        
        Parameters
        ----------
        freq : float
            Observing frequency in Hz.
        
        Returns
        -------
        float
            Synthesized beam FWHM in arcseconds.
        """
        freq = self.convert_freq_to_Hz(freq)
        is_scalar = isinstance(freq, (int, float))
        freq_array = np.atleast_1d(freq)
        
        wavelength = C / freq_array  # in meters
        max_baseline = np.array([self.params['dish_separation_ew'] * self.params['ndish_ew'],
                               self.params['dish_separation_ns'] * self.params['ndish_ns']])  # in meters
        fwhm_rad = wavelength / max_baseline  # in radians
        fwhm_arcsec = fwhm_rad * (180.0 / np.pi) * 3600.0  # Convert to arcseconds
        
        return fwhm_rad
    
    def S_solid_angle(self, freq: float | list[float]) -> NDArray[float]:
        """
        Calculate the synthesized beam solid angle.
        
        Parameters
        ----------
        freq : float
            Observing frequency in Hz.
        
        Returns
        -------
        float
            Synthesized beam solid angle in steradians.
        """
        return np.pi * np.prod(self.S_FWHM(freq)) / (4 * np.log(2))  # steradians
    
    def convert_freq_to_Hz(self, freq) -> float | list[float]:
        """
        Automatically detect range and convert to Hz.
        
        Parameters
        ----------
        freq : float or array-like
            Frequency value(s) in Hz, MHz, or GHz.
        
        Returns
        -------
        float or list[float]
            Frequency value(s) in Hz.
        """
        
        min_freq_MHz = self.params['frequencyMin']
        max_freq_MHz = self.params['frequencyMax']
        
        # Check if input is array-like
        is_array = isinstance(freq, (list, tuple, np.ndarray))
        freq_array = np.asarray(freq) if is_array else np.array([freq])
        
        # Initialize result array
        result = np.zeros_like(freq_array, dtype=float)
        
        for i, f in enumerate(freq_array):
            # Check if frequency is in Hz
            if min_freq_MHz * 1e6 <= f <= max_freq_MHz * 1e6:
                result[i] = f
            # Check if frequency is in MHz
            elif min_freq_MHz <= f <= max_freq_MHz:
                result[i] = f * 1e6
            # Check if frequency is in GHz
            elif (min_freq_MHz / 1e3) <= f <= (max_freq_MHz / 1e3):
                result[i] = f * 1e9
            else:
                raise ValueError(f"Frequency {f} is out of the telescope's operating range. Or not one of the supported units (Hz, MHz, GHz).")
        
        return result.tolist() if is_array else float(result[0])


    