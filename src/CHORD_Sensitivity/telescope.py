from __future__ import annotations

from .constants import CHORD, PATHFINDER, K_B, OMEGA, C
from .helpers import convert_freq_to_Hz

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
    
    # def params(self) -> dict:
    #     """Return the telescope parameters."""
    #     return self.params
        
    def P_FWHM(self, freq: float | list[float]) -> float | list[float]:
        """
        Calculate the primary beam full-width half-maximum (FWHM).
        
        Parameters
        ----------
        freq : float | list[float]
            Observing frequency in Hz.
        
        Returns
        -------
        float | list[float]
            Primary beam FWHM in radians.
        """
        freq = convert_freq_to_Hz(freq, self.params)
        is_scalar = isinstance(freq, (int, float))
        freq_array = np.atleast_1d(freq)
        
        dish_diameter = self.params['dish_diameter']  # in meters
        wavelength = C / freq_array  # in meters
        fwhm_rad = 1.029 * (wavelength / dish_diameter)  # in radians
        
        return float(fwhm_rad[0]) if is_scalar else fwhm_rad.tolist()
    
    def S_FWHM(self, freq: float | list[float]) -> tuple[float, float] | list[tuple[float, float]]:
        """
        Calculate the synthesized beam full-width half-maximum (FWHM).
        
        Parameters
        ----------
        freq : float | list[float]
            Observing frequency in Hz.
        
        Returns
        -------
        tuple[float, float] | list[tuple[float, float]]
            Synthesized beam FWHM in radians as (EW, NS).
        """
        freq = convert_freq_to_Hz(freq, self.params)
        is_scalar = isinstance(freq, (int, float))
        freq_array = np.atleast_1d(freq)
        
        wavelength = C / freq_array  # in meters
        max_baseline_ew = self.params['dish_separation_ew'] * self.params['ndish_ew']  # in meters
        max_baseline_ns = self.params['dish_separation_ns'] * self.params['ndish_ns']  # in meters

        fwhm_ew = wavelength / max_baseline_ew  # in radians
        fwhm_ns = wavelength / max_baseline_ns  # in radians

        if is_scalar:
            return (float(fwhm_ew[0]), float(fwhm_ns[0]))
        return list(zip(fwhm_ew.tolist(), fwhm_ns.tolist()))
    
    def S_solid_angle(self, freq: float | list[float]) -> float | list[float]:
        """
        Calculate the synthesized beam solid angle.
        
        Parameters
        ----------
        freq : float | list[float]
            Observing frequency in Hz.
        
        Returns
        -------
        float | list[float]
            Synthesized beam solid angle in steradians.
        """
        s_fwhm = self.S_FWHM(freq)
        if isinstance(s_fwhm, tuple):
            return np.pi * np.prod(s_fwhm) / (4 * np.log(2))  # steradians

        s_fwhm_array = np.asarray(s_fwhm)  # shape: (n_freq, 2)
        solid_angle = np.pi * np.prod(s_fwhm_array, axis=1) / (4 * np.log(2))  # steradians
        return solid_angle.tolist()
    
    def D_phi(self, phi_offset: float | list[float], freq: float | list[float]) -> float | list[float]:
        """
        Calculate the primary beam response at an angular offset.
        
        .. math::
        D(\\phi) = \\exp\\left(-8\\ln 2 \\frac{\\phi^2}{\\theta_{\\mathrm{FWHM}}^2}\\right)     
        
        Parameters
        ----------
        phi_offset : float | list[float]
            Angular offset from the pointing center in radians.
        freq : float | list[float]
            Observing frequency in Hz.
        
        Returns
        -------
        float | list[float]
            Primary beam response (dimensionless).
        """
        freq = convert_freq_to_Hz(freq, self.params)
        is_scalar = isinstance(freq, (int, float))

        p_fwhm = np.atleast_1d(self.P_FWHM(freq))  # ensure array
        phi_offset_arr = np.atleast_1d(phi_offset)
        response = np.exp(-8 * np.log(2) * (phi_offset_arr / p_fwhm)**2)

        return float(response[0]) if is_scalar else response.tolist()
        

    def sigma_rms(
            self,
            delta_nu: float,
            freq: float | list[float],
            phi_offset: None | float | list[float] = None,
            T_background: None | float | list[float] = None,
            dec_deg: None | float = None
    ) -> float | list[float]:
        """
        Calculate the RMS noise level.

        .. math::
        \\sigma_{RMS} = \\frac{SEFD}{\\sqrt{2 ~\\Delta\\nu~N(N-1) }}
            \\sqrt{\\frac{2\\sqrt{2\\ln2}\\omega~cos\\delta}{\\sqrt{\\pi}
            \\theta_{FWHM}~D(\\phi)}}
        
        Parameters
        ----------
        delta_nu : float
            Channel width in Hz.
        freq : float | list[float]
            Observing frequency in Hz.
        phi_offset : None | float | list[float]
            Angular offset from the pointing center in radians.
        T_background : None | float | list[float]
            Sky temperature in K. (System + Background)
        dec_deg : None | float
            Declination in degrees. If None, use the telescope's latitude.
        Returns
        -------
        float | list[float]
            RMS noise level in Jy/beam.
        """
        is_scalar = isinstance(freq, (int, float))
        if not is_scalar:
            if phi_offset and not isinstance(phi_offset, (list)):
                raise ValueError("If freq is an array, phi_offset must also be an array.")
            elif phi_offset and len(freq) != len(phi_offset):
                raise ValueError("If either freq or phi_offset are arrays, they must have the same length.")
    
        params = self.params
        T_sys = float(params['Tsys'])           # K
        diameter = params['dish_diameter']      # m
        efficiency = params['efficiency']       # dimensionless
        dec_deg = dec_deg if dec_deg is not None else params['latitude']

        # Number of antennas
        N = params['ndish_ew'] * params['ndish_ns']

        # Effective collecting area per dish (m²)
        A_eff = np.pi * (diameter / 2)**2 * efficiency

        # Number of baselines contributing independent noise
        N_baselines = N * (N - 1)

        # Phi offset correction
        if phi_offset is None:
            phi_offset = 0.0 if is_scalar else [0.0] * len(freq)

        # Get total temperature (system + background)
        if T_background is None:
            T = np.array([T_sys]) if is_scalar else np.array([T_sys] * len(freq))
        else:
            T = np.atleast_1d(T_background)

        # System Equivalent Flux Density (Jy)
        SEFD = 2 * K_B * T / A_eff  # Jy

        p_fwhm = np.atleast_1d(self.P_FWHM(freq))  # radians
        d_phi = np.atleast_1d(self.D_phi(phi_offset, freq))  # dimensionless

        term1 = SEFD / np.sqrt(2 * delta_nu * N_baselines)
        term2 = np.sqrt((2 * np.sqrt(2 * np.log(2)) * OMEGA * np.cos(np.radians(dec_deg))) /
            (np.sqrt(np.pi) * p_fwhm * d_phi))
        
        rms = term1 * term2

        return float(rms[0]) if is_scalar else rms.tolist()
    
    def surface_temperature(self,
                            freq: float | list[float],
                            sigma_rms: None | float | list[float] = None,
                            delta_nu: None | float = None) -> float | list[float]:
        """
        Convert RMS noise level to surface temperature sensitivity.

        $$ \sigma_T = \frac{\sigma c^2}{\Omega_s 2 k \nu^2} $$

        Parameters
        ----------
        sigma_rms : None | float | list[float]
            RMS noise level in Jy/beam. If None, delta_nu must be provided to calculate it.
        delta_nu : None | float
            Channel width in Hz. Required if sigma_rms is None.
        freq : float | list[float]
            Observing frequency in Hz.

        Returns
        -------
        float | list[float]
            Surface temperature sensitivity in K.
        """
        if sigma_rms is None:
            if delta_nu is None:
                raise ValueError("If sigma_rms is not provided, delta_nu must be provided to calculate it.")
            sigma_rms = self.sigma_rms(delta_nu=delta_nu, freq=freq)
        
        is_scalar = isinstance(sigma_rms, (int, float))
        freq = convert_freq_to_Hz(freq, self.params)  # Hz
        sigma_array = np.atleast_1d(sigma_rms)  # Jy/beam

        omega_s = self.S_solid_angle(freq)  # steradians

        sigma_array = np.atleast_1d(sigma_rms)  # Jy/beam
        omega_s_array = np.atleast_1d(omega_s)  # steradians
        freq_array = np.atleast_1d(freq)  # Hz

        sigma_T = (sigma_array * C**2) / (omega_s_array * 2 * K_B * freq_array**2)  # K
        return float(sigma_T[0]) if is_scalar else sigma_T.tolist()
