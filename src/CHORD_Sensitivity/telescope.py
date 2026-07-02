from __future__ import annotations

from .constants import CHORD, PATHFINDER, K_B, OMEGA, C
from .helpers import convert_freq_to_Hz

from typing import Literal
from numpy.typing import NDArray
import numpy as np
import warnings

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

    def P_FWHM(self, freq: float | NDArray) -> float | NDArray:
        """
        Calculate the primary beam full-width half-maximum (FWHM).

        Parameters
        ----------
        freq : float | NDArray
            Observing frequency in Hz.

        Returns
        -------
        float | NDArray
            Primary beam FWHM in radians.
        """
        freq = convert_freq_to_Hz(freq, self.params)
        is_scalar = np.ndim(freq) == 0
        freq_array = np.atleast_1d(freq)

        dish_diameter = self.params['dish_diameter']  # in meters
        wavelength = C / freq_array  # in meters
        fwhm_rad = 1.029 * (wavelength / dish_diameter)  # in radians

        return float(fwhm_rad[0]) if is_scalar else fwhm_rad

    def S_FWHM(self, freq: float | NDArray) -> tuple[float, float] | NDArray:
        """
        Calculate the synthesized beam full-width half-maximum (FWHM).

        Parameters
        ----------
        freq : float | NDArray
            Observing frequency in Hz.

        Returns
        -------
        tuple[float, float] | NDArray
            Synthesized beam FWHM in radians. For a scalar frequency a
            ``(EW, NS)`` tuple is returned; for an array of frequencies an
            array of shape ``(n_freq, 2)`` with columns ``(EW, NS)``.
        """
        freq = convert_freq_to_Hz(freq, self.params)
        is_scalar = np.ndim(freq) == 0
        freq_array = np.atleast_1d(freq)

        wavelength = C / freq_array  # in meters
        # Longest baseline is the center-to-center distance between the two
        # outermost dishes: (ndish - 1) gaps of the dish separation.
        max_baseline_ew = self.params['dish_separation_ew'] * (self.params['ndish_ew'] - 1)  # in meters
        max_baseline_ns = self.params['dish_separation_ns'] * (self.params['ndish_ns'] - 1)  # in meters

        fwhm_ew = wavelength / max_baseline_ew  # in radians
        fwhm_ns = wavelength / max_baseline_ns  # in radians

        if is_scalar:
            return (float(fwhm_ew[0]), float(fwhm_ns[0]))
        return np.column_stack([fwhm_ew, fwhm_ns])

    def S_solid_angle(self, freq: float | NDArray) -> float | NDArray:
        """
        Calculate the synthesized beam solid angle.

        Parameters
        ----------
        freq : float | NDArray
            Observing frequency in Hz.

        Returns
        -------
        float | NDArray
            Synthesized beam solid angle in steradians.
        """
        s_fwhm = self.S_FWHM(freq)
        if isinstance(s_fwhm, tuple):
            return float(np.pi * np.prod(s_fwhm) / (4 * np.log(2)))  # steradians

        # s_fwhm has shape (n_freq, 2)
        solid_angle = np.pi * np.prod(s_fwhm, axis=1) / (4 * np.log(2))  # steradians
        return solid_angle

    def D_phi(self, phi_offset: float | NDArray, freq: float | NDArray) -> float | NDArray:
        """
        Calculate the primary beam response at an angular offset.

        .. math::
        D(\\phi) = \\exp\\left(-8\\ln 2 \\frac{\\phi^2}{\\theta_{\\mathrm{FWHM}}^2}\\right)

        Parameters
        ----------
        phi_offset : float | NDArray
            Angular offset from the pointing center in radians.
        freq : float | NDArray
            Observing frequency in Hz.

        Returns
        -------
        float | NDArray
            Primary beam response (dimensionless).
        """
        freq = convert_freq_to_Hz(freq, self.params)
        is_scalar = np.ndim(freq) == 0 and np.ndim(phi_offset) == 0

        p_fwhm = np.atleast_1d(self.P_FWHM(freq))  # ensure array
        phi_offset_arr = np.atleast_1d(phi_offset)
        response = np.exp(-8 * np.log(2) * (phi_offset_arr / p_fwhm)**2)

        return float(response[0]) if is_scalar else response

    def tau_eff(self,
                phi_offset: float | NDArray,
                freq: float | NDArray,
                dec_deg: float | NDArray
            ) -> float | NDArray:

        """Calculate the effective integration time at an angular offset and declination.
        .. math::
        \\tau_{eff} = D(\\phi)\\sqrt{\frac{\\pi}{8\\ln2}}\\frac{\\theta_{\\mathrm{FWHM}}}{\\omega \\cos\\delta}

        Parameters
        ----------
        phi_offset : float | NDArray
            Angular offset from the pointing center in radians.
        freq : float | NDArray
            Observing frequency in Hz.
        dec_deg : float | NDArray
            Declination in degrees.

        Returns
        -------
        float | NDArray
            Effective integration time in seconds.
        """

        freq = convert_freq_to_Hz(freq, self.params)
        is_scalar = (np.ndim(freq) == 0
                     and np.ndim(phi_offset) == 0
                     and np.ndim(dec_deg) == 0)
        p_fwhm = np.atleast_1d(self.P_FWHM(freq))  # ensure array
        phi_offset_arr = np.atleast_1d(phi_offset)
        dec_arr = np.atleast_1d(dec_deg)
        tau_eff = np.atleast_1d(self.D_phi(phi_offset_arr, freq)) * np.sqrt(np.pi / (8 * np.log(2))) * (p_fwhm / (OMEGA * np.cos(np.radians(dec_arr))))
        return float(tau_eff[0]) if is_scalar else tau_eff


    def sigma_rms(
            self,
            delta_nu: float,
            freq: float | NDArray,
            phi_offset: None | float | NDArray = None,
            T_background: None | float | NDArray = None,
            dec_deg: None | float | NDArray = None
    ) -> float | NDArray:
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
        freq : float | NDArray
            Observing frequency in Hz.
        phi_offset : None | float | NDArray
            Angular offset from the pointing center in radians.
        T_background : None | float | NDArray
            Sky temperature in K. (System + Background)
        dec_deg : None | float | NDArray
            Declination in degrees. If None, use the telescope's latitude.
        Returns
        -------
        float | NDArray
            RMS noise level in Jy/beam.
        """
        params = self.params
        is_scalar = np.ndim(freq) == 0

        # 1. Validate that, when freq is an array, a supplied phi_offset is an
        #    array of matching length (the two are paired element-by-element).
        if not is_scalar and phi_offset is not None:
            if np.ndim(phi_offset) == 0:
                raise ValueError("If freq is an array, phi_offset must also be an array.")
            if np.size(freq) != np.size(phi_offset):
                raise ValueError("freq and phi_offset must have the same length.")

        # 2. Resolve optional arguments to concrete values. Scalars broadcast
        #    against array-valued freq/dec, so no per-element filling is needed.
        if phi_offset is None:
            phi_offset = 0.0
        if dec_deg is None:
            dec_deg = params['latitude']
        if T_background is None:
            T_background = float(params['Tsys'])
        T = np.atleast_1d(T_background)  # total (system + background) temperature, K

        # 3. Telescope-derived constants.
        N = params['ndish_ew'] * params['ndish_ns']                              # number of antennas
        N_baselines = N * (N - 1)                                                 # independent-noise baselines
        A_eff = np.pi * (params['dish_diameter'] / 2)**2 * params['efficiency']   # effective area per dish, m²

        # 4. Noise terms.
        SEFD = 2 * K_B * T / A_eff                                                # system equivalent flux density, Jy
        tau_eff = np.atleast_1d(self.tau_eff(phi_offset, freq, dec_deg))          # effective integration time, s

        # 5. Combine via the radiometer equation.
        rms = SEFD / np.sqrt(2 * delta_nu * tau_eff * N_baselines)               # Jy/beam

        # Collapse to a scalar only when the result is genuinely one value. A
        # scalar freq paired with an array T_background (or dec_deg) still
        # yields one rms per element, which must stay an array.
        return float(rms[0]) if (is_scalar and rms.size == 1) else rms

    def surface_temperature(self,
                            freq: float | NDArray,
                            sigma_rms: None | float | NDArray = None,
                            delta_nu: None | float = None,
                            smoothed_resolution: None | float = None) -> float | NDArray:
        r"""
        Convert RMS noise level to surface temperature sensitivity.

        $$ \sigma_T = \frac{\sigma c^2}{\Omega_s 2 k \nu^2} $$

        Parameters
        ----------
        sigma_rms : None | float | NDArray
            RMS noise level in Jy/beam. If None, delta_nu must be provided to calculate it.
        delta_nu : None | float
            Channel width in Hz. Required if sigma_rms is None.
        freq : float | NDArray
            Observing frequency in Hz.
        smoothed_resolution : None | float
            Smoothed resolution (FWHM) in arcmin. If None, the native synthesized beam
            solid angle is used. If provided, the native map is treated as smoothed to this
            coarser beam, averaging N = omega_smooth / omega_native independent synthesized
            beams so the brightness-temperature noise scales as 1/sqrt(N).

        Returns
        -------
        float | NDArray
            Surface temperature sensitivity in K.
        """
        if sigma_rms is None:
            if delta_nu is None:
                raise ValueError("If sigma_rms is not provided, delta_nu must be provided to calculate it.")
            sigma_rms = self.sigma_rms(delta_nu=delta_nu, freq=freq)

        is_scalar = np.ndim(sigma_rms) == 0
        freq = convert_freq_to_Hz(freq, self.params)  # Hz

        if smoothed_resolution is not None:
            # Smoothing the native map to a coarser beam averages N = omega_smooth / omega_native
            # independent synthesized beams, so the brightness-temperature noise drops as 1/sqrt(N).
            # This is equivalent to using the geometric-mean solid angle in the sigma_T conversion.
            omega_smooth = np.pi * (np.radians(smoothed_resolution / 60))**2 / (4 * np.log(2))  # steradians
            omega_native = self.S_solid_angle(freq)  # steradians
            if np.any(omega_smooth < omega_native):
                warnings.warn(
                    "smoothed_resolution is finer than the native synthesized beam "
                    "(omega_smooth < omega_native); this is not a valid smoothing. "
                    "Continuing with the calculation.",
                    stacklevel=2,
                )
            omega_s = np.sqrt(omega_native * omega_smooth)  # effective solid angle, steradians
        else:
            omega_s = self.S_solid_angle(freq)  # steradians

        sigma_array = np.atleast_1d(sigma_rms)  # Jy/beam
        omega_s_array = np.atleast_1d(omega_s)  # steradians
        freq_array = np.atleast_1d(freq)  # Hz

        sigma_T = (sigma_array * C**2) / (omega_s_array * 2 * K_B * freq_array**2)  # K
        return float(sigma_T[0]) if is_scalar else sigma_T
