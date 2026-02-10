import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from pygdsm import GlobalSkyModel16

from .helpers import convert_freq_to_Hz
from .telescope import Telescope
from .constants import CHORD, PATHFINDER

class SkyMap:
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

        self.telescope = Telescope(name)

    def sky_noise(self,
            delta_nu: float,
            freq: float | list[float]
        ) -> list[np.ndarray]:
        """

        """

        freq = convert_freq_to_Hz(freq, self.params)
        if isinstance(freq, (float, int)):
            freq = [freq]

        gsm = GlobalSkyModel16('Hz')
        background_maps = [gsm.generate(f) for f in freq]
        nside = hp.get_nside(background_maps[0])
        npix = hp.nside2npix(nside)
        dec_min = self.params['min_dec']
        dec_max = self.params['max_dec']

        theta_gal, phi_gal = hp.pix2ang(nside, np.arange(npix))

        # Convert to celestial (Galactic -> Equatorial J2000)
        rot = hp.Rotator(coord=["G", "C"])
        theta_eq, phi_eq = rot(theta_gal, phi_gal)

        # Declination
        dec_deg = np.degrees(0.5 * np.pi - theta_eq)
        mask = (dec_deg >= dec_min) & (dec_deg <= dec_max)

        rms_maps = []
        for i, f in enumerate(freq):
            Tsky = background_maps[i]
            Tsky[Tsky <=0] = np.nan # Fill negative values with NaN
            Tsky[~mask] = np.nan # Mask out pixels outside declination range

            Tsky = Tsky + self.params['Tsys']
            rms_map = Tsky.copy() # Placeholder for actual RMS calculation
            
            rms_map[~mask] = self.telescope.surface_temperature(
                sigma_rms = self.telescope.sigma_rms(
                    delta_nu=delta_nu,
                    central_freq=f,
                    phi_offset=None,
                    T_background=Tsky[~mask]
                ),
                freq=f
            )
            rms_maps.append(rms_map)
        
        return rms_maps
    
    def plot_sky_noise(self, 
            delta_nu: float,
            freqs: list[float]) -> None:

        sky_maps = self.sky_noise(delta_nu=delta_nu, freq=freqs)
        for i, sky_map in enumerate(sky_maps):
            hp.mollview(sky_map, title=f"Sky Noise at {freqs[i]} Hz", unit="K", norm="log")
            plt.show()
        
        return
            
            


