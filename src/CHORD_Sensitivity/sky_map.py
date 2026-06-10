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
            freq: float | list[float],
            type: Literal["background_only", "system_only", "total_noise"] = "total_noise",
            CHORD_range: bool = True,
        ) -> list[np.ndarray]:
        """
        Compute brightness-temperature sky maps over the observable sky.

        For each frequency a HEALPix map (in the GSM's native Galactic
        pixelisation) is returned. Pixels within the telescope's declination
        range ``[min_dec, max_dec]`` hold the requested quantity; pixels
        outside that range are set to ``NaN``.

        Parameters
        ----------
        delta_nu : float
            Channel bandwidth in Hz.
        freq : float | list[float]
            Observing frequency/frequencies (Hz, MHz, or GHz; auto-detected).
        type : {"background_only", "system_only", "total_noise"}
            Quantity to map over the observable sky:
            - ``"background_only"``: the masked GSM sky temperature [K].
            - ``"system_only"``: surface brightness sensitivity from the system
              temperature alone [K].
            - ``"total_noise"``: surface brightness sensitivity from the
              system + sky-background temperature [K].

        Returns
        -------
        list[np.ndarray]
            One HEALPix map per requested frequency.
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
        T_sys = self.params['Tsys']

        # Per-pixel declination: the GSM map is in Galactic coordinates, so
        # rotate each pixel to Equatorial (J2000) and take dec = 90 - colat.
        theta_gal, phi_gal = hp.pix2ang(nside, np.arange(npix))
        theta_eq, _ = hp.Rotator(coord=["G", "C"])(theta_gal, phi_gal)
        dec_deg = np.degrees(0.5 * np.pi - theta_eq)
        if CHORD_range:
            mask = (dec_deg >= dec_min) & (dec_deg <= dec_max)
        else:
            mask = np.ones_like(dec_deg, dtype=bool)
        dec_obs = dec_deg[mask]

        sky_maps = []
        for i, f in enumerate(freq):
            Tsky = background_maps[i]

            if type == "background_only":
                # Just the sky temperature itself, masked to the observable sky.
                sky_map = Tsky.copy()
                sky_map[sky_map <= 0] = np.nan      # drop unphysical pixels
                sky_map[~mask] = np.nan             # hide un-observable sky
                sky_maps.append(sky_map)
                continue

            # Total temperature entering the SEFD for each observable pixel.
            # sigma_rms treats T_background as the *total* temperature (it does
            # not add T_sys itself), so we add it here.
            if type == "system_only":
                T_background = T_sys
            elif type == "total_noise":
                T_background = T_sys + Tsky[mask]
            else:
                raise ValueError(f"Unknown type {type!r}. Choose 'background_only', "
                                 "'system_only', or 'total_noise'.")

            # Per-pixel sensitivity. Passing the per-pixel declination captures
            # the drift-scan cos(dec) dependence; the scalar freq broadcasts.
            sigma_rms = self.telescope.sigma_rms(
                delta_nu=delta_nu,
                freq=f,
                phi_offset=None,
                T_background=T_background,
                dec_deg=dec_obs,
            )
            sigma_T = self.telescope.surface_temperature(sigma_rms=sigma_rms, freq=f)

            sky_map = np.full(npix, np.nan)
            sky_map[mask] = sigma_T
            sky_maps.append(sky_map)

        return sky_maps
    
    def plot_sky_noise(self,
            delta_nu: float,
            freqs: list[float],
            title: str | None = None,
            type: Literal["background_only", "system_only", "total_noise"] = "total_noise",
            chord_range: bool = True,
            save_path: str | None = None,
            unit: str = "K",
            freq_unit: str = "MHz",
            cmap: str = "viridis",
            vmin: float | None = None,
            vmax: float | None = None,
            ) -> plt.Figure:
        """
        Plot the sky-noise map for each frequency as a 2-by-n grid of Mollview
        panels, where n = ceil(len(freqs) / 2).

        Parameters
        ----------
        delta_nu : float
            Channel bandwidth, passed through to ``sky_noise``.
        freqs : list[float]
            Frequencies to plot (same units as accepted by ``sky_noise``).
        save_path : str | None
            If given, the figure is written here; otherwise it is only shown.
        unit, freq_unit : str
            Labels for the colour bar and the per-panel frequency title.
        cmap : str
            Matplotlib colormap name.
        vmin, vmax : float | None
            Shared colour limits. If omitted, the 0.5/99th percentiles across
            all maps are used so a single colour scale spans every panel.

        Returns
        -------
        matplotlib.figure.Figure
            The figure holding the grid of panels.
        """
        maps = self.sky_noise(delta_nu=delta_nu, freq=freqs, type=type, CHORD_range=chord_range)

        nrows = 2
        ncols = int(np.ceil(len(maps) / nrows))

        # Shared colour scale across all panels (robust to NaN-masked pixels).
        if vmin is None or vmax is None:
            all_vals = np.hstack([m[np.isfinite(m)] for m in maps])
            vmin = np.nanpercentile(all_vals, 0.5) if vmin is None else vmin
            vmax = np.nanpercentile(all_vals, 99) if vmax is None else vmax

        fig = plt.figure(figsize=(4 * ncols, 2.2 * nrows))
        for i, (map_data, f) in enumerate(zip(maps, freqs), start=1):
            med = np.nanmedian(map_data)
            hp.mollview(
                map_data,
                coord="C",
                sub=(nrows, ncols, i),
                title=f"{f:.1f} {freq_unit}\n(median = {med:.1f} {unit})",
                unit=unit,
                cmap=cmap,
                min=vmin,
                max=vmax,
                cbar=False,
                notext=True,
                rot=(120, 0),
            )
            plt.gca().title.set_fontsize(14)

        # Single shared colour bar for the whole grid.
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
            ax=fig.axes,
            orientation="vertical",
            shrink=0.8,
            pad=0.02,
        )
        cbar.set_label(f"Temperature ({unit})", fontsize=14)

        if title is not None:
            plt.suptitle(title, fontsize=16)

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig
            
