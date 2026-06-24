import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from numpy.typing import NDArray
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

    def _gsm_maps(self, freq: NDArray) -> tuple[list[NDArray], int, int, NDArray]:
        """
        Generate the GSM background maps and the per-pixel declination grid.

        Parameters
        ----------
        freq : NDArray
            Observing frequencies in Hz.

        Returns
        -------
        background_maps : list[NDArray]
            One GSM brightness-temperature map per frequency [K].
        nside, npix : int
            HEALPix resolution parameters of the maps.
        dec_deg : NDArray
            Per-pixel declination in degrees. The GSM maps are in Galactic
            coordinates, so each pixel is rotated to Equatorial (J2000) and
            dec is taken as ``90 - colatitude``.
        """
        gsm = GlobalSkyModel16('Hz')
        background_maps = [gsm.generate(f) for f in freq]
        nside = hp.get_nside(background_maps[0])
        npix = hp.nside2npix(nside)

        theta_gal, phi_gal = hp.pix2ang(nside, np.arange(npix))
        theta_eq, _ = hp.Rotator(coord=["G", "C"])(theta_gal, phi_gal)
        dec_deg = np.degrees(0.5 * np.pi - theta_eq)
        return background_maps, nside, npix, dec_deg

    def sky_noise(self,
            delta_nu: float,
            freq: float | NDArray,
            type: Literal["background_only", "system_only", "total_noise"] = "total_noise",
            CHORD_range: bool = True,
        ) -> list[NDArray]:
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
        freq : float | NDArray
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
        list[NDArray]
            One HEALPix map per requested frequency.
        """
        # convert_freq_to_Hz returns a float for scalar input and a list for
        # array-like input; atleast_1d normalises both (and bare numpy arrays)
        # to a 1-D array we can iterate over.
        freq = np.atleast_1d(convert_freq_to_Hz(freq, self.params))
        background_maps, nside, npix, dec_deg = self._gsm_maps(freq)
        dec_min = self.params['min_dec']
        dec_max = self.params['max_dec']
        T_sys = self.params['Tsys']

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
            freqs: list[float] | NDArray,
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
        freqs : list[float] | NDArray
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
        return self._plot_map_grid(
            maps, freqs, title=title, save_path=save_path, unit=unit,
            freq_unit=freq_unit, cmap=cmap, vmin=vmin, vmax=vmax,
        )

    def _plot_map_grid(self,
            maps: list[NDArray],
            freqs: list[float] | NDArray,
            title: str | None = None,
            save_path: str | None = None,
            unit: str = "K",
            freq_unit: str = "MHz",
            cmap: str = "viridis",
            vmin: float | None = None,
            vmax: float | None = None,
            ) -> plt.Figure:
        """
        Render one HEALPix map per frequency as a 2-by-n grid of Mollview
        panels sharing a single colour bar, where n = ceil(len(maps) / 2).

        Parameters
        ----------
        maps : list[NDArray]
            One HEALPix map per frequency (e.g. the output of ``sky_noise``).
        freqs : list[float] | NDArray
            Frequencies labelling each panel (same order as ``maps``).
        title : str | None
            Optional figure-level super-title.
        save_path : str | None
            If given, the figure is written here; otherwise it is only shown.
        unit, freq_unit : str
            Labels for the colour bar and the per-panel frequency title. When
            ``unit == "mK"`` the maps are scaled from K to mK before plotting.
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
        if unit == "mK":
            maps = [m * 1e3 for m in maps]  # convert K to mK

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

    def plot_strip(self,
            delta_nu: float,
            freqs: list[float] | NDArray,
            title: str | None = None,
            type: Literal["background_only", "system_only", "total_noise"] = "total_noise",
            dec_deg: float | None = None,
            strip_halfwidth: float = 10.0,
            min_beam_response: float = 0.5,
            save_path: str | None = None,
            unit: str = "K",
            freq_unit: str = "MHz",
            cmap: str = "viridis",
            vmin: float | None = None,
            vmax: float | None = None,
            ) -> plt.Figure:
        """
        Plot the sky-noise map for a *single fixed pointing* declination.

        Unlike ``plot_sky_noise`` (which assumes a drift scan that crosses
        every declination at beam centre), this models CHORD pointed only at
        ``dec_deg``. Each pixel's offset from that pointing centre feeds the
        primary-beam attenuation ``D(phi)`` in ``sigma_rms`` (via
        ``phi_offset``), so sensitivity degrades away from the pointing dec.
        Pixels more than ``strip_halfwidth`` degrees from ``dec_deg`` lie
        outside the strip and are set to ``NaN``.

        Far from the pointing centre the beam response ``D(phi)`` falls to
        essentially zero, which sends the effective integration time to zero
        and makes ``sigma_rms`` diverge (values of order 1e12). Those pixels
        are not really observed, so pixels whose beam response drops below
        ``min_beam_response`` are also masked to ``NaN``. This cut is applied
        per-frequency, since the beam (and hence the observable strip width)
        shrinks with increasing frequency.

        Parameters
        ----------
        delta_nu : float
            Channel bandwidth in Hz.
        freqs : list[float] | NDArray
            Observing frequency/frequencies (Hz, MHz, or GHz; auto-detected).
        title : str | None
            Optional figure-level super-title.
        type : {"background_only", "system_only", "total_noise"}
            Quantity to map (see ``sky_noise``).
        dec_deg : float | None
            Pointing-centre declination in degrees. Defaults to the
            telescope's latitude (zenith pointing).
        strip_halfwidth : float
            Half-width of the observable strip in degrees; pixels farther than
            this from ``dec_deg`` are masked to ``NaN``. Defaults to 10.
        min_beam_response : float
            Minimum primary-beam response ``D(phi)`` a pixel must have to be
            kept; below this the noise diverges and the pixel is masked to
            ``NaN``. Defaults to 0.5 (the half-power beam edge). Lower it
            (e.g. 1e-2) to show a wider but noisier strip.

        Returns
        -------
        matplotlib.figure.Figure
            The figure holding the grid of panels.
        """
        if dec_deg is None:
            dec_deg = self.params['latitude']

        freq = np.atleast_1d(convert_freq_to_Hz(freqs, self.params))
        background_maps, nside, npix, dec_pix = self._gsm_maps(freq)
        T_sys = self.params['Tsys']

        # Single fixed pointing: keep only sky within strip_halfwidth degrees of
        # the pointing centre; everything else is unobservable for this strip.
        mask = np.abs(dec_pix - dec_deg) <= strip_halfwidth
        dec_obs = dec_pix[mask]
        # Angular offset of each pixel from the pointing centre, used by the
        # primary-beam attenuation D(phi) inside sigma_rms.
        phi_offset = np.radians(dec_obs - dec_deg)

        strip_maps = []
        for i, f in enumerate(freq):
            Tsky = background_maps[i]

            if type == "background_only":
                sky_map = Tsky.copy()
                sky_map[sky_map <= 0] = np.nan      # drop unphysical pixels
                sky_map[~mask] = np.nan             # hide sky outside the strip
                strip_maps.append(sky_map)
                continue

            # Total temperature entering the SEFD for each observed pixel.
            if type == "system_only":
                T_background = T_sys
            elif type == "total_noise":
                T_background = T_sys + Tsky[mask]
            else:
                raise ValueError(f"Unknown type {type!r}. Choose 'background_only', "
                                 "'system_only', or 'total_noise'.")

            # Per-pixel sensitivity at the fixed pointing: dec_obs sets the
            # drift-scan cos(dec) factor and phi_offset the beam attenuation.
            sigma_rms = self.telescope.sigma_rms(
                delta_nu=delta_nu,
                freq=f,
                phi_offset=phi_offset,
                T_background=T_background,
                dec_deg=dec_obs,
            )
            sigma_T = self.telescope.surface_temperature(sigma_rms=sigma_rms, freq=f)

            # Mask pixels the beam effectively cannot see. There D(phi) -> 0,
            # so tau_eff -> 0 and sigma_rms diverges; those points are not
            # observed and would otherwise swamp the map with ~1e12 values.
            beam = np.atleast_1d(self.telescope.D_phi(phi_offset, f))
            sigma_T = np.where(beam >= min_beam_response, sigma_T, np.nan)

            sky_map = np.full(npix, np.nan)
            sky_map[mask] = sigma_T
            strip_maps.append(sky_map)

        return self._plot_map_grid(
            strip_maps, freqs, title=title, save_path=save_path, unit=unit,
            freq_unit=freq_unit, cmap=cmap, vmin=vmin, vmax=vmax,
        )
            
