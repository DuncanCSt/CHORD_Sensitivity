"""
Utility functions for CHORD UV coverage and dirty beam analysis.

Provides helpers for:
- Loading antenna positions from UTM coordinate files
- Constructing pyuvdata UVData objects for interferometric simulations
- Plotting antenna layouts, UV coverage, and synthesized beams (dirty beams)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import geopandas as gpd
from pyproj import Transformer
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.modeling import models, fitting
import astropy.units as u
from pyuvdata import UVData, Telescope

FREQ_HZ = 900e6
# ---------------------------------------------------------------------------
# Antenna position I/O
# ---------------------------------------------------------------------------

def utm_txt_to_earthlocations(filename, height_default=545.0):
    """Convert a two-column UTM (Zone 11N) text file to EarthLocations.

    Parameters
    ----------
    filename : str
        Path to a whitespace-delimited file with columns (easting, northing).
    height_default : float
        Elevation in metres applied to every antenna (default: 545 m, DRAO).

    Returns
    -------
    EarthLocation
        Vectorised EarthLocation array in WGS84.
    """
    data = np.loadtxt(filename)
    east, north = data[:, 0], data[:, 1]
    heights = np.full_like(east, height_default)

    transformer = Transformer.from_crs("EPSG:26911", "EPSG:4326", always_xy=True)
    lon_deg, lat_deg = transformer.transform(east, north)

    locations = EarthLocation.from_geodetic(
        lon=lon_deg * u.deg,
        lat=lat_deg * u.deg,
        height=heights * u.m,
        ellipsoid="WGS84",
    )
    print(f"Loaded {len(locations)} antennas from '{filename}'")
    return locations


def combine_locations(*loc_arrays):
    """Concatenate multiple EarthLocation arrays into one."""
    return EarthLocation.from_geocentric(
        x=np.concatenate([loc.x.to_value(u.m) for loc in loc_arrays]) * u.m,
        y=np.concatenate([loc.y.to_value(u.m) for loc in loc_arrays]) * u.m,
        z=np.concatenate([loc.z.to_value(u.m) for loc in loc_arrays]) * u.m,
    )


# ---------------------------------------------------------------------------
# UVData construction
# ---------------------------------------------------------------------------

def setup_uvdata(ant_pos_geo, telescope_loc, n_times=300,
                 obs_duration_hours=1.0, freq_hz=900e6):
    """Build a UVData object for a drift-scan observation.

    Parameters
    ----------
    ant_pos_geo : EarthLocation
        Geocentric antenna positions.
    telescope_loc : EarthLocation
        Array centre location.
    n_times : int
        Number of time integrations.
    obs_duration_hours : float
        Total observing duration in hours.
    freq_hz : float
        Centre frequency in Hz.

    Returns
    -------
    UVData
    """
    Nants = ant_pos_geo.shape[0]

    ant_pos_ecef = np.column_stack((
        ant_pos_geo.x.to(u.m).value,
        ant_pos_geo.y.to(u.m).value,
        ant_pos_geo.z.to(u.m).value,
    ))

    telescope = Telescope.new(
        name="CHORD",
        location=telescope_loc,
        antenna_positions=ant_pos_ecef,
        antenna_names=[f"ANT{i:03d}" for i in range(Nants)],
        antenna_numbers=np.arange(Nants, dtype=int),
        x_orientation="east",
        update_from_known=False,
        instrument="interferometer",
    )

    start_time = Time("2025-03-01 12:00:00")
    times = start_time + np.linspace(
        -obs_duration_hours / 2, obs_duration_hours / 2, n_times
    ) * u.hour

    ant1, ant2 = np.triu_indices(Nants, k=0)
    antpairs = list(zip(ant1, ant2))
    Nblts = len(antpairs) * n_times

    phase_center_catalog = {
        0: {
            "cat_name": "zenith",
            "cat_type": "driftscan",
            "cat_lon": 0.0,
            "cat_lat": np.pi / 2,
            "cat_frame": "altaz",
        }
    }

    uvd = UVData.new(
        telescope=telescope,
        times=times.jd,
        freq_array=np.array([freq_hz]),
        polarization_array=np.array(["xx", "yy"]),
        integration_time=10.0,
        channel_width=np.array([1e6]),
        antpairs=antpairs,
        do_blt_outer=True,
        empty=True,
        history="CHORD array design study",
        phase_center_catalog=phase_center_catalog,
        phase_center_id_array=np.zeros(Nblts, dtype=int),
    )

    uvd.set_uvws_from_antenna_positions()
    uvd.check()
    return uvd


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_antenna_positions(locations, title=""):
    """Plot antenna positions on a UTM map."""
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(locations.lon.deg, locations.lat.deg),
        crs="EPSG:4326",
    ).to_crs("EPSG:26911")

    fig, ax = plt.subplots(figsize=(11, 9))
    gdf.plot(ax=ax, markersize=15, color="blue",
             label=f"Antennas (N={len(gdf)})", zorder=3)
    ax.set_title(f"{title} Antenna Positions")
    ax.set_xlabel("Easting (m) — NAD83 / UTM Zone 11N")
    ax.set_ylabel("Northing (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_uv_coverage(uvd, title="", freq_hz=900e6):
    """Plot UV coverage and print the maximum baseline / angular resolution.

    Returns
    -------
    tuple
        (fig, ax, resolution_arcsec)
    """
    u_km = uvd.uvw_array[:, 0] / 1e3
    v_km = uvd.uvw_array[:, 1] / 1e3

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(u_km, v_km, s=0.5, alpha=0.25, color="blue")
    ax.scatter(-u_km, -v_km, s=0.5, alpha=0.25, color="blue")
    ax.set_xlabel("u (km)")
    ax.set_ylabel("v (km)")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    lambda_m = 3e8 / freq_hz
    b_max = np.max(np.sqrt(uvd.uvw_array[:, 0] ** 2 + uvd.uvw_array[:, 1] ** 2))
    res_arcsec = np.degrees(lambda_m / b_max) * 3600
    print(f"{title}: max baseline = {b_max / 1e3:6.1f} km"
          f" -> resolution ~ {res_arcsec:5.1f} arcsec")
    return fig, ax, res_arcsec


# ---------------------------------------------------------------------------
# Dirty beam (synthesized PSF)
# ---------------------------------------------------------------------------

def _measure_fwhm_1d(profile, half_max=0.5):
    """Return FWHM in pixels from a 1-D beam slice."""
    peak = np.argmax(profile)

    left = peak
    while left > 0 and profile[left] >= half_max:
        left -= 1
    right = peak
    while right < len(profile) - 1 and profile[right] >= half_max:
        right += 1

    if left == peak or right == peak:
        return np.nan

    frac_left = ((half_max - profile[left])
                 / (profile[left + 1] - profile[left]))
    pos_left = left + frac_left

    frac_right = ((half_max - profile[right - 1])
                  / (profile[right] - profile[right - 1]))
    pos_right = right - 1 + frac_right

    return pos_right - pos_left


def eff_halfmax_extent(beam, x_coords, y_coords, level=0.5):
    """Max extent of the half-maximum region in x and y, treated independently.

    The beam is assumed normalised to a peak of unity. For every adjacent
    pixel pair that straddles ``level``, the crossing location is found by
    linear interpolation; the extent along each axis is the span between the
    outermost crossings (so multiple separate half-max islands are included).

    Returns (extent_x, extent_y) in the units of *x_coords* / *y_coords*.
    """
    b = beam - level

    # Crossings along x (between columns j and j+1) across every row.
    sx = b[:, :-1] * b[:, 1:] < 0
    if np.any(sx):
        rows, cols = np.nonzero(sx)
        b0, b1 = b[rows, cols], b[rows, cols + 1]
        frac = b0 / (b0 - b1)
        xc = x_coords[cols] + frac * (x_coords[cols + 1] - x_coords[cols])
        extent_x = xc.max() - xc.min()
    else:
        extent_x = np.nan

    # Crossings along y (between rows i and i+1) across every column.
    sy = b[:-1, :] * b[1:, :] < 0
    if np.any(sy):
        rows, cols = np.nonzero(sy)
        b0, b1 = b[rows, cols], b[rows + 1, cols]
        frac = b0 / (b0 - b1)
        yc = y_coords[rows] + frac * (y_coords[rows + 1] - y_coords[rows])
        extent_y = yc.max() - yc.min()
    else:
        extent_y = np.nan

    return extent_x, extent_y


def _apply_weighting(uv_density, v_idx, u_idx, valid,
                     weighting, robust, sum_w_natural, sum_w2_natural):
    """Grid visibilities with the requested weighting scheme.

    Returns (uv_grid, sum_w, sum_w2).
    """
    npix = uv_density.shape[0]
    uv_grid = np.zeros((npix, npix), dtype=float)

    if weighting == "natural":
        uv_grid = uv_density.copy()
        return uv_grid, sum_w_natural, sum_w2_natural

    if weighting == "uniform":
        if uv_density.max() == 0:
            return uv_density.copy(), sum_w_natural, sum_w2_natural
        w = 1.0 / (uv_density[v_idx[valid], u_idx[valid]] + 1e-12)
        np.add.at(uv_grid, (v_idx[valid], u_idx[valid]), w)
        return uv_grid, np.sum(uv_grid), np.sum(w ** 2)

    if weighting == "briggs":
        if uv_density.max() == 0:
            return uv_density.copy(), sum_w_natural, sum_w2_natural
        sum_Wk2 = np.sum(uv_density ** 2)
        f2 = (5.0 * 10 ** (-robust)) ** 2 * sum_w_natural / sum_Wk2
        Wk = uv_density[v_idx[valid], u_idx[valid]]
        w = 1.0 / (1.0 + Wk * f2 + 1e-12)
        np.add.at(uv_grid, (v_idx[valid], u_idx[valid]), w)
        return uv_grid, np.sum(uv_grid), np.sum(w ** 2)

    raise ValueError("weighting must be 'natural', 'uniform', or 'briggs'")


def compute_dirty_beam(uvd, title="", npix=128, freq_hz=900e6,
                       padding=10.0, image_extent_deg=None,
                       zoom_factor=1.5,
                       weighting="natural", robust=None,
                       zoom_extent=None, plot_beam=True):
    """Compute and plot the normalised dirty beam (synthesized PSF).

    Parameters
    ----------
    uvd : UVData
        Visibility data with populated uvw coordinates.
    title : str
        Plot title prefix.
    npix : int
        Number of pixels per side for the UV grid.
    freq_hz : float
        Observing frequency in Hz.
    padding : float
        Factor by which to extend the UV extent beyond the data.
        Ignored when *image_extent_deg* is set.
    image_extent_deg : float or None
        If given, set the image half-width to this many degrees and derive
        the UV grid extent automatically.  Overrides *padding*.
    zoom_factor : float
        Multiplier on the measured FWHM for the plot window.
    weighting : str
        ``'natural'``, ``'uniform'``, or ``'briggs'``.
    robust : float or None
        Briggs robustness parameter (only used when *weighting='briggs'*).
    zoom_extent : float or None
        If given, fix the plot window to +/-zoom_extent arcmin (overrides
        zoom_factor). Useful for consistent axes when comparing weightings.

    Returns
    -------
    beam : ndarray
        Normalised dirty beam (npix, npix).
    l_arcmin, m_arcmin : ndarray
        Coordinate axes in arcminutes.
    fwhm_l_arcmin, fwhm_m_arcmin : float
        Measured FWHM in arcminutes along l and m.
    """
    lambda_m = 3e8 / freq_hz
    u_lam = uvd.uvw_array[:, 0] / lambda_m
    v_lam = uvd.uvw_array[:, 1] / lambda_m
    u_lam = np.concatenate([u_lam, -u_lam])
    v_lam = np.concatenate([v_lam, -v_lam])
    
    # Grid extents
    if image_extent_deg is not None:
        # delta_theta = image_extent_deg / (npix/2)  (pixel size in radians)
        # uv_res_lam  = 1 / (npix * delta_theta_rad)
        # uv_extent   = npix/2 * uv_res_lam
        delta_theta_rad = np.radians(image_extent_deg) / (npix / 2)
        uv_res_lam = 1.0 / (npix * delta_theta_rad)
        uv_extent_lam = (npix / 2) * uv_res_lam
    else:
        bl_len = np.sqrt(uvd.uvw_array[:, 0] ** 2 + uvd.uvw_array[:, 1] ** 2)
        uv_extent_lam = (bl_len.max() * padding) / lambda_m
        uv_res_lam = 2 * uv_extent_lam / npix

    u_idx = np.round((u_lam + uv_extent_lam) / uv_res_lam).astype(int)
    v_idx = np.round((v_lam + uv_extent_lam) / uv_res_lam).astype(int)
    valid = (u_idx >= 0) & (u_idx < npix) & (v_idx >= 0) & (v_idx < npix)

    # Density grid (natural weights)
    uv_density = np.zeros((npix, npix), dtype=float)
    np.add.at(uv_density, (v_idx[valid], u_idx[valid]), 1.0)

    sum_w_natural = np.sum(uv_density)
    sum_w2_natural = sum_w_natural  # per-visibility: each w_i=1, so sum(w_i^2) = N_vis

    # Weighted grid
    uv_grid, sum_w, sum_w2 = _apply_weighting(
        uv_density, v_idx, u_idx, valid,
        weighting, robust, sum_w_natural, sum_w2_natural,
    )

    # Noise factor relative to natural weighting
    noise_nat = np.sqrt(sum_w2_natural) / sum_w_natural if sum_w_natural > 0 else 1.0
    noise_w = np.sqrt(sum_w2) / sum_w if sum_w > 0 else 1.0
    relative_noise = noise_w / noise_nat

    # FFT -> dirty beam
    if uv_grid.max() > 0:
        uv_grid /= uv_grid.max()
    beam = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid))))
    if beam.max() > 0:
        beam /= beam.max()

    # Angular coordinates
    delta_theta_arcmin = (1.0 / (2 * uv_extent_lam)) * (180 / np.pi * 60)
    l_arcmin = (np.arange(npix) - npix // 2) * delta_theta_arcmin
    m_arcmin = l_arcmin.copy()

    # # Effective resolution: fit a 2D Gaussian to the main lobe and extract the FWHM
    # y, x = np.mgrid[:beam.shape[0], :beam.shape[1]]
    # gaussian_init = models.Gaussian2D(amplitude=beam.max(), x_mean=beam.shape[1]//2, y_mean=beam.shape[0]//2, x_stddev=5, y_stddev=5)
    # fitter = fitting.LevMarLSQFitter()
    # gaussian_fit = fitter(gaussian_init, x, y, beam)
    # gauss_beam_fit = gaussian_fit(x, y)
    # peak_idx = np.unravel_index(np.argmax(gauss_beam_fit), gauss_beam_fit.shape)
    # eff_l_pix = _measure_fwhm_1d(gauss_beam_fit[peak_idx[0], :], half_max = 0.5) 
    # eff_m_pix = _measure_fwhm_1d(gauss_beam_fit[:, peak_idx[1]], half_max = 0.5)
    # eff_l_arcmin = eff_l_pix * delta_theta_arcmin if not np.isnan(eff_l_pix) else np.nan
    # eff_l_arcmin = eff_m_pix * delta_theta_arcmin if not np.isnan(eff_m_pix) else np.nan

    # # FWHM measurement (1-D slices through peak for the main lobe)
    # peak_idx = np.unravel_index(np.argmax(beam), beam.shape)
    # fwhm_l_pix = _measure_fwhm_1d(beam[peak_idx[0], :])
    # fwhm_m_pix = _measure_fwhm_1d(beam[:, peak_idx[1]])
    # fwhm_l_arcmin = fwhm_l_pix * delta_theta_arcmin if not np.isnan(fwhm_l_pix) else np.nan
    # fwhm_m_arcmin = fwhm_m_pix * delta_theta_arcmin if not np.isnan(fwhm_m_pix) else np.nan

    # Max FWHM: the beam is normalised to a peak of unity above, so find every
    # half-maximum crossing (sub-pixel, by interpolation) along l and m and take
    # the outermost span on each axis. l and m are treated independently.
    eff_l_arcmin, eff_m_arcmin = eff_halfmax_extent(beam, l_arcmin, m_arcmin)

    # Theoretical resolution
    max_u_lam = np.abs(u_lam).max()
    max_v_lam = np.abs(v_lam).max()
    theta_l = (1.0 / max_u_lam) * (180 / np.pi * 60) if max_u_lam > 0 else np.nan
    theta_m = (1.0 / max_v_lam) * (180 / np.pi * 60) if max_v_lam > 0 else np.nan

    # --- Plot ---
    if plot_beam:
        extent = [l_arcmin[0], l_arcmin[-1], m_arcmin[0], m_arcmin[-1]]
        fig = plt.figure(figsize=(8, 7))
        im = plt.imshow(beam, origin="lower", extent=extent, cmap="viridis")
        plt.colorbar(im, label="Normalised amplitude")
        plt.contour(beam, levels=[0.5], colors="red", linewidths=2.0,
                    linestyles="--", extent=extent)
        ellipse = Ellipse((0, 0), width=eff_l_arcmin, height=eff_m_arcmin,
                          edgecolor="cyan", facecolor="none",
                          linewidth=2.0, linestyle=":")
        plt.gca().add_patch(ellipse)
        plt.plot([], [], color="red", linestyle="--", linewidth=2,
                 label="FWHM contour (half-maximum)")
        plt.plot([], [], color="cyan", linestyle=":", linewidth=2,
                 label="Effective resolution contour (half-maximum)")
        plt.legend(loc="upper right")

        if zoom_extent is not None:
            zoom = zoom_extent
        else:
            zoom = zoom_factor * max(
                eff_l_arcmin if not np.isnan(eff_l_arcmin) else 10,
                eff_m_arcmin if not np.isnan(eff_m_arcmin) else 10,
            )
        plt.xlim(-zoom, zoom)
        plt.ylim(-zoom, zoom)
        plt.xlabel("l (arcmin)")
        plt.ylabel("m (arcmin)")
        plt.title(f"{title} ({weighting.capitalize()} Weighting)")

        info = (
            f"Theoretical: l ~ {theta_l:.2f}', m ~ {theta_m:.2f}'\n"
            f"Effective res: l = {eff_l_arcmin:.3f}', m = {eff_m_arcmin:.3f}'\n"
            f"Noise factor: {relative_noise:.3f}x)"
        )
        fig.text(0.45, 0.18, info, ha="center", va="center", fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                           alpha=0.85, edgecolor="gray"))
        plt.tight_layout()
        plt.show()

    return beam, l_arcmin, m_arcmin, eff_l_arcmin, eff_m_arcmin, relative_noise

def adaptive_briggs_sweep(dish_array, r_lo=-2, r_hi=2, tol=0.1,
                          max_iter=50, freq_hz=FREQ_HZ, npix=256, image_extent_deg=1):
    """Sample robust values adaptively so adjacent noise factors differ by <= tol."""

    def _eval(r):
        """Compute and cache noise factor + effective resolution for a robust value."""
        if r not in cache:
            _, _, _, eff_l, eff_m, nf = compute_dirty_beam(
                dish_array, npix=npix, freq_hz=freq_hz, image_extent_deg=image_extent_deg,
                weighting="briggs", robust=r, plot_beam=False,
            )
            cache[r] = (nf, eff_l, eff_m)
        return cache[r]

    cache = {}

    # Seed with endpoints
    _eval(r_lo)
    _eval(r_hi)

    # Intervals still to check: (left_R, right_R)
    intervals = [(r_lo, r_hi)]
    n_iter = 0

    while intervals and n_iter < max_iter:
        r_left, r_right = intervals.pop(0)
        eff_l_left = cache[r_left][1]
        eff_l_right = cache[r_right][1]

        eff_m_left = cache[r_left][2]
        eff_m_right = cache[r_right][2]

        eff_max = max(abs(eff_l_left - eff_l_right), abs(eff_m_left - eff_m_right))

        if eff_max > tol and (r_right - r_left) > 0.05:  # Also avoid infinite loop on very small intervals 
            r_mid = (r_left + r_right) / 2
            _eval(r_mid)
            # Queue both halves for further checking
            intervals.append((r_left, r_mid))
            intervals.append((r_mid, r_right))
            n_iter += 1
        # else: interval is fine, no further subdivision needed

    # Sort by robust value
    r_sorted = sorted(cache.keys())
    r_arr = np.array(r_sorted)
    nf_arr = np.array([cache[r][0] for r in r_sorted])
    eff_l_arr = np.array([cache[r][1] for r in r_sorted])
    eff_m_arr = np.array([cache[r][2] for r in r_sorted])

    print(f"Sampled {len(r_arr)} robust values in {n_iter} bisections, total iterations: {n_iter}")
    return r_arr, nf_arr, eff_l_arr, eff_m_arr

def plot_briggs_values(title, r_arr, nf_arr, eff_l_arr, eff_m_arr):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "tab:blue"
    ax1.plot(nf_arr, eff_l_arr, "o-", color=color1, alpha=0.7, label="Resolution E-W (arcmin)")

    # Annotate each Briggs robust value
    for nf_i, res_l_i, r_i in zip(nf_arr, eff_l_arr, r_arr):
        ax1.annotate(
            f"R={r_i:.2g}",
            (nf_i, res_l_i),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color=color1,
            alpha=0.9,
        )
    ax1.set_xlabel("Noise Factor (relative to natural weighting)")
    ax1.set_ylabel("Resolution (arcmin)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
 
    color2 = "tab:red"
    ax1.plot(nf_arr, eff_m_arr, "o-", color=color2, alpha=0.7, label="Resolution N-S (arcmin)")

    ax1.set_title(title or "Briggs Weighting Trade-off")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.88), ncol=2)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()