#!/usr/bin/env python
"""Precompute CHORD total-noise sensitivity sky maps for the interactive dashboard.

For each frequency the ``SkyMap.sky_noise`` HEALPix map (GSM Galactic
pixelisation, masked to CHORD's declination band) is resampled onto a regular
RA/Dec grid (Method A) so it can be shown as a Plotly heatmap. All frequencies
share one colour scale. Outputs:

    sky_noise_maps.npz     ra, dec, freqs, maps (nfreq, ndec, nra)  [mK]
    sky_noise_params.json  shared colour limits, delta_nu, units, dec band

Run from anywhere (needs the CHORD_Sensitivity package + healpy, i.e. the
project environment):  python components/sensitivity/generate_data.py
"""

import json
import os
import sys

import numpy as np
import healpy as hp

# --------------------------------------------------------------------------- #
# Paths and configuration
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC not in sys.path:                 # run without installing the package
    sys.path.insert(0, _SRC)

from CHORD_Sensitivity.sky_map import SkyMap

DATA_DIR = os.path.join(_HERE, "data")
MAPS_NPZ = os.path.join(DATA_DIR, "sky_noise_maps.npz")
PARAMS_JSON = os.path.join(DATA_DIR, "sky_noise_params.json")

FREQS_MHZ = [300, 500, 700, 900, 1100, 1300, 1500]  # frequencies to map
DELTA_NU = 5e3          # channel bandwidth [Hz] (5 kHz)
NOISE_TYPE = "total_noise"

GRID_DEG = 0.5          # RA/Dec grid spacing in degrees
UNIT = "mK"             # maps are stored in milli-kelvin
CBAR_PERCENTILES = (0.5, 99.0)  # shared colour limits across all maps


# --------------------------------------------------------------------------- #
# Resample a HEALPix map (Galactic) onto a regular RA/Dec grid
# --------------------------------------------------------------------------- #
def equatorial_grid(step_deg=GRID_DEG):
    """Return the RA and Dec axes (degrees) of the output grid."""
    ra = np.arange(0.0, 360.0 + step_deg, step_deg)
    dec = np.arange(-90.0, 90.0 + step_deg, step_deg)
    return ra, dec


def resample_to_radec(healpix_map, ra, dec):
    """Sample a Galactic HEALPix map on an equatorial RA/Dec grid.

    Each grid point (RA, Dec) is rotated to Galactic coordinates and the map is
    bilinearly interpolated there. Grid points whose HEALPix neighbours are all
    NaN (outside CHORD's dec band) come back NaN, preserving the mask.
    """
    RA, DEC = np.meshgrid(ra, dec)                  # (ndec, nra)
    theta_eq = np.radians(90.0 - DEC.ravel())       # colatitude
    phi_eq = np.radians(RA.ravel())
    theta_g, phi_g = hp.Rotator(coord=["C", "G"])(theta_eq, phi_eq)
    vals = hp.get_interp_val(healpix_map, theta_g, phi_g)
    return vals.reshape(DEC.shape)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    sm = SkyMap("CHORD")

    print(f"Computing {NOISE_TYPE} maps at {FREQS_MHZ} MHz (delta_nu={DELTA_NU:g} Hz)...")
    healpix_maps = sm.sky_noise(
        delta_nu=DELTA_NU, freq=FREQS_MHZ, type=NOISE_TYPE, CHORD_range=True
    )

    ra, dec = equatorial_grid()
    print(f"Resampling to a {dec.size}x{ra.size} RA/Dec grid...")
    maps = np.stack([
        resample_to_radec(m, ra, dec) * 1e3          # K -> mK
        for m in healpix_maps
    ])

    # Shared colour scale across every frequency (ignoring the NaN-masked sky).
    finite = maps[np.isfinite(maps)]
    vmin, vmax = np.percentile(finite, CBAR_PERCENTILES)
    print(f"Colour scale [{vmin:.3f}, {vmax:.3f}] {UNIT}")

    np.savez_compressed(
        MAPS_NPZ,
        ra=ra, dec=dec,
        freqs=np.asarray(FREQS_MHZ, dtype=float),
        maps=maps.astype(np.float32),
    )
    print(f"Saved -> {MAPS_NPZ}  (maps shape {maps.shape})")

    params = {
        "noise_type": NOISE_TYPE,
        "delta_nu_hz": DELTA_NU,
        "freqs_mhz": FREQS_MHZ,
        "unit": UNIT,
        "grid_deg": GRID_DEG,
        "colour_scale": {
            "vmin": float(vmin),
            "vmax": float(vmax),
            "percentiles": list(CBAR_PERCENTILES),
        },
        "dec_band_deg": [sm.params["min_dec"], sm.params["max_dec"]],
        "telescope": sm.name,
    }
    with open(PARAMS_JSON, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved -> {PARAMS_JSON}")


if __name__ == "__main__":
    main()
