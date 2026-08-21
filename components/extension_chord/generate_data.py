#!/usr/bin/env python
"""Precompute CHORD extension-chord dirty beams for the interactive dashboard.

For every single extension location and every location pair (each combined with
the compact core) this computes the synthesized (dirty) beam under three
weightings -- natural, Briggs R=0, uniform -- plus an adaptive Briggs
robustness sweep.

Beams are computed on an ``NPIX x NPIX`` grid (``image_extent_deg=1`` for correct
FFT sampling) and only the inner ``SAVE_PIX x SAVE_PIX`` block is kept -- that
window holds the main lobe and near sidelobes. Beams are normalized to peak 1
(linear) and stored as int16 (value x BEAM_SCALE) to keep the embedded arrays
small. Set ``TEST = True`` to run only a small subset of locations. Outputs:

    extension_beams.npz    combos, l/m axes, beam_{natural,briggs0,uniform}
                           (ncombos, SAVE_PIX, SAVE_PIX) int16, per-combo
                           metrics, and per-combo Briggs sweep arrays
    extension_params.json  grid/quantization metadata + combo list

Run (needs the project env: pyuvdata, astropy, geopandas, ...):
    python components/extension_chord/generate_data.py
"""

import itertools
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:                 # run without installing the package
    sys.path.insert(0, _SRC)

from astropy.coordinates import EarthLocation
import astropy.units as u

from CHORD_Sensitivity.extension_chord import (
    utm_txt_to_earthlocations,
    combine_locations,
    setup_uvdata,
    compute_dirty_beam,
    adaptive_briggs_sweep,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_DIR = os.path.join(_HERE, "data")
NPZ = os.path.join(DATA_DIR, "extension_beams.npz")
PARAMS_JSON = os.path.join(DATA_DIR, "extension_params.json")

LAYOUTS = os.path.join(_ROOT, "geo_data", "layouts")
COMPACT_FILE = os.path.join(LAYOUTS, "shifted_SW_trimmed2.txt")
LOCATION_FILES = {
    "A":  "location_A.txt",
    "A2": "location_A_2.txt",
    "B":  "location_B.txt",
    "C":  "location_C.txt",
    "D":  "location_D.txt",
    "E":  "location_E.txt",
    "F":  "location_F.txt",
    "G":  "location_G.txt",
}

FREQ_HZ = 900e6
N_TIMES = 10             # UV coverage is geometry-driven; a few integrations suffice
NPIX = 512              # FFT grid side (accurate sampling)
SAVE_PIX = 60           # inner block kept (main lobe + near sidelobes)
IMAGE_EXTENT_DEG = 1    # >= 1 deg required for correct beam FFT sampling
BEAM_SCALE = 10000      # int16 quantization: stored = round(beam * BEAM_SCALE)

# When True, only process a small subset of locations (fast pipeline test).
TEST = False
TEST_LOCATIONS = ["B", "D"]   # -> combos: B, D, B_D

# (dashboard key, compute_dirty_beam weighting, robust)
WEIGHTINGS = [
    ("natural", "natural", None),
    ("briggs0", "briggs",  0),
    ("uniform", "uniform", None),
]


def get_drao_location():
    """DRAO array centre; fall back to fixed coords if the site registry is offline."""
    try:
        return EarthLocation.of_site("DRAO")
    except Exception:
        return EarthLocation.from_geodetic(
            lon=-119.6237 * u.deg, lat=49.3208 * u.deg, height=545.0 * u.m,
        )


def crop_slice(npix=NPIX, save_pix=SAVE_PIX):
    """Central `save_pix` slice of a `npix`-wide axis."""
    start = npix // 2 - save_pix // 2
    return slice(start, start + save_pix)


def quantize(beam):
    """Linear beam (0..1) -> int16 via BEAM_SCALE."""
    return np.clip(np.round(beam * BEAM_SCALE), 0, 32767).astype(np.int16)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    drao_loc = get_drao_location()

    names = TEST_LOCATIONS if TEST else list(LOCATION_FILES)   # canonical order for keys
    if TEST:
        print(f"*** TEST mode: only locations {names} ***", flush=True)

    compact = utm_txt_to_earthlocations(COMPACT_FILE)
    locations = {name: utm_txt_to_earthlocations(os.path.join(LAYOUTS, LOCATION_FILES[name]))
                 for name in names}

    # 1-location and 2-location combinations with the compact core.
    combos = (list(itertools.combinations(names, 1))
              + list(itertools.combinations(names, 2)))

    sl = crop_slice()
    keys = []
    beams = {w: [] for w, _, _ in WEIGHTINGS}
    eff_l = {w: [] for w, _, _ in WEIGHTINGS}
    eff_m = {w: [] for w, _, _ in WEIGHTINGS}
    noise = {w: [] for w, _, _ in WEIGHTINGS}
    sweeps = {}                              # per-combo Briggs sweep arrays
    l_axis = m_axis = None

    t_start = time.time()
    for n, combo in enumerate(combos, 1):
        key = "_".join(combo)                # "A" or "A_B"
        keys.append(key)
        locs = [locations[c] for c in combo]
        array = combine_locations(compact, *locs)
        uvd = setup_uvdata(array, drao_loc, n_times=N_TIMES, freq_hz=FREQ_HZ)
        print(f"[{n}/{len(combos)}] {key}: {array.shape[0]} dishes", flush=True)
        t0 = time.time()

        for w_key, weighting, robust in WEIGHTINGS:
            beam, l_arcmin, m_arcmin, _, _, el, em, nf = compute_dirty_beam(
                uvd, npix=NPIX, freq_hz=FREQ_HZ, image_extent_deg=IMAGE_EXTENT_DEG,
                weighting=weighting, robust=robust, plot_beam=False,
            )
            beams[w_key].append(quantize(beam[sl, sl]))
            eff_l[w_key].append(el)
            eff_m[w_key].append(em)
            noise[w_key].append(nf)
            if l_axis is None:               # identical for every combo/weighting
                l_axis = l_arcmin[sl].astype(np.float32)
                m_axis = m_arcmin[sl].astype(np.float32)

        r, nf_arr, sl_arr, sm_arr = adaptive_briggs_sweep(
            uvd, npix=NPIX, freq_hz=FREQ_HZ, image_extent_deg=IMAGE_EXTENT_DEG,
        )
        sweeps[key] = (r, nf_arr, sl_arr, sm_arr)
        print(f"    beams + {len(r)}-point sweep in {time.time() - t0:.1f}s", flush=True)

    # --- assemble and save ---
    save = {
        "combos": np.array(keys),
        "l_arcmin": l_axis,
        "m_arcmin": m_axis,
    }
    for w, _, _ in WEIGHTINGS:
        save[f"beam_{w}"] = np.stack(beams[w])                       # (ncombos,120,120) int16
        save[f"eff_l_{w}"] = np.asarray(eff_l[w], dtype=np.float32)
        save[f"eff_m_{w}"] = np.asarray(eff_m[w], dtype=np.float32)
        save[f"noise_{w}"] = np.asarray(noise[w], dtype=np.float32)
    for key, (r, nf_arr, sl_arr, sm_arr) in sweeps.items():
        save[f"sweep_r_{key}"] = r.astype(np.float32)
        save[f"sweep_nf_{key}"] = nf_arr.astype(np.float32)
        save[f"sweep_eff_l_{key}"] = sl_arr.astype(np.float32)
        save[f"sweep_eff_m_{key}"] = sm_arr.astype(np.float32)

    np.savez_compressed(NPZ, **save)
    print(f"Saved -> {NPZ}", flush=True)

    params = {
        "freq_hz": FREQ_HZ,
        "npix": NPIX,
        "save_pix": SAVE_PIX,
        "image_extent_deg": IMAGE_EXTENT_DEG,
        "beam_scale": BEAM_SCALE,
        "n_times": N_TIMES,
        "weightings": [w for w, _, _ in WEIGHTINGS],
        "locations": names,
        "combos": keys,
        "n_combos": len(keys),
        "beam_extent_arcmin": [float(l_axis[0]), float(l_axis[-1])],
    }
    with open(PARAMS_JSON, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved -> {PARAMS_JSON}")
    print(f"Done in {time.time() - t_start:.1f}s for {len(combos)} combos.")


if __name__ == "__main__":
    main()
