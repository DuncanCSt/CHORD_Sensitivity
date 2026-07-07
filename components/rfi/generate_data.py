#!/usr/bin/env python
"""Generate per-line RFI visibility scores for the CHORD hydrogen recombination lines.

Distilled from notebooks/rfi/rfi_analysis_2026_data.ipynb. The pipeline:

    1. Load the RFI-monitor spectrogram as a read-only memmap.
    2. Split it into time segments (integration cycles), dropping calibration cycles.
    3. Compute the Spectral Kurtosis (SK) of each segment over frequency.
    4. Derive clean/RFI SK thresholds from a simulated gamma distribution.
    5. Combine the per-segment RFI masks into one label (clean if >=50% of
       segments agree the channel is clean).
    6. Score each H line by the Gaussian-weighted (100 km/s sigma) fraction of
       clean bins under the line.
    7. Save the annotated H-line table to CSV.

Run from anywhere:  python components/rfi/generate_data.py
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

DATA_FILE = os.path.join(_ROOT, "rfi_data", "RFInd-calibrated-2026-05-06T01.sigmf-data")
TRANSITIONS_FILE = os.path.join(_ROOT, "rfi_data", "transitions_parsed.zip")
DATA_DIR = os.path.join(_HERE, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "H_lines_with_clean_scores.csv")
TRACES_CSV = os.path.join(DATA_DIR, "rfi_spectrum_traces.csv")
PARAMS_JSON = os.path.join(DATA_DIR, "rfi_params.json")

CLEAN_PERCENTILE = 0.1   # SK gamma percentile used for the clean/RFI cut (and 100 - this)
SEGMENT_AGREE_FRAC = 0.5  # channel is clean if >this fraction of segments agree

N_TRACES = 10            # number of random raw time traces to save
V_KMS_RESAMPLE = 50.0    # velocity resolution of the saved spectrum grid

N_CHANNELS = 600000       # frequency channels per row in the raw file
F_MAX = 2000              # MHz, full band of the RFI monitor
CHORD_MIN, CHORD_MAX = 300, 1500  # MHz, CHORD band

REF_DB = -173            # dB reference used to stabilise the moment sums
CAL_CUTOFF_DB = -174     # rows below this (at 1420 MHz) are calibration cycles
SK_N = 2500 * 15 / 16    # accumulation length N for the SK estimator
BLOCK = 256              # time-block size for the streaming moment pass

V_KMS = 100.0            # radial-velocity width used for the line Gaussian
C_KMS = 299792.458
N_SIGMA = 3             # integrate the line Gaussian out to +/- N_SIGMA
CLEAN_THRESHOLD = 0.99   # visibility score above which a line is "clean"

R_H_C = 3.288051e15      # Hz, reduced-mass Rydberg frequency for hydrogen


# --------------------------------------------------------------------------- #
# 1. Load the data as a memmap
# --------------------------------------------------------------------------- #
def load_data(file):
    """Memory-map the raw float32 spectrogram as (n_time, N_CHANNELS)."""
    data = np.memmap(file, dtype="f4", mode="r")
    return data.reshape(-1, N_CHANNELS)


# --------------------------------------------------------------------------- #
# 2. Split into segments following the notebook's processing
# --------------------------------------------------------------------------- #
def segment_data(data):
    """Split into integration-cycle segments and drop calibration cycles.

    Segment boundaries are the large jumps in the 1420 MHz power trace; within
    each segment, individual low-power (calibration) rows are removed, and the
    incomplete first/last segments are discarded.
    """
    idx_1420 = int(1420 / F_MAX * data.shape[1])

    # Column restricting each row to the 300-1500 MHz CHORD band.
    chord_range = np.zeros(data.shape[1], dtype=bool)
    chord_range[int(CHORD_MIN / F_MAX * data.shape[1]):
                int(CHORD_MAX / F_MAX * data.shape[1])] = True

    time_trace = data[:, 100]
    diffs = np.diff(time_trace)
    starts = np.where(diffs < -1e5)[0] + 1
    ends = np.where(diffs > 1e5)[0] + 1
    starts = np.insert(starts, 0, 0)
    ends = np.append(ends, data.shape[0])

    segments = []
    for start, end in zip(starts, ends):
        data_1420 = data[start:end, idx_1420]
        segment = data[start:end, chord_range][data_1420 > CAL_CUTOFF_DB]
        print(f"Segment {start}-{end}: {segment.shape[0]} rows after filtering")
        segments.append(segment)

    # Drop the incomplete first and last segments.
    return segments[1:12]


# --------------------------------------------------------------------------- #
# 3. Spectral kurtosis on each segment
# --------------------------------------------------------------------------- #
def block_moments(data, ref_db, block=BLOCK):
    """One streaming pass: raw moments of x (dB) and of P (linear power).

    Blocks over the (small) time axis so peak RAM is ~block x n_freq rather than
    the whole array.
    """
    n_time, n_freq = data.shape
    S1 = np.zeros(n_freq)
    S2 = np.zeros(n_freq)
    M = 0

    ln10_10 = np.float32(np.log(10) / 10)

    for i in range(0, n_time, block):
        sl = slice(i, min(i + block, n_time))
        x = np.asarray(data[sl], dtype=np.float32)
        xc = x - np.float32(ref_db)              # shift for numerical stability

        P = np.exp(xc * ln10_10)                 # == 10**(xc/10), linear power
        S1 += P.sum(0, dtype=np.float64)
        S2 += (P * P).sum(0, dtype=np.float64)
        M += x.shape[0]

    return M, S1, S2


def spectral_kurtosis(M, S1, S2, N, d):
    """Generalised Spectral Kurtosis estimator from the power-sum moments."""
    return (N * M * d + 1) / (M - 1) * (M * S2 / S1**2 - 1)


def chord_freq_to_index(freq_mhz, n_bins):
    """Map a frequency (MHz) to a bin index within the CHORD band."""
    return int((freq_mhz - CHORD_MIN) / (CHORD_MAX - CHORD_MIN) * n_bins)


def velocity_resampled_indices(n_bins, v_kms):
    """Bin indices spaced by a constant radial velocity v_kms across the band.

    A fixed velocity step means a fixed *fractional* frequency step
    (Delta f / f = v / c), so the target grid is geometric and the sampling
    density falls with frequency. Each target frequency is snapped to its
    nearest existing bin (no interpolation); duplicate snaps are dropped.
    """
    freqs = np.linspace(CHORD_MIN, CHORD_MAX, n_bins)
    ratio = 1.0 + v_kms / C_KMS

    targets = []
    f = freqs[0]
    while f <= freqs[-1]:
        targets.append(f)
        f *= ratio
    targets = np.array(targets)

    # Nearest existing bin for each target (uniform grid -> closed form).
    idx = np.round((targets - CHORD_MIN) / (CHORD_MAX - CHORD_MIN) * (n_bins - 1))
    idx = np.clip(idx, 0, n_bins - 1).astype(int)
    return np.unique(idx)


def compute_segment_sk(segments):
    """Compute calibrated SK per segment.

    First pass with d=1 gives the empirical SK, whose mode over a known-clean
    band (1400-1430 MHz) sets the calibration factor d = 1/mode; the calibrated
    SK is then formed from the same cached moments.
    """
    n_bins = segments[0].shape[1]
    moments = [block_moments(seg, REF_DB) for seg in segments]

    # Uncalibrated SK (d = 1) to find the empirical mode.
    SK = np.array([spectral_kurtosis(M, S1, S2, SK_N, 1.0) for M, S1, S2 in moments])

    lo = chord_freq_to_index(1400, n_bins)
    hi = chord_freq_to_index(1430, n_bins)
    clean_sk = np.clip(SK[:, lo:hi].flatten(), 0, 2)
    grid = np.linspace(clean_sk.min(), clean_sk.max(), 2000)
    mode_sk = grid[np.argmax(gaussian_kde(clean_sk)(grid))]
    print(f"Empirical SK mode (clean band): {mode_sk:.4f} -> d = {1 / mode_sk:.4f}")

    d = 1.0 / mode_sk
    SK_cal = np.array([spectral_kurtosis(M, S1, S2, SK_N, d) for M, S1, S2 in moments])
    return SK_cal, SK_N * d, mode_sk


# --------------------------------------------------------------------------- #
# 4. SK thresholds from simulated data
# --------------------------------------------------------------------------- #
def simulate_sk_thresholds(Nd, n_time_samples, n_trials=100000, seed=None):
    """0.1 / 99.9 percentile SK bounds from a simulated gamma population.

    Each simulated segment is the SK of ``n_time_samples`` Gamma(Nd, 1) draws,
    matching the number of time integrations (M) in a real segment; the SK
    estimator applied to those draws gives the null distribution used for the
    clean/RFI cut.
    """
    rng = np.random.default_rng(seed)
    gamma_samples = rng.gamma(shape=Nd, scale=1.0, size=(n_time_samples, n_trials))
    M_g = gamma_samples.shape[0]
    S1_g = gamma_samples.sum(axis=0)
    S2_g = np.square(gamma_samples).sum(axis=0)

    SK_gamma = ((Nd * M_g + 1.0) / (M_g - 1.0)) * (M_g * S2_g / S1_g**2 - 1.0)
    SK_lo, SK_hi = np.percentile(SK_gamma, [CLEAN_PERCENTILE, 100 - CLEAN_PERCENTILE])
    print(f"SK clean band: [{SK_lo:.4f}, {SK_hi:.4f}]")
    return SK_lo, SK_hi


# --------------------------------------------------------------------------- #
# 5. Combine segments into a single RFI label
# --------------------------------------------------------------------------- #
def combine_clean_mask(SK_cal, SK_lo, SK_hi):
    """Per-channel clean label: clean where >SEGMENT_AGREE_FRAC of segments agree."""
    clean_spectrum = (SK_cal > SK_lo) & (SK_cal < SK_hi)
    return (clean_spectrum.mean(axis=0) > SEGMENT_AGREE_FRAC).astype(float)


# --------------------------------------------------------------------------- #
# 6. Per-line label via a Gaussian average (100 km/s sigma)
# --------------------------------------------------------------------------- #
def load_h_lines():
    """H recombination lines within the CHORD band, plus manual Hn-alpha lines."""
    transitions = pd.read_csv(TRANSITIONS_FILE)
    H_lines = transitions[
        (transitions.Species == "H")
        & transitions["Frequency (MHz)"].between(CHORD_MIN, CHORD_MAX)
    ].sort_values("Frequency (MHz)", ascending=False).reset_index(drop=True)

    def H_alpha(n):
        return R_H_C * (1 / n**2 - 1 / (n + 1)**2) / 1e6

    for n in range(254, 280):
        H_lines.loc[len(H_lines)] = [f"H{n}alpha", "H", f"H{n}", H_alpha(n)]

    # Gaussian width from a 100 km/s radial velocity (non-relativistic Doppler).
    H_lines["delta_f_MHz_100kms"] = H_lines["Frequency (MHz)"] * (V_KMS / C_KMS)
    return H_lines


def line_clean_score(f0, sigma_mhz, freqs, clean_spectrum, n_sigma=N_SIGMA):
    """Gaussian-weighted fraction of clean bins around a line center.

    Returns a value in [0, 1]: 1.0 = fully clean within the line width, 0.0 =
    fully flagged. NaN if the line falls outside the covered band.
    """
    lo = np.searchsorted(freqs, f0 - n_sigma * sigma_mhz)
    hi = np.searchsorted(freqs, f0 + n_sigma * sigma_mhz)
    if hi <= lo:
        return np.nan
    w = np.exp(-0.5 * ((freqs[lo:hi] - f0) / sigma_mhz) ** 2)
    return np.sum(w * clean_spectrum[lo:hi]) / np.sum(w)


def score_lines(H_lines, combined_clean_spectrum):
    """Attach the Gaussian-weighted clean score (and boolean label) to each line."""
    freqs = np.linspace(CHORD_MIN, CHORD_MAX, combined_clean_spectrum.size)
    H_lines["clean_score"] = [
        line_clean_score(f0, sigma, freqs, combined_clean_spectrum)
        for f0, sigma in zip(H_lines["Frequency (MHz)"],
                             H_lines["delta_f_MHz_100kms"])
    ]
    H_lines["clean"] = H_lines["clean_score"] >= CLEAN_THRESHOLD
    return H_lines


# --------------------------------------------------------------------------- #
# Velocity-resampled spectrum: frequency, random traces, calibrated SK
# --------------------------------------------------------------------------- #
def save_spectrum_traces(segments, SK_cal, combined_clean_spectrum, output_path,
                         n_traces=N_TRACES, v_kms=V_KMS_RESAMPLE, seed=None):
    """Save a 50 km/s-resampled spectrum: frequency, N raw traces, SK, clean label.

    The per-segment calibrated SK is aggregated (median over segments) to one
    value per channel. ``n_traces`` random time rows are drawn from the segments,
    and the traces, SK and combined clean label are sampled at the
    velocity-resampled bins.
    """
    n_bins = SK_cal.shape[1]
    freqs = np.linspace(CHORD_MIN, CHORD_MAX, n_bins)
    idx = velocity_resampled_indices(n_bins, v_kms)

    sk_median = np.median(SK_cal, axis=0)

    rng = np.random.default_rng(seed)
    columns = {"Frequency (MHz)": freqs[idx]}
    for i in range(n_traces):
        s = int(rng.integers(len(segments)))
        r = int(rng.integers(segments[s].shape[0]))
        row = np.asarray(segments[s][r], dtype=np.float64)   # full-band row
        columns[f"trace_{i}"] = row[idx]
    columns["SK_cal"] = sk_median[idx]
    columns["combined_clean"] = combined_clean_spectrum[idx]

    pd.DataFrame(columns).to_csv(output_path, index=False)
    print(f"Saved {len(idx)} resampled channels x {n_traces} traces -> {output_path}")


# --------------------------------------------------------------------------- #
# Save run parameters and summary stats
# --------------------------------------------------------------------------- #
def save_params(output_path, *, n_segments, mode_sk, Nd, SK_lo, SK_hi,
                combined_clean_spectrum, H_lines):
    """Write the calibrated thresholds, cutoffs and summary stats to JSON."""
    d = 1.0 / mode_sk
    n_clean = int(H_lines["clean"].sum())
    params = {
        "sk_estimator": {
            "N": SK_N,
            "empirical_mode": float(mode_sk),   # SK mode over the clean 1400-1430 MHz band
            "d": float(d),                      # calibration factor 1/mode
            "Nd": float(Nd),                    # gamma shape used for the null model
            "ref_db": REF_DB,
        },
        "rfi_thresholds": {                     # calibrated clean/RFI SK cut points
            "SK_low": float(SK_lo),
            "SK_high": float(SK_hi),
            "gamma_percentile": CLEAN_PERCENTILE,
            "gamma_percentile_upper": 100 - CLEAN_PERCENTILE,
        },
        "cutoffs": {
            "segment_agreement_fraction": SEGMENT_AGREE_FRAC,   # >this -> channel clean
            "line_clean_threshold": CLEAN_THRESHOLD,            # score >=this -> line clean
        },
        "line_scoring": {
            "velocity_width_kms": V_KMS,        # Gaussian sigma width (100 km/s)
            "n_sigma": N_SIGMA,                 # integration half-window in sigmas
        },
        "resampling": {
            "velocity_resolution_kms": V_KMS_RESAMPLE,
            "n_traces": N_TRACES,
        },
        "band": {
            "chord_min_mhz": CHORD_MIN,
            "chord_max_mhz": CHORD_MAX,
            "n_channels": int(combined_clean_spectrum.size),
        },
        "summary": {
            "n_segments": int(n_segments),
            "cal_cutoff_db": CAL_CUTOFF_DB,
            "clean_channel_fraction": float(combined_clean_spectrum.mean()),
            "n_lines": int(len(H_lines)),
            "n_clean_lines": n_clean,
            "clean_line_fraction": float(n_clean / len(H_lines)),
        },
    }
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved -> {output_path}")


# --------------------------------------------------------------------------- #
# 7. Driver
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Loading data...")
    data = load_data(DATA_FILE)

    print("Segmenting data...")
    segments = segment_data(data)

    print("Computing spectral kurtosis per segment...")
    SK_cal, Nd, mode_sk = compute_segment_sk(segments)

    print("Simulating SK thresholds...")
    # M = time integrations per segment; segments vary slightly, use the median.
    n_time_samples = int(np.median([seg.shape[0] for seg in segments]))
    SK_lo, SK_hi = simulate_sk_thresholds(Nd, n_time_samples=n_time_samples)

    print("Combining segments into a single clean mask...")
    combined_clean_spectrum = combine_clean_mask(SK_cal, SK_lo, SK_hi)
    print(f"Fraction of clean channels: {combined_clean_spectrum.mean():.3f}")

    print("Scoring H lines...")
    H_lines = load_h_lines()
    H_lines = score_lines(H_lines, combined_clean_spectrum)
    print(f"Clean lines: {int(H_lines['clean'].sum())} / {len(H_lines)}")

    H_lines.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved -> {OUTPUT_CSV}")

    print("Saving velocity-resampled spectrum traces...")
    save_spectrum_traces(segments, SK_cal, combined_clean_spectrum, TRACES_CSV)

    print("Saving run parameters...")
    save_params(
        PARAMS_JSON,
        n_segments=SK_cal.shape[0],
        mode_sk=mode_sk,
        Nd=Nd,
        SK_lo=SK_lo,
        SK_hi=SK_hi,
        combined_clean_spectrum=combined_clean_spectrum,
        H_lines=H_lines,
    )


if __name__ == "__main__":
    main()
