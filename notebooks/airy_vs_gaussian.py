"""Compare the Airy (uniform-illumination) primary beam against a FWHM-matched
Gaussian for a 6 m dish at 400 MHz and 1200 MHz."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

sol = 3e8  # speed of light [m/s], matches constants.C


def airy_beam(a, D, freq):
    # angle in deg, D in m, freq in MHz
    arad = a * np.pi / 180
    wavelength = sol * 1e-6 / freq
    ap = np.pi * D * np.sin(arad) / wavelength
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(
            a < 180,
            np.where(ap > 1e-30, (2 * jv(1, ap) / ap) ** 2, (jv(0, ap) - jv(2, ap)) ** 2),
            0,
        )
    return out

def gaussian_beam(a, D, freq):
    # FWHM-matched Gaussian, peak 1. theta_FWHM = 1.029 * lambda / D (in radians).
    wavelength = sol * 1e-6 / freq
    fwhm_deg = np.degrees(1.029 * wavelength / D)
    return np.exp(-4 * np.log(2) * (a / fwhm_deg) ** 2)


D = 6.0  # m
freqs = [400, 1200]  # MHz
a = np.linspace(-15, 15, 4001)  # deg

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for col, f in enumerate(freqs):
    airy = airy_beam(a, D, f)
    gauss = gaussian_beam(a, D, f)
    wavelength = sol * 1e-6 / f
    fwhm_deg = np.degrees(1.029 * wavelength / D)

    # linear
    ax = axes[0, col]
    ax.plot(a, airy, label="Airy", lw=1.8)
    ax.plot(a, gauss, "--", label="Gaussian (FWHM-matched)", lw=1.8)
    ax.axhline(0.5, color="grey", lw=0.7, ls=":")
    ax.set_title(f"{f} MHz   (FWHM = {fwhm_deg:.2f}$^\\circ$)")
    ax.set_ylabel("Normalized power")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # dB
    ax = axes[1, col]
    floor = 1e-4
    ax.plot(a, 10 * np.log10(np.maximum(airy, floor)), label="Airy", lw=1.8)
    ax.plot(a, 10 * np.log10(np.maximum(gauss, floor)), "--", label="Gaussian", lw=1.8)
    ax.axhline(-3, color="grey", lw=0.7, ls=":")  # half power
    ax.set_ylim(-40, 2)
    ax.set_xlabel("Off-axis angle [deg]")
    ax.set_ylabel("Power [dB]")
    ax.grid(alpha=0.3)

fig.suptitle("6 m dish primary beam: Airy vs FWHM-matched Gaussian", fontsize=13)
fig.tight_layout()
out = "thesis/plots/airy_vs_gaussian.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
