#!/usr/bin/env python
"""Build the interactive sky-noise dashboard from the precomputed maps.

Reads the RA/Dec grids written by ``generate_data.py`` (in ``data/``), builds a
Plotly heatmap per frequency behind a frequency slider (Method A), and writes
the standalone dashboard to ``docs/sensitivity/index.html``.

Run:  python components/sensitivity/generate_html.py
"""

import json
import os
import sys

import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------------------------------- #
# Paths and display config
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DATA_DIR = os.path.join(_HERE, "data")
DOCS_SENS = os.path.join(_ROOT, "docs", "sensitivity")

# component_nav.py sits in components/, one level up from this script, and is
# shared by all three dashboards.
sys.path.insert(0, os.path.dirname(_HERE))
from component_nav import nav_html  # noqa: E402

RA_CENTER = -120.0   # RA (deg) placed at the centre of the panel (240 deg)
COLORSCALE = "Viridis"


# --------------------------------------------------------------------------- #
# Load precomputed data
# --------------------------------------------------------------------------- #
def load_data():
    npz = np.load(os.path.join(DATA_DIR, "sky_noise_maps.npz"))
    ra, dec, freqs, maps = npz["ra"], npz["dec"], npz["freqs"], npz["maps"]
    with open(os.path.join(DATA_DIR, "sky_noise_params.json")) as f:
        params = json.load(f)
    return ra, dec, freqs, maps, params


def recenter_ra(ra, maps, center=RA_CENTER):
    """Re-order the RA columns so ``center`` sits at the middle of the panel.

    Each RA is wrapped into ``(center-180, center+180]`` and the columns are
    sorted, so the heatmap x-axis is monotonic and centred on ``center``. The
    unwrapped actual RA (0-360) is returned for hover labels.
    """
    # drop the duplicated 360-deg column if present
    if np.isclose(ra[0] % 360, ra[-1] % 360):
        ra, maps = ra[:-1], maps[..., :-1]

    disp = center + ((ra - center + 180) % 360) - 180   # wrapped display RA
    order = np.argsort(disp)
    ra_disp = disp[order]
    ra_actual = ra[order] % 360
    maps = maps[:, :, order]
    return ra_disp, ra_actual, maps


# --------------------------------------------------------------------------- #
# Build the figure
# --------------------------------------------------------------------------- #
def build_figure(ra, dec, freqs, maps, params):
    ra_disp, ra_actual, maps = recenter_ra(ra, maps)
    unit = params["unit"]
    vmin = params["colour_scale"]["vmin"]
    vmax = params["colour_scale"]["vmax"]
    dec_lo, dec_hi = params["dec_band_deg"]

    # Crop to CHORD's observable declination band: the rest of the sky is
    # NaN-masked and base64 encoding still spends full bytes on it, so cropping
    # removes those rows and shrinks the embedded arrays.
    band = (dec >= dec_lo) & (dec <= dec_hi)
    dec = dec[band]
    maps = maps[:, band, :]

    # per-cell actual RA for the hover label (same for every frequency).
    # int16 keeps the base64-encoded customdata small (RA is integer degrees).
    ra_grid = np.tile(np.round(ra_actual).astype(np.int16), (dec.size, 1))

    def title_for(f):
        return f"CHORD Brightness Temperature Sensitivity Map (mK); {f:.0f} MHz"

    fig = go.Figure()
    for i, f in enumerate(freqs):
        fig.add_trace(go.Heatmap(
            z=maps[i].astype(np.float32), x=ra_disp, y=dec,
            zmin=vmin, zmax=vmax, colorscale=COLORSCALE,
            colorbar=dict(title=f"σ<sub>T</sub> ({unit})"),
            customdata=ra_grid,
            hovertemplate="RA: %{customdata:.1f}°<br>"
                          "Dec: %{y:.1f}°<br>"
                          "σ<sub>T</sub>: %{z:.1f} " + unit + "<extra></extra>",
            name=f"{f:.0f} MHz",
            visible=(i == 0),
        ))

    # frequency slider: each step shows one heatmap and updates the title
    steps = []
    for i, f in enumerate(freqs):
        steps.append(dict(
            method="update", label=f"{f:.0f}",
            args=[{"visible": [j == i for j in range(len(freqs))]},
                  {"title.text": title_for(f)}],
        ))
    sliders = [dict(
        active=0, pad={"t": 60},
        currentvalue={"prefix": "Frequency: ", "suffix": " MHz"},
        steps=steps,
    )]

    fig.update_xaxes(
        title_text=f"Right Ascension (deg, centred on {(RA_CENTER) % 360:.0f}°)",
        range=[ra_disp.min(), ra_disp.max()],
    )
    fig.update_yaxes(
        title_text="Declination (deg)",
        range=[dec.min(), dec.max()],
    )
    fig.update_layout(
        height=620,
        autosize=True,          # width follows the grid cell (avoids overflow)
        title=title_for(freqs[0]),
        margin=dict(t=70, b=40, l=60, r=20),
        sliders=sliders,
    )
    return fig


# --------------------------------------------------------------------------- #
# Assemble the HTML page
# --------------------------------------------------------------------------- #
def build_page(fig, params):
    # Force a resize once the grid has its final width: Plotly measures the
    # container at creation (still too wide in the grid) and otherwise only
    # recomputes on an interaction, so the plot renders oversized until touched.
    resize_js = (
        "var gd = document.getElementById('sky-plot');"
        "function fit(){ if (window.Plotly) Plotly.Plots.resize(gd); }"
        "window.addEventListener('resize', fit);"
        "window.addEventListener('load', fit);"
        "requestAnimationFrame(fit);"
    )
    plot_div = fig.to_html(
        full_html=False, include_plotlyjs="cdn", div_id="sky-plot",
        config={"responsive": True}, post_script=resize_js,
    )

    unit = params["unit"]
    delta_nu = params["delta_nu_hz"]
    dec_lo, dec_hi = params["dec_band_deg"]
    vmin = params["colour_scale"]["vmin"]
    vmax = params["colour_scale"]["vmax"]
    freqs = ", ".join(f"{f:.0f}" for f in params["freqs_mhz"])

    description = f"""
<h2>Instructions</h2>
<p>Use the <em>Frequency</em> slider to step through observing frequency options. Drag a box to zoom into a region, double-click to reset, hovering reports the RA, Dec and
surface-brightness sensitivity of each pixel.</p>

<h2>Description</h2>
<p>Each pixel is brightness temperature sensitivity using a \\(\\Delta\\nu = {delta_nu*1e-3:.0f}\\,\\mathrm{{kHz}}\\) channel bandwidth and one observing day, displayed in mK.</p>
<p>The maps cover CHORD's observable declination band
\\({dec_lo:.0f}^\\circ\\!\\le\\!\\delta\\!\\le\\!{dec_hi:.0f}^\\circ\\), on a shared linear
colour scale (\\({vmin:.0f}\\!-\\!{vmax:.0f}\\,\\mathrm{{{unit}}}\\)).</p>

<h2>Methods</h2>
<h3>Point source sensitivity</h3>
<p>The point-source RMS noise follows the radiometer equation:</p>
$$\\sigma_{{\\mathrm{{RMS}}}}\\,[\\mathrm{{Jy/beam}}] =
\\frac{{2k\\,(T_{{\\mathrm{{ant}}}}+T_{{\\mathrm{{sky}}}})}}
{{A_e\\sqrt{{2\\,\\Delta\\nu\\,\\tau_{{\\mathrm{{eff}}}}\\,N(N-1)}}}}$$
<p>where \\(k\\) is the Boltzmann constant, \\(T_{{\\mathrm{{ant}}}} = 30\\,\\mathrm{{K}}\\) is the
antenna temperature, \\(T_{{\\mathrm{{sky}}}}\\) is the radio background obtained per location from
the 2016 Global Sky Model, \\(A_e\\) is the per-dish collecting area,
\\(\\Delta\\nu = {delta_nu*1e-3:.0f}\\,\\mathrm{{kHz}}\\) is the per-channel bandwidth,
\\(\\tau_{{\\mathrm{{eff}}}}\\) is the effective per-day integration time (which increases towards
high declination), and \\(N = 512\\) is the number of dishes.</p>

<h3>Brightness temperature sensitivity</h3>
<p>The brightness temperature sensitivity is obtained by dividing the point-source RMS by the
synthesised-beam solid angle and applying the Rayleigh&ndash;Jeans limit:</p>
$$\\sigma_T\\,[\\mathrm{{K}}] = \\frac{{\\sigma_{{\\mathrm{{RMS}}}}}}{{\\Omega_s}}\\,
\\frac{{c^2}}{{2k\\nu^2}}$$
<p>where \\(\\Omega_s\\) is the synthesised-beam solid angle, \\(c\\) is the speed of light, and
\\(\\nu\\) is the observing frequency.</p>

<h3>Plotting maps yourself</h3>
<p>These maps are reproducible with:</p>
<pre><code>from CHORD_Sensitivity.sky_map import SkyMap

chord = SkyMap("CHORD")
chord.sky_noise(
    delta_nu=5e3, freq=500, type='total_noise', CHORD_range=True
)</code></pre>
"""

    nav = nav_html(active="sensitivity")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CHORD Sensitivity Maps</title>
  <link rel="stylesheet" href="../assets/style.css">
  <script>
    window.MathJax = {{ tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['$$', '$$']] }} }};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
</head>
<body>
  <header>
    <p class="crumb"><a href="../index.html">&larr; Return to presentation</a></p>
    <h1>CHORD Sensitivity</h1>
    <p>Interactive dashboards from the thesis 'Radio recombination line forecasts with CHORD'.</p>
  </header>
{nav}
  <section class="page-intro">
    <h2>CHORD Brightness Temperature Sensitivity Maps</h2>
    <p>CHORD total brightness temperature sensitivity maps for observable declinations {dec_lo:.0f}&deg; to {dec_hi:.0f}&deg;, by frequency.</p>
  </section>
  <div class="layout">
    <div class="plot">{plot_div}</div>
    <aside class="desc">{description}</aside>
  </div>
</body>
</html>
"""
    return page


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ra, dec, freqs, maps, params = load_data()
    fig = build_figure(ra, dec, freqs, maps, params)
    page = build_page(fig, params)

    os.makedirs(DOCS_SENS, exist_ok=True)
    out = os.path.join(DOCS_SENS, "index.html")
    with open(out, "w") as f:
        f.write(page)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
