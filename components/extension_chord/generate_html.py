#!/usr/bin/env python
"""Build the interactive extension-chord dashboard from the precomputed beams.

Reads ``data/extension_beams.npz`` (written by ``generate_data.py``) and builds a
2x2 Plotly figure -- natural / Briggs-0 / uniform dirty beams + the Briggs
sweep -- with every location combination embedded as (hidden) traces. Two HTML
dropdowns (location 1, location 2) drive which combination is shown. Writes the
standalone dashboard to ``docs/extension_chord/index.html``.

Run:  python components/extension_chord/generate_html.py
"""

import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Paths / config
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DATA_DIR = os.path.join(_HERE, "data")
DOCS_EXT = os.path.join(_ROOT, "docs", "extension_chord")

# component_nav.py sits in components/, one level up from this script, and is
# shared by all three dashboards.
sys.path.insert(0, os.path.dirname(_HERE))
from component_nav import nav_html  # noqa: E402

COLORSCALE = "Viridis"
DEFAULT_COMBO = "B_D"          # combination shown on load
PER = 5                        # traces per combo: 3 beams + 2 sweep lines
C_EW, C_NS = "#1f77b4", "#d62728"


def load_data():
    d = np.load(os.path.join(DATA_DIR, "extension_beams.npz"))
    with open(os.path.join(DATA_DIR, "extension_params.json")) as f:
        params = json.load(f)
    return d, params


# --------------------------------------------------------------------------- #
# Build the 2x2 figure with every combo embedded
# --------------------------------------------------------------------------- #
def build_figure(d, params):
    combos = [str(c) for c in d["combos"]]
    weightings = params["weightings"]          # [natural, briggs0, uniform]
    scale = params["beam_scale"]
    l = d["l_arcmin"]
    m = d["m_arcmin"]

    titles = {"natural": "Natural", "briggs0": "Briggs R=0", "uniform": "Uniform"}
    fig = make_subplots(
        rows=2, cols=2, vertical_spacing=0.11, horizontal_spacing=0.12,
        subplot_titles=(titles[weightings[0]], titles[weightings[1]],
                        titles[weightings[2]], "Briggs sweep"),
    )
    # subplot (row,col) for each of the 5 traces of a combo
    beam_rc = [(1, 1), (1, 2), (2, 1)]         # natural, briggs0, uniform

    for i, key in enumerate(combos):
        vis = (key == DEFAULT_COMBO)
        for w, (r_, c_) in zip(weightings, beam_rc):
            beam = d[f"beam_{w}"][i].astype(np.float32) / scale
            fig.add_trace(go.Heatmap(
                z=beam, x=l, y=m, coloraxis="coloraxis", visible=vis,
                name=f"{key}:{w}",
                hovertemplate="l=%{x:.2f}'<br>m=%{y:.2f}'<br>amp=%{z:.3f}<extra></extra>",
            ), row=r_, col=c_)

        # Briggs sweep: two lines (E-W, N-S) vs noise factor, R in customdata
        r_arr = d[f"sweep_r_{key}"]
        nf = d[f"sweep_nf_{key}"]
        for arr, colour, label in ((d[f"sweep_eff_l_{key}"], C_EW, "E-W"),
                                   (d[f"sweep_eff_m_{key}"], C_NS, "N-S")):
            order = np.argsort(nf)
            fig.add_trace(go.Scatter(
                x=nf[order], y=arr[order], customdata=r_arr[order],
                mode="lines+markers", line=dict(color=colour),
                marker=dict(size=5), visible=vis, showlegend=False,
                name=f"{key}:{label}",
                hovertemplate=(f"{label}<br>R=%{{customdata:.2g}}<br>"
                               "noise=%{x:.2f}<br>res=%{y:.2f}'<extra></extra>"),
            ), row=2, col=2)

    # shared colour axis for the three beams
    fig.update_layout(coloraxis=dict(
        colorscale=COLORSCALE, cmin=0, cmax=1,
        colorbar=dict(title="Norm.<br>amplitude", len=0.45, y=0.78, x=1.0),
    ))

    # beam axes: equal aspect so beams stay circular
    for ax in ("yaxis", "yaxis2", "yaxis3"):
        anchor = {"yaxis": "x", "yaxis2": "x2", "yaxis3": "x3"}[ax]
        fig.update_layout(**{ax: dict(scaleanchor=anchor, title_text="m (arcmin)")})
    for ax in ("xaxis", "xaxis2", "xaxis3"):
        fig.update_layout(**{ax: dict(title_text="l (arcmin)")})
    fig.update_xaxes(title_text="Noise factor (vs natural)", row=2, col=2)
    fig.update_yaxes(title_text="Resolution (arcmin)", row=2, col=2)

    # colour key for the sweep lines
    fig.add_annotation(text=f"<span style='color:{C_EW}'>&#9679; E-W</span>&nbsp;&nbsp;"
                            f"<span style='color:{C_NS}'>&#9679; N-S</span>",
                       xref="paper", yref="paper", x=0.99, y=0.0,
                       xanchor="right", yanchor="bottom", showarrow=False, font=dict(size=12))

    fig.update_layout(height=820, margin=dict(t=50, b=50, l=60, r=90))
    return fig, combos


# --------------------------------------------------------------------------- #
# Assemble the page (with the two dropdowns + toggle JS)
# --------------------------------------------------------------------------- #
def build_page(fig, combos, params):
    locations = params["locations"]                  # canonical order
    combo_index = {key: i for i, key in enumerate(combos)}

    # JS run after the plot: two <select>s pick a combo -> toggle its PER traces.
    toggle_js = f"""
var gd = document.getElementById('ext-plot');
var LOC_ORDER = {json.dumps(locations)};
var COMBO_INDEX = {json.dumps(combo_index)};
var NTRACES = {len(combos) * PER};
var PER = {PER};
function keyFor(l1, l2) {{
    if (!l2 || l2 === l1) return l1;
    var a = LOC_ORDER.indexOf(l1), b = LOC_ORDER.indexOf(l2);
    return a <= b ? l1 + '_' + l2 : l2 + '_' + l1;
}}
function updateCombo() {{
    var l1 = document.getElementById('loc1').value;
    var l2 = document.getElementById('loc2').value;
    var key = keyFor(l1, l2);
    if (!(key in COMBO_INDEX)) key = l1;
    var i = COMBO_INDEX[key];
    var vis = new Array(NTRACES).fill(false);
    for (var k = 0; k < PER; k++) vis[i * PER + k] = true;
    Plotly.restyle(gd, {{'visible': vis}});
    document.getElementById('combo-label').textContent = key.replace('_', ' + ');
}}
document.getElementById('loc1').addEventListener('change', updateCombo);
document.getElementById('loc2').addEventListener('change', updateCombo);
"""

    plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn",
                           div_id="ext-plot", config={"responsive": True},
                           post_script=toggle_js)

    def options(selected, include_none=False):
        opts = []
        if include_none:
            sel = " selected" if selected == "" else ""
            opts.append(f'<option value=""{sel}>(none)</option>')
        for loc in locations:
            sel = " selected" if loc == selected else ""
            opts.append(f'<option value="{loc}"{sel}>{loc}</option>')
        return "\n".join(opts)

    # Site map + location description table, shown above the plot.
    site_rows = [
        ("<b>A</b>",  "CHORD fabrication tent and gravel parking area.", "600", "250"),
        ("<b>A2</b>", "Alternate placement of location A avoiding the CHORD fabrication tent.", "600", "250"),
        ("<b>B</b>",  "Gravel immediately south of Synthesis Telescope Track.", "220", "120"),
        ("<b>C</b>",  "Disturbed land immediately west of CHIME array.", "200", "0"),
        ("<b>D</b>",  "Disturbed land south of CHIME array.", "180", "150"),
        ("<b>E</b>",  "Disturbed land northeast of blockhouse.", "20", "220"),
        ("<b>F</b>",  "CHIME pathfinder.", "250", "70"),
        ("<b>G</b>",  "Natural land near north water pump station.", "250", "550"),
    ]
    table_rows = "\n".join(
        f"      <tr><td>{loc}</td><td>{desc}</td><td>{dew}</td><td>{dns}</td></tr>"
        for loc, desc, dew, dns in site_rows
    )
    # Location table -> lives at the top of the description column.
    table_html = f"""
<div class="site-table">
<h2>Description of extension sites</h2>
  <table>
    <thead><tr><th>Location</th><th>Description</th>
      <th>D<sub>EW</sub> (m)</th><th>D<sub>NS</sub> (m)</th></tr></thead>
    <tbody>
{table_rows}
    </tbody>
  </table>
</div>
"""
    # Site map -> above the plot.
    site_block = """
<div class="site-map-wrap">
  <img class="site-map" src="site_map_v2.png" alt="Map of CHORD extension sites">
</div>
"""

    d1, d2 = (DEFAULT_COMBO.split("_") + [""])[:2]
    controls = f"""
<div class="controls">
  <label>Location 1
    <select id="loc1">{options(d1)}</select>
  </label>
  <label>Location 2
    <select id="loc2">{options(d2, include_none=True)}</select>
  </label>
  <span class="combo">Showing: <strong id="combo-label">{DEFAULT_COMBO.replace('_', ' + ')}</strong></span>
</div>
"""

    freq_mhz = params["freq_hz"] / 1e6
    extent = params["beam_extent_arcmin"]
    npix = params["npix"]
    save_pix = params["save_pix"]
    description = f"""
<h2>Instructions</h2>
<p>Choose a site with <em>Location&nbsp;1</em> and leave <em>Location&nbsp;2</em> on
<em>(none)</em> to view a single-site configuration, or set both dropdowns to view a two-site
configuration. Every configuration also includes the CHORD compact core.</p>

<h2>Description</h2>
<p>The three heatmaps are the normalized synthesized beam at {freq_mhz:.0f}&nbsp;MHz over a
&plusmn;{abs(extent[0]):.0f}&nbsp;arcminute field of view: <strong>natural</strong> weighting
(top left), <strong>Briggs&nbsp;R=0</strong> (top right) and <strong>uniform</strong> weighting
(bottom left). The bottom-right panel shows the effective angular resolution (E&ndash;W and
N&ndash;S) against sensitivity loss (noise factor) as the Briggs robustness \\(R\\) varies.</p>

<h2>Methods</h2>
<h3>Synthesized Beam Plots</h3>
<p>The synthesized beam is constructed as follows:</p>
<ol>
<li>Antenna positions for the compact core and the selected site(s) are taken from the survey
layout shown in the site map above.</li>
<li><code>pyuvdata</code> constructs the UV coverage from the antenna positions.</li>
<li>The UV coverage is gridded and sampled with the chosen weighting (natural, Briggs, or
uniform).</li>
<li>An inverse FFT of the gridded samples gives the synthesized beam, normalized to a peak of 1.</li>
<li>The FFT is evaluated on a {npix}&times;{npix} grid with a \\(1^\\circ\\) field of view for
adequate sampling; only the inner {save_pix}&times;{save_pix} block of pixels
(\\(\\pm{abs(extent[0]):.1f}\\)&nbsp;arcminutes), which contains the main lobe, is shown.</li>
</ol>

<h3>Briggs sweep</h3>
<p>Effective angular resolution against noise factor as the Briggs robustness \\(R\\) is swept from
\\(+2\\) to \\(-2\\). The noise factor is the point-source noise of a given weighting relative to
natural weighting, which is the most sensitive. Natural-like weighting (\\(R \\approx +2\\)) sits
on the left and uniform-like weighting (\\(R \\approx -2\\)) on the right, with Briggs&nbsp;R=0
roughly in the middle.</p>
"""

    nav = nav_html(active="extension_chord")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CHORD extension-chord beams</title>
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
    <h2>Extension-CHORD: Site locations and synthesized beams</h2>
    <p>Plots of the synthesized beam patterns for different extension-CHORD configurations.</p>
  </section>
  <div class="layout">
    {site_block}
    <aside class="desc">{table_html}</aside>
    <div class="plot">{controls}{plot_div}</div>
    <aside class="desc">{description}</aside>
  </div>
</body>
</html>
"""
    return page


def main():
    d, params = load_data()
    fig, combos = build_figure(d, params)
    page = build_page(fig, combos, params)

    os.makedirs(DOCS_EXT, exist_ok=True)
    out = os.path.join(DOCS_EXT, "index.html")
    with open(out, "w") as f:
        f.write(page)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
