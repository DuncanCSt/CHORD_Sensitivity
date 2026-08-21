#!/usr/bin/env python
"""Build the interactive RFI dashboard page from the precomputed data.

Reads the CSV/JSON written by ``generate_data.py`` (in ``data/``), builds the
Plotly figure, and writes the standalone dashboard to ``docs/rfi/index.html``.

Run:  python components/rfi/generate_html.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DATA_DIR = os.path.join(_HERE, "data")
DOCS_RFI = os.path.join(_ROOT, "docs", "rfi")

# component_nav.py sits in components/, one level up from this script, and is
# shared by all three dashboards.
sys.path.insert(0, os.path.dirname(_HERE))
from component_nav import nav_html  # noqa: E402

WINDOW_MHZ = 2.0   # half-width of the dropdown zoom window around a line


# --------------------------------------------------------------------------- #
# Load precomputed data
# --------------------------------------------------------------------------- #
def load_data():
    traces = pd.read_csv(os.path.join(DATA_DIR, "rfi_spectrum_traces.csv"))
    h_lines = pd.read_csv(os.path.join(DATA_DIR, "H_lines_with_clean_scores.csv"))
    with open(os.path.join(DATA_DIR, "rfi_params.json")) as f:
        rfi_params = json.load(f)
    return traces, h_lines, rfi_params


# --------------------------------------------------------------------------- #
# Build the figure
# --------------------------------------------------------------------------- #
def build_figure(traces, h_lines, rfi_params):
    freq = traces["Frequency (MHz)"].to_numpy()
    trace_cols = [c for c in traces.columns if c.startswith("trace_")]
    SK = np.clip(traces["SK_cal"], 0, 3)

    is_clean = traces["combined_clean"] == 1
    status = np.where(is_clean, "clean", "RFI")   # per-bin label for hover

    # Two panes sharing the x-axis: zoom/pan on either drives both.
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.65, 0.35],
        subplot_titles=("RFI-monitor Power Spectral Density (dBm/Hz)",
                        "Calibrated spectral kurtosis"),
    )

    # --- top pane: all raw traces in ONE trace (NaN-separated) for speed ---
    raw = np.clip(traces[trace_cols].to_numpy(), None, -155)   # peaks clipped
    nan = np.array([np.nan])
    sep_obj = np.array([""], dtype=object)
    xs = np.concatenate([np.concatenate([freq, nan]) for _ in trace_cols])
    ys = np.concatenate([np.concatenate([raw[:, j], nan]) for j in range(len(trace_cols))])
    cd = np.concatenate([np.concatenate([status, sep_obj]) for _ in trace_cols])

    fig.add_trace(
        go.Scattergl(
            x=xs, y=ys, mode="markers", name="raw traces",
            marker=dict(size=1, color="blue"), opacity=0.6,
            customdata=cd,
            hovertemplate="Frequency: %{x:.3f} MHz<br>"
                          "Power: %{y:.2f} dBm/Hz<br>"
                          "Status: %{customdata}<extra></extra>",
        ),
        row=1, col=1,
    )

    # --- bottom pane: calibrated SK in ONE trace, coloured per-point ---
    sk_colors = np.where(is_clean, "green", "red")
    sk_customdata = np.empty((len(traces), 2), dtype=object)
    sk_customdata[:, 0] = traces["SK_cal"].to_numpy()   # unclipped SK value
    sk_customdata[:, 1] = status                        # "clean" / "RFI"

    fig.add_trace(
        go.Scattergl(
            x=freq, y=SK, mode="markers", name="SK_cal",
            marker=dict(size=3, color=sk_colors),
            customdata=sk_customdata,
            hovertemplate="Frequency: %{x:.3f} MHz<br>"
                          "SK: %{customdata[0]:.3f}<br>"
                          "Status: %{customdata[1]}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=rfi_params["rfi_thresholds"]["SK_low"],
                  line=dict(color="gray", dash="dash", width=1), row=2, col=1)
    fig.add_hline(y=rfi_params["rfi_thresholds"]["SK_high"],
                  line=dict(color="gray", dash="dash", width=1), row=2, col=1)

    # --- H-line markers as ONE line trace per pane (NaN-separated), not shapes ---
    hlf = h_lines["Frequency (MHz)"].to_numpy()
    top_lo, top_hi = np.nanmin(raw), -155         # y-extent of the raw pane
    vx = np.repeat(hlf, 3)
    vline_kw = dict(mode="lines", line=dict(color="black", dash="dash", width=0.5),
                    opacity=0.3, hoverinfo="skip", showlegend=False, name="H lines")
    fig.add_trace(go.Scattergl(x=vx, y=np.tile([top_lo, top_hi, np.nan], len(hlf)), **vline_kw),
                  row=1, col=1)
    fig.add_trace(go.Scattergl(x=vx, y=np.tile([0, 3, np.nan], len(hlf)), **vline_kw),
                  row=2, col=1)

    fig.update_yaxes(title_text="Power (dBm/Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Calibrated Spectral Kurtosis", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (MHz)", row=2, col=1)

    # --- dropdown: pick an H line and zoom the shared x-axis to it ---
    hl = h_lines.sort_values("Frequency (MHz)", ascending=False)

    def zoom_args(lo, hi):
        return [{"xaxis.range": [lo, hi], "xaxis2.range": [lo, hi]}]

    band_lo = rfi_params["band"]["chord_min_mhz"]
    band_hi = rfi_params["band"]["chord_max_mhz"]
    buttons = [dict(label="Full band", method="relayout",
                    args=zoom_args(band_lo, band_hi))]
    for _, r in hl.iterrows():
        f0 = r["Frequency (MHz)"]
        line_status = "clean" if r["clean"] else "RFI"
        buttons.append(dict(
            label=f'{f0:.2f} MHz ({line_status})',
            method="relayout",
            args=zoom_args(f0 - WINDOW_MHZ, f0 + WINDOW_MHZ),
        ))

    fig.update_layout(
        height=750,
        title="CHORD RFI-monitor: Measured Power, Spectral Kurtosis RFI labels, and H-line Visibility",
        hovermode="x unified",
        legend=dict(orientation="v", x=1.02, xanchor="left", y=0.5),
        margin=dict(t=110, b=50, r=140),
        updatemenus=[dict(
            buttons=buttons, direction="down", showactive=True,
            x=1.02, xanchor="left", y=1.0, yanchor="top", pad=dict(t=2, b=2),
        )],
        annotations=list(fig.layout.annotations) + [dict(
            text="Zoom to line:", x=1.02, xref="paper", xanchor="left",
            y=1.02, yref="paper", yanchor="bottom", showarrow=False,
            font=dict(size=15),
        )],
    )
    return fig


# --------------------------------------------------------------------------- #
# Assemble the HTML page
# --------------------------------------------------------------------------- #
def build_page(fig, rfi_params):
    band_lo = rfi_params["band"]["chord_min_mhz"]
    band_hi = rfi_params["band"]["chord_max_mhz"]

    # JS run after the plot is built: zoom-in grows the markers and thickens /
    # darkens the H-line verticals. z = 0 at full band, grows (log2) on zoom-in.
    resize_js = f"""
var gd = document.getElementById('rfi-plot');
var FULL = {band_hi - band_lo};
function resizeMarkers() {{
    var ax = gd.layout.xaxis;
    var r = (ax && ax.range) ? ax.range : null;
    var span = r ? Math.abs(r[1] - r[0]) : FULL;
    var z = Math.max(0, Math.log2(FULL / span));
    Plotly.restyle(gd, {{'marker.size': Math.min(6, 1 + z)}}, [0]);   // raw traces
    Plotly.restyle(gd, {{'marker.size': Math.min(6, 3 + z)}}, [1]);   // SK markers
    Plotly.restyle(gd, {{
        'line.width': Math.min(2.5, 0.5 + 0.4 * z),
        'opacity':    Math.min(1.0, 0.3 + 0.15 * z)
    }}, [2, 3]);                                                      // H-line verticals
}}
gd.on('plotly_relayout', resizeMarkers);
"""

    plot_div = fig.to_html(
        full_html=False, include_plotlyjs="cdn", div_id="rfi-plot",
        config={"responsive": True}, post_script=resize_js,
    )

    th = rfi_params["rfi_thresholds"]
    cut = rfi_params["cutoffs"]
    summ = rfi_params["summary"]
    res = rfi_params["resampling"]["velocity_resolution_kms"]
    vwidth = rfi_params["line_scoring"]["velocity_width_kms"]
    nseg = summ["n_segments"]
    agree = cut["segment_agreement_fraction"] * 100
    cleanfrac = summ["clean_channel_fraction"] * 100
    lo, hi = th["SK_low"], th["SK_high"]
    plo, phi = th["gamma_percentile"], th["gamma_percentile_upper"]
    thr = cut["line_clean_threshold"]
    nclean, nlines = summ["n_clean_lines"], summ["n_lines"]

    description = f"""
<h2>Instructions</h2>
<p>Pick a line from the <em>Zoom&nbsp;to&nbsp;line</em> dropdown to jump to a given
H&thinsp;n&alpha; line, or drag a box on either panel to zoom into a region, double click to reset zoom. The two panels
share the frequency axis, and hovering reports the frequency, value and clean/RFI status.</p>

<h2>Description</h2>
<p><strong>Top panel</strong> &mdash; 10 power samples from the RFI monitor for each frequency
bin, resampled to {res:.0f}&nbsp;km/s resolution.</p>
<p><strong>Bottom panel</strong> &mdash; the calibrated spectral kurtosis (SK) per channel,
<span style="color:green">green</span> where the channel is clean and
<span style="color:red">red</span> where it is flagged as RFI. Dashed verticals mark the
H&thinsp;n&alpha; lines.</p>

<h2>Methods</h2>
<h3>Segments</h3>
<p>The monitor records {nseg} usable time segments between calibration cycles. All statistics
are computed independently per segment and combined at the end.</p>

<h3>Spectral kurtosis</h3>
<p>Within a segment, each channel accumulates the power sums \\(S_1=\\sum P\\) and
\\(S_2=\\sum P^2\\) over its \\(M\\) time integrations. The generalized SK estimator is</p>
$$\\widehat{{SK}} = \\frac{{Nd\\,M + 1}}{{M-1}}\\left(\\frac{{M\\,S_2}}{{S_1^2}} - 1\\right),$$
<p>where \\(Nd\\) is the calibrated shape parameter. RFI-free data sits at
\\(\\widehat{{SK}}\\approx 1\\).</p>

<h3>Clean / RFI thresholds</h3>
<p>A channel is flagged as RFI when its SK falls outside the central
{plo}%&ndash;{phi}% range of an RFI-free SK distribution, simulated from
\\(\\mathrm{{Gamma}}(Nd,1)\\) draws:</p>
$$SK_{{{plo}\\%}} &lt; \\widehat{{SK}} &lt; SK_{{{phi}\\%}},
\\qquad [{lo:.3f},\\ {hi:.3f}].$$
<p>Segments are combined by majority vote: a channel is clean when more than {agree:.0f}% of
segments agree, leaving <strong>{cleanfrac:.1f}%</strong> of channels clean.</p>

<h3>Per-line visibility score</h3>
<p>Each H&thinsp;n&alpha; line is scored by the Gaussian-weighted fraction of clean bins in a
{vwidth:.0f}&nbsp;km/s window (\\(\\sigma = f_0\\,v/c\\)):</p>
$$\\mathrm{{score}}(f_0) = \\frac{{\\sum_i w_i\\,\\mathrm{{clean}}_i}}{{\\sum_i w_i}},
\\quad w_i = e^{{-\\tfrac{{1}}{{2}}\\left(\\tfrac{{f_i - f_0}}{{\\sigma}}\\right)^2}}.$$
<p>A line is usable when its score \\(\\ge {thr}\\):
<strong>{nclean} / {nlines}</strong> lines qualify.</p>
"""

    nav = nav_html(active="rfi")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CHORD RFI spectrum &amp; H-line visibility</title>
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
    <h2>CHORD RFI-monitor spectrum &amp; H-line visibility</h2>
    <p>Interactive assessment of hydrogen recombination line visibility against measured RFI.</p>
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
    traces, h_lines, rfi_params = load_data()
    fig = build_figure(traces, h_lines, rfi_params)
    page = build_page(fig, rfi_params)

    os.makedirs(DOCS_RFI, exist_ok=True)
    out = os.path.join(DOCS_RFI, "index.html")
    with open(out, "w") as f:
        f.write(page)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
