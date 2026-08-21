# CHORD_Sensitivity

Supporting code and analysis of the thesis *Radio recombination line
forecasts with CHORD*.

## Interactive dashboards

Select results are published as an interactive website:

### 👉 https://duncancst.github.io/CHORD_Sensitivity/

- **RFI classification** — spectral-kurtosis flagging of the DRAO RFI monitor data.
- **CHORD sensitivity** — brightness-temperature sensitivity maps across the sky, by frequency.
- **Extension CHORD** — synthesized beams and Briggs resolution/noise trade-offs for
  compact-core extension configurations.

## Setup

Requires **Python ≥ 3.10**. Create a virtual environment and install the package
(editable), which pulls in all dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Check the python version if you hit import errors — the package needs 3.10+:

```bash
python --version
```

