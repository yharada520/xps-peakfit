# xps-peakfit

Automated XPS peak fitting with **BIC model selection**, **MAP estimation**, and **physics-constrained spin-orbit doublets**.

Designed for hard cases such as weak chemical-state shoulders sitting on steep metallic backgrounds (e.g. siloxane Si 2p on an Au electrode with an Au ghost line underneath).

## Key features

- **Pseudo-Voigt peaks** with analytically correct areas (validated against numerical integration in the test suite)
- **Active backgrounds**: Shirley and Tougaard (universal cross-section) backgrounds are computed *inside* the fit model from the current peak sum, instead of being pre-subtracted — background estimation errors no longer propagate silently into peak areas
- **Physics constraints from a line database**: spin-orbit splitting and branching ratios (Si 2p: Δ0.61 eV 2:1, Au 4f: Δ3.67 eV 4:3, ...) are enforced exactly; FWHM and η are shared between chemical states of the same line
- **MAP estimation**: literature chemical-shift positions act as Gaussian priors on peak centers; the prior penalty is included in model comparison so physically implausible solutions are ranked down
- **BIC model selection**: candidate component sets (subsets of the physical pool, or 1..N unconstrained peaks) are fitted exhaustively; near-ties (ΔBIC < 10) are reported honestly instead of silently picking one
- **Noise-calibrated weighting**: Poisson weights with a robust scale factor estimated from second differences, so BIC stays meaningful for averaged/scaled data
- **Multi-start optimization** (jittered initial values) on top of `lmfit`
- **Automatic fit-range detection** (peak-prominence based, user-adjustable)
- Streamlit GUI + batch CLI

## Install

```bash
pip install -e .[gui,dev]
```

## Quick start

### CLI

```bash
# Physics-constrained mode: all Si2p chemical states + an Au4f-shaped ghost near 99.8 eV
xps-peakfit data/XPS_Si2p_siloxane_NG.csv \
    --window 98 104.5 --background tougaard \
    --line Si2p --ghost Au4f@99.8:0.5 --min-components 2

# Unconstrained mode: up to 5 independent pseudo-Voigt peaks, auto range
xps-peakfit data/XPS_Si2p.csv --auto-range --generic 5
```

### GUI

```bash
streamlit run app_streamlit.py
```

Upload a CSV (columns `energy`,`int` — flexible aliases accepted), adjust the auto-detected window, pick lines/states, run. You get the best-fit decomposition, the full BIC comparison table, and CSV downloads.

### Python API

```python
from xps_peakfit import load_spectrum, select_model
from xps_peakfit.fitting import Component
from xps_peakfit.model_select import subset_candidates

spec = load_spectrum("data/XPS_Si2p_siloxane_NG.csv").crop(98.0, 104.5)
pool = [
    Component.from_line("Au4f", "Au0", name="Au4f_ghost", center=99.8, center_sigma=0.5),
    Component.from_line("Si2p", "Si0"),
    Component.from_line("Si2p", "SiOx"),
    Component.from_line("Si2p", "SiO2"),
]
sel = select_model(spec, subset_candidates(pool, min_size=2), background="tougaard")
print(sel.best.peak_table())   # centers, FWHM, areas, area%
print(sel.summary())           # BIC ranking of all candidate compositions
```

## Benchmark: Si 2p on an Au ghost

`data/XPS_Si2p_siloxane_NG.csv` is a deliberately hard spectrum: a thin siloxane layer on an Au electrode, where the Si 2p region overlaps an Au ghost line and the SiO2 shoulder is a subtle feature on a steep background. The physically correct decomposition (SiO2 at 102–103 eV, SiOx/siloxane near 101 eV, ghost/elemental Si near 99–100 eV) is selected automatically by BIC:

| Component | Center (eV) | Area % |
|-----------|-------------|--------|
| Si0 / Au ghost | 99.91 | 70.0 |
| SiOx (siloxane) | 101.43 | 16.1 |
| SiO2 | 102.73 | 13.9 |

This case is locked in as a regression test (`tests/test_benchmark_ng.py`).

## Why "active" backgrounds and shared line widths matter

Classic workflows subtract a Shirley background first and fit peaks afterwards; any background error becomes an invisible peak-area error, and on steep metallic backgrounds the fit tends to park spurious peaks at the window edges. Fitting the background scale *simultaneously* with the peaks, constraining chemical states of one line to share FWHM/η, and letting spin-orbit partners be generated exactly (not fitted) reduces the free-parameter count enough that BIC can distinguish real chemical states from background artifacts.

## Line database

`xps_peakfit/lines.py` registers lines with spin-orbit constants and chemical-state priors (Si2p, Au4f, C1s, O1s included; extendable via `register(LineShape(...))`). Ghost lines borrow a registered line's shape while freeing its position prior — no element-specific code anywhere in the fitting engine.

## Tests

```bash
pytest            # 27 tests incl. two real-data benchmarks
pytest -m "not slow"
```

## License

MIT
