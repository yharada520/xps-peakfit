# xps-peakfit

Automated XPS peak fitting combining **BIC-based model selection**, **MAP estimation**, and **physics-constrained spin-orbit doublets**.

*[日本語版READMEはこちら / Japanese README → README.md](README.md)*

Designed for hard cases such as weak chemical-state shoulders sitting on steep metallic backgrounds (e.g. siloxane Si 2p on an Au electrode overlapping an Au ghost line): specify the window and the lines involved, and the physically plausible decomposition is selected automatically.

## Key features

- **Pseudo-Voigt peaks** (analytical areas validated against numerical integration in the test suite) and the **Doniach-Šunjić line shape** for asymmetric metal peaks
- **Active backgrounds**: Shirley and Tougaard backgrounds are computed *inside* the fit model from the current peak sum, instead of being pre-subtracted — background estimation errors no longer propagate silently into peak areas
- **Background selection by BIC** (`background="auto"`): Shirley vs Tougaard is decided by the data, not the user
- **Physics constraints from a line database**: spin-orbit splitting and branching ratios (Si 2p: Δ0.61 eV 2:1, Au 4f: Δ3.67 eV 4:3, ...) are enforced exactly; FWHM and η are shared between chemical states of the same line
- **Spin-orbit fine-tuning** (`vary_so=True`): splitting width and branching ratio can vary under tight Gaussian priors (±0.02 eV / ±0.02) to absorb instrument-level deviations
- **MAP estimation**: literature chemical-shift positions act as Gaussian priors on peak centers; the prior penalty is included in model comparison so physically implausible solutions are ranked down
- **BIC model selection with posterior probabilities**: candidate component sets (subsets of the physical pool, or 1..N unconstrained peaks) are fitted exhaustively; results are reported as approximate posterior probabilities per candidate and per component count (e.g. "n=3: 60%, n=2: 40%") — near-ties are reported honestly instead of silently picking one
- **Noise-calibrated weighting**: Poisson weights with a robust scale factor estimated from second differences, so BIC stays meaningful for averaged/scaled data
- **Adaptive multi-start optimization** (early stop on solution agreement, ~5x speedup) plus **spectrum-adapted EM initial guesses** for the unconstrained mode
- **Bayesian uncertainty quantification** (opt-in, `emcee`): 68% credible intervals for centers, widths, and areas from MCMC over the exact MAP posterior — seconds, not hours
- **Automatic fit-range detection** (peak-prominence based, user-adjustable)
- Streamlit GUI + batch CLI

## Install

```bash
pip install -e .[gui,bayes,dev]
```

## Quick start

### CLI

```bash
# Physics-constrained mode: all Si2p chemical states + an Au4f-shaped ghost near 99.8 eV
xps-peakfit data/XPS_Si2p_siloxane_NG.csv \
    --window 98 104.5 --background auto \
    --line Si2p --ghost Au4f@99.8:0.5 --min-components 2

# Unconstrained mode: up to 5 pseudo-Voigt peaks, auto range, with MCMC credible intervals
xps-peakfit data/XPS_Si2p.csv --auto-range --generic 5 --bayes
```

### GUI

```bash
streamlit run app_streamlit.py
```

Upload a CSV (columns `energy`,`int` — flexible aliases accepted), adjust the auto-detected window, pick lines/chemical states, run. You get the best-fit decomposition, the full BIC comparison table, component-count posterior probabilities, CSV downloads, and optional emcee credible intervals.

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
sel = select_model(spec, subset_candidates(pool, min_size=2), background="auto")
print(sel.best.peak_table())              # centers, FWHM, areas, area%
print(sel.n_component_probabilities())    # approximate posterior per component count
```

## Benchmark: Si 2p on an Au ghost

`data/XPS_Si2p_siloxane_NG.csv` is a deliberately hard spectrum: a thin siloxane layer on an Au electrode, where the Si 2p region overlaps an Au ghost line and the SiO2 shoulder is a subtle feature on a steep background. The physically correct decomposition (SiO2 at 102–103 eV, SiOx/siloxane near 101 eV, ghost/elemental Si near 99–100 eV) is selected automatically by BIC:

| Component | Center (eV) | Area % |
|-----------|-------------|--------|
| Si0 / Au ghost | 99.91 | 70.0 |
| SiOx (siloxane) | 101.43 | 16.1 |
| SiO2 | 102.73 | 13.9 |

This case is locked in as a regression test (`tests/test_benchmark_ng.py`).

## How this differs from other approaches

Excellent prior work exists on automated XPS analysis, and this package has learned a great deal from it. **The following is a comparison of design goals, not a ranking** — the right tool depends on the task.

| | EMPeaks (Matsumura, Ando, et al.) | Bayesian method by Shinotsuka et al. (NIMS) | xps-peakfit (this package) |
|---|---|---|---|
| Primary goal | Fast peak-shift tracking across large series of spectra | Rigorous Bayesian inference and UQ for a single spectrum | Practical automated chemical-state decomposition of hard single spectra |
| Optimizer | Spectrum-adapted EM/ECM (derivative-free) | Replica-exchange Monte Carlo | lmfit least-squares + adaptive multi-start |
| Number of peaks | User-specified (fixed K) | Automatic via Bayesian free energy F(K) | Automatic via BIC + approximate posteriors |
| Background | Mixture component (uniform/linear/ramp) | Endpoint-intensity parameters with priors | Active Shirley/Tougaard, selected by BIC |
| Physics knowledge | Deliberately none (data-driven) | Tight priors on spin-orbit constants only | Line database (splittings, ratios, chemical-shift priors, shared widths) |
| Uncertainty | — | Exact, from the posterior | Opt-in emcee MCMC (tens of seconds) |
| Sweet spot | Large operando/mapping series | Publication-grade rigorous analysis | Hard single spectra: metallic substrates, ghost overlaps |

- **EMPeaks** (Tarojiro Matsumura / AIST, Yasunobu Ando / Institute of Science Tokyo, et al.) formulates spectral intensity as mixture-model weights — an original EM approach enabling prior-free, data-driven, high-throughput analysis. A reference comparison on shared data lives in `benchmarks/`; **EMPeaks was chosen as the reference point simply because it is a valuable pip-installable public implementation** — the comparison is not a contest between tools designed for different jobs (see `benchmarks/RESULTS.md` for detailed fairness notes).
- **The Bayesian estimation method of Hiroshi Shinotsuka et al. (NIMS)** obtains full posteriors — including peak-count probabilities and credible intervals — via replica-exchange MC, and is methodologically the most rigorous approach. This package's spin-orbit fine-tuning (`vary_so`) and component-count posterior display are directly inspired by that line of work.
- **This package sits in between**: by injecting explicit physics knowledge (the line database), it aims to return a chemically interpreted answer ("SiO2: 13.9%") with optional credible intervals at a practical speed of seconds per spectrum.

## Line database

`xps_peakfit/lines.py` registers lines with spin-orbit constants and chemical-state priors (Si2p, Au4f, C1s, O1s, Cr2p, Ni2p included; extendable via `register(LineShape(...))`). Ghost lines borrow a registered line's shape while freeing its position prior — no element-specific code anywhere in the fitting engine.

## Tests

```bash
pytest            # 42 tests incl. six real-data benchmarks
pytest -m "not slow"
```

## License

MIT
