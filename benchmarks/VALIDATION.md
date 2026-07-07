# Statistical validation for publication (2026-07-06/07)

Scripts: `validation_montecarlo.py` (detection / area recovery),
`validation_coverage.py` (credible-interval calibration),
`benchmark_series.py` (peak-shift tracking vs EMPeaks).
All synthetic truths use the NG-equivalent 3-component Si2p model
(doublets, shared FWHM/η, Tougaard+linear background, measurement-matched
noise scale s=0.638 unless stated).

## 1. Detection probability and false positives (BIC, n=30/cell)

Candidates: {Si0,SiOx} vs {Si0,SiOx,SiO2}, Tougaard background,
FWHM bounds (1.2, 2.4), η ≤ 0.6.

| true SiO2 area | s=0.4 | s=0.7 | s=1.0 | s=1.5 |
|---:|---:|---:|---:|---:|
| 0% (false-positive rate) | 0% | 0% | 6.7% | 0% |
| 3% | 83% | 10% | 23% | 13% |
| 6% | 100% | 70% | 40% | 20% |
| 9% | 100% | 80% | 40% | 10% |
| 14% | 100% | 90% | 63% | 43% |
| 20% | 100% | 80% | 63% | 33% |

- False positives stay at/below ~7% everywhere — BIC is properly conservative.
- At the measurement-like noise (s≈0.6–0.7), a 6–9% area minor component is
  detected in 70–80% of realizations; detection is noise-limited, matching the
  honest n-component posterior split (n=3: 60%) seen on the NG-equivalent data.
- Area recovery when detected: bias ≤ +25% at s=0.4–0.7 for ≥6% components,
  vanishing for larger components; the positive bias at the detection limit is
  the expected selection effect ("winner's curse") and should be quoted in the
  paper (figure `figures/mc_detection.png`).

## 2. Credible-interval calibration (emcee, 68% CI, n=50, steps=3000)

**Representative NG-like truth (fixed; "conflict" mode):** coverage is nominal.

| parameter | coverage | SD(est)/mean(σ_post) |
|---|---:|---:|
| center Si0 | 68% | 0.94 |
| center SiOx | 66% | 0.96 |
| center SiO2 | 72% | 0.87 |
| FWHM | 62% | 0.60 |
| SiO2 area | 64% | 0.86 |

**Prior-drawn truths ("calibrated" mode, centers~N(prior), FWHM~U(1.3,2.3),
η~U(0.1,0.55)):** coverage degrades to 44–66% (worst for FWHM, 36%), with
under-coverage concentrated in draws with small true FWHM (≤2.0 eV: 32–41%
vs >2.0 eV: 44–62%). Window-edge proximity and peak overlap were tested and
are *not* the drivers. The residual under-coverage is attributed to the
multi-parameter soft degeneracy (η–FWHM–background trade-off) that the
affine-invariant sampler explores slowly; chain lengths up to 6000 steps do
not resolve it.

**Design finding:** removing the practice FWHM bounds (1.2–2.4 → 1.2–3.2)
catastrophically degrades estimates (+0.27 eV FWHM bias, ~2× SiO2 area bias):
the width bounds act as essential regularization of the peak–background
degeneracy and must be treated as part of the prior, also during validation.

**Guidance for reported uncertainties:** at NG-like conditions the 68% CIs can
be used as-is; for strongly overlapped fits with near-limit widths, treat area
CIs as optimistic by up to ~×1.5–2 and always report the ΔBIC-competitive
alternative models alongside.

Noise-scale estimator check (200 realizations): bias +1.7%, relative SD 17% —
negligible contribution.

## 3. Peak-shift tracking (EMPeaks home-turf task)

25-spectrum synthetic series (static peak + peak shifting 100.0→100.8 eV,
Poisson noise). EMPeaks: Gaussian mixture (the model class of the original
2019 high-throughput paper), K=2, uniform background, trial=3.
xps_peakfit: 2 unconstrained components, warm-started from the previous
spectrum (series mode).

| method | time/spectrum | absolute RMSE | relative-shift RMSE |
|---|---:|---:|---:|
| EMPeaks (GMM) | 18–20 ms | 730 meV | 170 meV |
| xps_peakfit (warm start) | 17–18 ms | 6 meV | 13 meV |

EMPeaks tracks the *shape* of the shift with a constant offset (its component
mean is pulled by the neighbouring peak/background weight); for relative-shift
readout its error is ~170 meV here, while explicit peak fitting reaches
~10 meV at the same per-spectrum cost. Figure `figures/series_tracking.png`.

## Reproduction

```bash
python -X utf8 benchmarks/validation_montecarlo.py out/
python -X utf8 benchmarks/validation_coverage.py out/ calibrated
python -X utf8 benchmarks/validation_coverage.py out/ conflict
python -X utf8 benchmarks/benchmark_series.py out/
```
