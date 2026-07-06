# EMPeaks reference comparison (2026-07-06)

**Purpose.** This is a *reference characterization*, not a contest. EMPeaks was
chosen as the comparison point simply because it is a valuable, pip-installable
public implementation of automated spectral fitting. EMPeaks and xps-peakfit
target different jobs: EMPeaks is built for high-throughput, prior-free
peak-shift tracking across large spectrum series, while xps-peakfit targets
physics-constrained decomposition of hard *single* spectra. The results below
characterize how each design behaves outside/inside its intended regime — they
should not be read as a ranking of the tools overall.

Environment: Windows 11, Python 3.12, EMPeaks 2.1.0 (PyPI, BSD-3) with scipy≥1.14
compat shims (`integrate.trapz/simps`). Script: `benchmarks/benchmark_empeaks.py`.

**Data note.** The Si2p_NG row below was measured on the original (proprietary,
not distributed) spectrum. The repository ships a synthetic equivalent
(`data/XPS_Si2p_siloxane_synthetic.csv`, generated from the published fit
parameters with measurement-matched noise); the benchmark script automatically
falls back to it when the original file is absent. The corresponding comparison
figure is likewise not distributed.

Conditions: identical data and energy windows; EMPeaks was **given the correct
number of peaks K** (it has no model selection) and 8 random restarts (`trial=8`,
matching our `n_starts` cap); pseudo-Voigt mixture with `linear` background
component, falling back to endpoint-linear pre-subtraction + `uniform` when the
raw fit crashes. `np.random.seed(0)` fixed externally (the sampling API has no
seed parameter). xps_peakfit v0.2.0 ran its standard physics-constrained mode.

## Summary

| case | EMPeaks time | ours time | EMPeaks RMSE | ours RMSE | EMPeaks outcome |
|------|-------------:|----------:|-------------:|----------:|-----------------|
| Si2p_NG | 897.0 s | 0.28 s | 3393 | 191 | raw crashed (2 configs, IndexError); pre-subtracted run put peaks at 99.4/100.3/101.4 eV — **no SiO2 (102.7) found, main-peak position missed** |
| Au4f | 9.5 s | 0.01 s | 38350 | 13414 | centers 84.07/87.73 (split 3.66 eV emerges from data — good); FWHM 2.3 eV (true ≈1.0) and area ratio 0.79 (true 0.75) degraded by background leakage |
| C1s | 83.7 s | 0.02 s | 105 | 100 | found 285.1/286.6 but "C-O" has FWHM 5.8 eV and 18% area (unphysical background absorber); ours: 2.6% area at fixed shared width |
| Cr2p | 23.6 s | 0.02 s | 348 | 117 | **2p1/2 partner at 563.7 eV completely missed** (2nd component became a 4.7 eV-wide background absorber); ours captures both via doublet constraint, fitted split 9.207 eV |
| Ni2p | 106.3 s | 0.09 s | 270 | 280 | positions roughly right (852.1/853.6/855.7/857.3) but FWHM 3.0–10.4 eV, area fractions 41/25/23/11% vs ours 67/14/10/9% |
| Si2p_reg | 253.3 s | 0.04 s | 124 | 120 | forced K=3 split the single SiO2 peak into 3 overlapping components |

## Findings

1. **Speed** (single spectra, this machine): xps_peakfit is 500–3200× faster on
   these cases. EMPeaks' pseudo-Voigt CM-step does bracketed root searches
   (`brentq`) per component per iteration; the published high-throughput claims
   are for the Gaussian-mixture EM and peak-shift tracking workloads, and do not
   transfer to pseudo-Voigt fits of wide-window XPS spectra.
2. **Robustness**: the pseudo-Voigt γ update uses a hard-coded search interval
   `(0.1, x_max−x_min)` that ignores the user's `gamma_min/gamma_max`; on
   spectra dominated by a flat background the bracket search finds no sign change
   and raises `IndexError` (2/6 datasets in raw form). The `eta` update searches
   only 0.8–1.0. No RNG seed parameter → run-to-run variability unless the
   global NumPy seed is fixed externally.
3. **Background is the decisive difference**: EMPeaks models background as a
   mixture component (uniform/linear/ramp), so on steep metallic backgrounds
   the extra "peaks" widen (up to 10 eV FWHM) to absorb it, corrupting both
   FWHM and area fractions even when centers are acceptable.
4. **Physics constraints are the second decisive difference**: without doublet
   locking, the Cr2p 2p1/2 partner is simply not found at K=2; with it, one
   component explains both peaks and even refines the splitting (9.21 eV).
5. **Where EMPeaks is fine**: well-separated peaks on mild backgrounds
   (Au4f centers; Ni2p centers) — consistent with its design target of
   high-throughput peak-shift mapping over spectrum series.

Caveat for any publication use: this is EMPeaks outside its primary use case; a
balanced comparison should also include a peak-shift-tracking task over a
spectrum series (EMPeaks' home turf) and note that K was supplied to EMPeaks
manually while xps_peakfit selects it via BIC. The robustness observations in
finding 2 concern implementation details of one released version (2.1.0) and
say nothing about the validity of the spectrum-adapted EM method itself.
