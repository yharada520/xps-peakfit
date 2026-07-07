# GaN initial oxidation series (O 1s, Spring-8 BL23SU)

Synchrotron XPS series tracking the initial oxidation of a Ga-polar (+c)
GaN surface under a supersonic O2/He beam (translational energy 2.26 eV,
substrate at 200 °C), measured at Spring-8 beamline BL23SU with photon
energy **hv = 730 eV** (pass energy 5.0 eV, resolution ≈250 meV).
Measurement by M. Sumiya and co-workers (NIMS / JAEA BL23SU); see
M. Sumiya et al., *J. Phys. Chem. C* **124** (2020) 25282 for the
experiment and the chemical-state assignment
(O-O ≈530 eV molecular adsorption / Ga-O ≈531 eV dissociative /
N-O ≈532 eV), supported by DFMD calculations.

- Files: `<yymmdd><nnnn>_dsp.csv` — O 1s scans recorded sequentially
  during O2 exposure (≈30 s per spectrum in the steady phase).
- Format: headerless CSV, two columns: **kinetic energy (eV)**, intensity.
  Convert to binding energy as BE = 730 − KE, e.g.
  `load_spectrum(path, hv=730.0)` or CLI `--hv 730`.
- Region: KE 192–205 eV ⇔ BE 525–538 eV (O 1s), 0.1 eV steps.
- Note: scan `...0015` is an anomalous low-intensity acquisition
  (~1/10 counts) and is flagged as an outlier in the case study.

This dataset is openly available via the NIMS Materials Data Repository
(MDR): **https://doi.org/10.48505/nims.3848**

Analysis case study: `benchmarks/casestudy_gan_oxidation.py`.

Analysis case study: `benchmarks/casestudy_gan_oxidation.py`
(automatic chemical-state decomposition and oxidation kinetics extraction).
