# GaN initial oxidation series (O 1s, Spring-8 BL23SU)

Synchrotron XPS series tracking the initial oxidation of a c-plane GaN
surface, measured at Spring-8 beamline BL23SU with photon energy
**hv = 730 eV**.

- Files: `<run><step>_dsp.csv` — the trailing 4 digits are the oxidation
  step index (5 … 340).
- Format: headerless CSV, two columns: **kinetic energy (eV)**, intensity.
  Convert to binding energy as BE = 730 − KE, e.g.
  `load_spectrum(path, hv=730.0)` or CLI `--hv 730`.
- Region: KE 192–205 eV ⇔ BE 525–538 eV (O 1s), 0.1 eV steps.

This dataset is openly available via the NIMS Materials Data Repository
(MDR). <!-- TODO: add MDR DOI and measurement credit (measurer /
beamline staff) before journal submission -->

Analysis case study: `benchmarks/casestudy_gan_oxidation.py`
(automatic chemical-state decomposition and oxidation kinetics extraction).
