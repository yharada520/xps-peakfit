"""実データ回帰テスト: Au4f / C1s / Cr2p.

ダブレット拘束・ラインDB事前分布が標準的な実データで機能することを確認。
"""
from pathlib import Path

import pytest

from xps_peakfit import load_spectrum, select_model
from xps_peakfit.fitting import Component, fit_components
from xps_peakfit.model_select import subset_candidates

DATA = Path(__file__).parent.parent / "data"


@pytest.mark.slow
def test_au4f_doublet_metal() -> None:
    """Au4f: ダブレット拘束フィットで4f7/2=84.0 eVを再現."""
    spec = load_spectrum(DATA / "XPS_Au4f.csv").crop(80.5, 91.5)
    comp = Component.from_line("Au4f", "Au0", fwhm_bounds=(0.7, 2.0))
    res = fit_components(spec, [comp], background="shirley")
    row = res.peak_table()[0]
    assert row["Center_eV"] == pytest.approx(84.0, abs=0.15)
    assert 0.9 <= row["FWHM_eV"] <= 1.6


@pytest.mark.slow
def test_c1s_cc_dominant() -> None:
    """C1s: C-C主体（>50%）でC-C位置が284.5–285.4 eVに収束."""
    spec = load_spectrum(DATA / "XPS_C1s.csv").crop(282.0, 291.0)
    pool = [Component.from_line("C1s", s) for s in
            ("C-C", "C-O", "C=O", "O-C=O")]
    sel = select_model(spec, subset_candidates(pool, min_size=1),
                       background="auto")
    table = {r["Component"]: r for r in sel.best.peak_table()}
    assert "C1s_C-C" in table
    assert table["C1s_C-C"]["Area_pct"] > 50.0
    assert 284.5 <= table["C1s_C-C"]["Center_eV"] <= 285.4


@pytest.mark.slow
def test_ni2p_four_state_decomposition() -> None:
    """Ni2p: 金属+NiO+Ni(OH)2+サテライトの4成分がBIC選択されること."""
    spec = load_spectrum(DATA / "XPS_Ni2p.csv").crop(846.0, 866.0)
    pool = [Component.from_line("Ni2p", s, fwhm_bounds=(0.8, 3.5))
            for s in ("Ni0", "NiO", "Ni(OH)2", "Ni0_sat")]
    sel = select_model(spec, subset_candidates(pool, min_size=1),
                       background="auto")
    table = {r["Component"]: r for r in sel.best.peak_table()}
    assert len(table) == 4
    assert table["Ni2p_Ni0"]["Center_eV"] == pytest.approx(852.2, abs=0.4)
    assert table["Ni2p_Ni0"]["Area_pct"] > 50.0
    assert sel.best.reduced_chi2 < 1.3


@pytest.mark.slow
def test_cr2p_shifted_doublet_free_center() -> None:
    """Cr2p: 装置系で約-20 eVシフトしたデータを「位置フリーのダブレット」
    （ゴースト機構）でフィットし、2p3/2と2p1/2 (Δ9.3 eV) を同時再現."""
    spec = load_spectrum(DATA / "XPS_Cr2p.csv").crop(548.0, 566.0)
    comp = Component.from_line(
        "Cr2p", "Cr0", name="Cr2p_shifted",
        center=554.5, center_sigma=1.0, fwhm_bounds=(1.0, 3.0),
    )
    res = fit_components(spec, [comp], background="shirley")
    row = res.peak_table()[0]
    assert row["Center_eV"] == pytest.approx(554.5, abs=0.5)
    assert row["Doublet"] is True
