"""フィッティングエンジンの単体テスト（合成スペクトルでの復元精度）."""
import numpy as np
import pytest

from xps_peakfit.background import shirley_from_peaks
from xps_peakfit.fitting import Component, estimate_noise_scale, fit_components
from xps_peakfit.io import Spectrum
from xps_peakfit.models import doublet_pseudo_voigt


@pytest.fixture()
def synthetic_doublet() -> tuple[Spectrum, dict]:
    """Au4f風ダブレット + Shirley背景 + ポアソンノイズの合成スペクトル."""
    rng = np.random.default_rng(0)
    x = np.arange(78.0, 96.0, 0.1)
    truth = dict(amp=8000.0, cen=84.0, fwhm=1.4, eta=0.3, split=3.67, ratio=0.75)
    peaks = doublet_pseudo_voigt(
        x, truth["amp"], truth["cen"], truth["fwhm"], truth["eta"],
        truth["split"], truth["ratio"],
    )
    bg = 2000.0 + shirley_from_peaks(x, peaks, k=0.02)
    y = rng.poisson(peaks + bg).astype(float)
    return Spectrum(x, y, "synthetic"), truth


def test_recover_doublet_parameters(synthetic_doublet) -> None:
    """既知パラメータのダブレットを高精度で復元できること."""
    spec, truth = synthetic_doublet
    comp = Component.from_line("Au4f", "Au0", fwhm_bounds=(0.8, 2.5))
    res = fit_components(spec, [comp], background="shirley", n_starts=4)
    assert res.success
    row = res.peak_table()[0]
    assert row["Center_eV"] == pytest.approx(truth["cen"], abs=0.05)
    assert row["FWHM_eV"] == pytest.approx(truth["fwhm"], abs=0.1)
    assert row["Height"] == pytest.approx(truth["amp"], rel=0.05)
    # ノイズスケールがほぼ1（生ポアソン）と推定されること
    assert estimate_noise_scale(spec.intensity) == pytest.approx(1.0, abs=0.25)
    # χ²_reduced が1近傍
    assert 0.7 < res.reduced_chi2 < 1.4


def test_fwhm_eta_shared_within_group(synthetic_doublet) -> None:
    """同一グループの成分はFWHM・ηパラメータを共有すること."""
    spec, _ = synthetic_doublet
    c1 = Component.from_line("Si2p", "Si0", center=83.5, center_sigma=0.5)
    c2 = Component.from_line("Si2p", "SiOx", center=86.5, center_sigma=0.5)
    res = fit_components(spec, [c1, c2], background="shirley", n_starts=2)
    assert c1.fwhm_pname == c2.fwhm_pname == "grp_Si2p_fwhm"
    assert c1.eta_pname == c2.eta_pname
    rows = res.peak_table()
    assert rows[0]["FWHM_eV"] == rows[1]["FWHM_eV"]
    assert rows[0]["Eta"] == rows[1]["Eta"]


def test_center_hard_bounds_two_sigma(synthetic_doublet) -> None:
    """事前分布のある成分は中心が±2σを超えないこと."""
    spec, _ = synthetic_doublet
    comp = Component(name="c", center=90.0, center_sigma=0.3)
    res = fit_components(spec, [comp], background="linear", n_starts=2)
    cen = res.params["c_cen"].value
    assert 90.0 - 0.6 - 1e-9 <= cen <= 90.0 + 0.6 + 1e-9


def test_invalid_background_raises(synthetic_doublet) -> None:
    spec, _ = synthetic_doublet
    with pytest.raises(ValueError):
        fit_components(spec, [Component(name="p", center=84.0)], background="spline")


def test_empty_components_raises(synthetic_doublet) -> None:
    spec, _ = synthetic_doublet
    with pytest.raises(ValueError):
        fit_components(spec, [])
