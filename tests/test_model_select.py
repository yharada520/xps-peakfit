"""モデル選択（BIC・背景auto・適応マルチスタート）のテスト."""
import numpy as np
import pytest

from xps_peakfit.background import shirley_from_peaks
from xps_peakfit.fitting import Component, fit_components
from xps_peakfit.io import Spectrum
from xps_peakfit.model_select import select_model, subset_candidates
from xps_peakfit.models import pseudo_voigt


@pytest.fixture()
def two_peak_spectrum() -> Spectrum:
    rng = np.random.default_rng(7)
    x = np.arange(95.0, 112.0, 0.1)
    peaks = (pseudo_voigt(x, 6000.0, 101.0, 1.5, 0.2)
             + pseudo_voigt(x, 3000.0, 104.5, 1.5, 0.2))
    bg = 1500.0 + shirley_from_peaks(x, peaks, k=0.03)
    return Spectrum(x, rng.poisson(peaks + bg).astype(float))


def _pool() -> list[Component]:
    return [
        Component(name="p101", center=101.0, center_sigma=0.5),
        Component(name="p104", center=104.5, center_sigma=0.5),
        Component(name="p107", center=107.5, center_sigma=0.5),
    ]


def test_bic_selects_true_component_count(two_peak_spectrum) -> None:
    """真の2成分構成がBIC最小になること."""
    sel = select_model(two_peak_spectrum, subset_candidates(_pool()),
                       background="shirley", n_starts=4)
    names = {c.name for c in sel.best.components}
    assert names == {"p101", "p104"}


def test_background_auto_compares_both(two_peak_spectrum) -> None:
    """background='auto' でshirley/tougaard両方が候補に含まれること."""
    sel = select_model(two_peak_spectrum, [_pool()[:2]],
                       background="auto", n_starts=2)
    bgs = {row["Background"] for row in sel.summary()}
    assert bgs == {"shirley", "tougaard"}
    assert sel.best.background_kind in bgs


def test_adaptive_multistart_early_stop(two_peak_spectrum, caplog) -> None:
    """単峰な問題では早期終了し、n_starts上限より少ない回数で済むこと."""
    import logging
    with caplog.at_level(logging.DEBUG, logger="xps_peakfit.fitting"):
        fit_components(two_peak_spectrum, _pool()[:2],
                       background="shirley", n_starts=16)
    assert any("early stop" in r.message for r in caplog.records)


def test_adaptive_multistart_same_answer(two_peak_spectrum) -> None:
    """早期終了ありでも上限まで回した場合と同じ解に到達すること."""
    comps = _pool()[:2]
    r_fast = fit_components(two_peak_spectrum, comps, background="shirley",
                            n_starts=16)
    r_full = fit_components(two_peak_spectrum, comps, background="shirley",
                            n_starts=16, min_agree=999)  # 早期終了無効化
    assert r_fast.chi2 == pytest.approx(r_full.chi2, rel=1e-3)
