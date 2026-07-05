"""改善1-5（DS形状・vary_so・事後確率・EM初期値・emcee）のテスト."""
import numpy as np
import pytest

from xps_peakfit.background import shirley_from_peaks
from xps_peakfit.fitting import Component, fit_components
from xps_peakfit.io import Spectrum
from xps_peakfit.model_select import (
    em_initial_centers,
    select_model,
    subset_candidates,
)
from xps_peakfit.models import doniach_sunjic, pseudo_voigt


# ---------- 改善3: Doniach-Šunjić ----------

def test_doniach_lorentzian_limit() -> None:
    """asym=0 でローレンツ（pv eta=1）に厳密一致."""
    x = np.linspace(80, 90, 2001)
    ds = doniach_sunjic(x, 1000.0, 84.0, 1.2, 0.0)
    lor = pseudo_voigt(x, 1000.0, 84.0, 1.2, 1.0)
    np.testing.assert_allclose(ds, lor, rtol=1e-10)


def test_doniach_tail_on_high_be_side() -> None:
    """asym>0 で高束縛エネルギー側の裾が重いこと."""
    x = np.linspace(70, 100, 3001)
    y = doniach_sunjic(x, 1000.0, 84.0, 1.2, 0.2)
    y_hi = float(np.interp(84.0 + 4.0, x, y))
    y_lo = float(np.interp(84.0 - 4.0, x, y))
    assert y_hi > 1.5 * y_lo
    # 高さ正規化: 定義は「x=cen で amp」。非対称頂点はcenからわずかに
    # 高BE側へずれ、最大値はampを数%超える（仕様）
    assert float(np.interp(84.0, x, y)) == pytest.approx(1000.0, rel=1e-6)
    assert 1000.0 <= float(np.max(y)) < 1100.0


# ---------- 改善1: スピン軌道定数の微変動 ----------

def test_vary_so_recovers_perturbed_split() -> None:
    """真の分裂幅がDB値から0.05 eVずれた合成ダブレットを、
    vary_so=True (σ=0.02) が固定より正しく追従すること."""
    rng = np.random.default_rng(11)
    x = np.arange(78.0, 96.0, 0.05)
    true_split = 3.72  # DB値3.67から+0.05
    peaks = (pseudo_voigt(x, 50000.0, 84.0, 1.2, 0.3)
             + pseudo_voigt(x, 37500.0, 84.0 + true_split, 1.2, 0.3))
    bg = 5000.0 + shirley_from_peaks(x, peaks, k=0.02)
    spec = Spectrum(x, rng.poisson(peaks + bg).astype(float))

    fixed = Component.from_line("Au4f", "Au0", fwhm_bounds=(0.8, 2.0))
    varied = Component.from_line("Au4f", "Au0", fwhm_bounds=(0.8, 2.0),
                                 vary_so=True)
    r_fix = fit_components(spec, [fixed], background="shirley")
    r_var = fit_components(spec, [varied], background="shirley")

    so_fit = r_var.params["Au4f_Au0_so"].value
    assert so_fit == pytest.approx(true_split, abs=0.02)
    assert r_var.chi2 < r_fix.chi2  # 微変動でデータ適合が改善
    # peak_table にも実効値が出ること
    row = r_var.peak_table()[0]
    assert row["SO_split_eV"] == pytest.approx(true_split, abs=0.02)


# ---------- 改善2: モデル事後確率 ----------

def test_model_posterior_probabilities_sum_to_one() -> None:
    rng = np.random.default_rng(5)
    x = np.arange(95.0, 110.0, 0.1)
    peaks = pseudo_voigt(x, 5000.0, 101.0, 1.5, 0.2)
    spec = Spectrum(x, rng.poisson(peaks + 1000.0).astype(float))
    pool = [
        Component(name="a", center=101.0, center_sigma=0.5),
        Component(name="b", center=105.0, center_sigma=0.5),
    ]
    sel = select_model(spec, subset_candidates(pool), background="linear",
                       n_starts=2)
    probs = [row["P_model"] for row in sel.summary()]
    assert sum(probs) == pytest.approx(1.0, abs=0.01)
    nprobs = sel.n_component_probabilities()
    assert sum(nprobs.values()) == pytest.approx(1.0, abs=0.01)
    # 真は1成分 → n=1 が優勢
    assert nprobs[1] > 0.5


# ---------- 改善4: EM初期値生成 ----------

def test_em_initial_centers_finds_two_peaks() -> None:
    rng = np.random.default_rng(9)
    x = np.arange(90.0, 115.0, 0.1)
    peaks = (pseudo_voigt(x, 8000.0, 97.0, 1.5, 0.2)
             + pseudo_voigt(x, 5000.0, 107.0, 1.5, 0.2))
    spec = Spectrum(x, rng.poisson(peaks + 500.0).astype(float))
    mu = em_initial_centers(spec, 2)
    assert mu[0] == pytest.approx(97.0, abs=0.5)
    assert mu[1] == pytest.approx(107.0, abs=0.5)


def test_em_initial_centers_flat_fallback() -> None:
    x = np.arange(0.0, 10.0, 0.1)
    spec = Spectrum(x, np.full_like(x, 100.0))
    mu = em_initial_centers(spec, 3)
    assert len(mu) == 3
    assert np.all((mu >= x[0]) & (mu <= x[-1]))


# ---------- 改善5: emcee不確かさ ----------

@pytest.mark.slow
def test_bayesian_uncertainty_smoke() -> None:
    """emceeが動作し、真値がCI近傍に入ること（短鎖スモーク）."""
    pytest.importorskip("emcee")
    from xps_peakfit.uncertainty import bayesian_uncertainty

    rng = np.random.default_rng(3)
    x = np.arange(95.0, 108.0, 0.1)
    peaks = pseudo_voigt(x, 20000.0, 101.0, 1.5, 0.3)
    spec = Spectrum(x, rng.poisson(peaks + 2000.0).astype(float))
    comp = Component(name="pk", center=101.0, center_sigma=0.5)
    res = fit_components(spec, [comp], background="linear")

    br = bayesian_uncertainty(res, steps=600, burn=200, thin=5)
    p16, p50, p84 = br.param_ci["pk_cen"]
    assert p16 < 101.0 < p84 or abs(p50 - 101.0) < 0.05
    assert br.n_samples > 100
    assert 0.05 < br.acceptance_fraction < 0.95
    assert br.table[0]["Center_err"] < 0.1  # 高S/Nなので誤差は小さい
