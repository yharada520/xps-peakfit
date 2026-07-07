"""系列モード（構成固定＋形状ロック＋ウォームスタート）のテスト."""
import numpy as np
import pytest

from xps_peakfit.background import shirley_from_peaks
from xps_peakfit.fitting import Component
from xps_peakfit.io import Spectrum
from xps_peakfit.model_select import select_model, subset_candidates
from xps_peakfit.models import pseudo_voigt
from xps_peakfit.series import fit_series


def _pool(**kw):
    kw.setdefault("fwhm_bounds", (1.0, 2.2))
    kw.setdefault("eta_bounds", (0.0, 0.6))
    return [
        Component(name="A", center=100.0, center_sigma=0.5,
                  fwhm_group="g", eta_group="g", **kw),
        Component(name="B", center=102.5, center_sigma=0.5,
                  fwhm_group="g", eta_group="g", **kw),
    ]


def _make_series(n: int = 12, seed: int = 21) -> list[Spectrum]:
    """成分Bが検出限界近傍から単調成長する合成系列.

    真値: A固定(100.0, 8000)、Bは0→3000へ線形成長しつつ+0.3 eVシフト。
    """
    rng = np.random.default_rng(seed)
    x = np.arange(95.0, 108.0, 0.1)
    specs = []
    for i in range(n):
        amp_b = 3000.0 * i / (n - 1)
        cen_b = 102.5 + 0.3 * i / (n - 1)
        peaks = pseudo_voigt(x, 8000.0, 100.0, 1.6, 0.3)
        if amp_b > 0:
            peaks = peaks + pseudo_voigt(x, amp_b, cen_b, 1.6, 0.3)
        bg = 2000.0 + shirley_from_peaks(x, peaks, k=0.02)
        y = rng.poisson(peaks + bg).astype(float)
        specs.append(Spectrum(x, y, name=f"t{i}"))
    return specs


def test_series_mode_single_composition_and_smooth() -> None:
    """系列モード: 構成が全スペクトルで共通、中心軌跡が滑らかなこと."""
    specs = _make_series()
    sr = fit_series(specs, _pool(), background="shirley", n_starts=4)

    # 全スペクトルが同一構成（成分数の切替なし）
    assert all(len(r.components) == len(sr.composition) for r in sr.results)
    # 主成分Aの中心は系列を通じてほぼ一定（隣接差RMSが小さい）
    sm = sr.smoothness()
    assert sm["A"] < 0.05
    # 成分Bの面積は単調成長を再現（先頭より末尾が大きい）
    df = sr.table()
    b = df[df["component"] == "B"].sort_values("index")["area"].to_numpy()
    assert b[-1] > 5 * max(b[0], 1.0)


def test_series_mode_smoother_than_independent() -> None:
    """独立解析（毎回モデル選択）より中心軌跡が滑らかであること.

    BIC Top-Nから人が選ぶ方式で系統性が壊れる、という課題への回答。
    """
    specs = _make_series()
    sr = fit_series(specs, _pool(), background="shirley", n_starts=4)

    # 独立解析: スペクトル毎にselect_model（成分数が揺れ得る）
    indep_b_centers = []
    for sp in specs:
        sel = select_model(sp, subset_candidates(_pool(), min_size=1),
                           background="shirley", n_starts=4)
        tab = {r["Component"]: r for r in sel.best.peak_table()}
        indep_b_centers.append(tab.get("B", {}).get("Center_eV", np.nan))
    indep_b = np.array(indep_b_centers, dtype=float)

    # 系列モードはBが全点で存在し、NaNなし
    df = sr.table()
    series_b = (df[df["component"] == "B"].sort_values("index")
                ["center_eV"].to_numpy())
    assert not np.isnan(series_b).any()
    # 独立解析は初期（B極小）でBが欠落する点が存在する（=系統性の断絶）
    assert np.isnan(indep_b).any()


def test_series_shape_lock_fixes_fwhm() -> None:
    """形状ロック: FWHMの系列内変動が許容幅(±3%)以内に収まること."""
    specs = _make_series()
    sr = fit_series(specs, _pool(), background="shirley", lock_shape=True)
    df = sr.table()
    fw = df[df["component"] == "A"]["fwhm_eV"].to_numpy()
    assert (fw.max() - fw.min()) / fw.mean() < 0.07  # ±3%ロック+丸め余裕
    assert "shapes" in sr.lock_info