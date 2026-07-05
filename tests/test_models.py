"""ピークモデルの単体テスト（面積は数値積分と照合）."""
import numpy as np
import pytest

from xps_peakfit.models import (
    doublet_area,
    doublet_pseudo_voigt,
    pseudo_voigt,
    pv_area,
)


@pytest.mark.parametrize("eta", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_pv_area_matches_numerical_integration(eta: float) -> None:
    """解析面積式が数値積分と一致すること（legacyの2倍バグの回帰防止）."""
    amp, cen, fwhm = 1000.0, 100.0, 1.5
    # ローレンツ裾は長いので十分広い範囲で積分
    x = np.linspace(cen - 400, cen + 400, 400001)
    numerical = np.trapezoid(pseudo_voigt(x, amp, cen, fwhm, eta), x)
    analytical = pv_area(amp, fwhm, eta)
    assert analytical == pytest.approx(numerical, rel=2e-3)


def test_pv_height_and_fwhm() -> None:
    """高さ振幅定義とFWHM定義の検証."""
    amp, cen, fwhm, eta = 500.0, 99.0, 2.0, 0.4
    x = np.linspace(cen - 10, cen + 10, 20001)
    y = pseudo_voigt(x, amp, cen, fwhm, eta)
    assert np.max(y) == pytest.approx(amp, rel=1e-6)
    # 半値幅
    above = x[y >= amp / 2]
    assert (above[-1] - above[0]) == pytest.approx(fwhm, abs=0.01)


def test_doublet_ratio_and_split() -> None:
    """ダブレットの分裂幅・面積比が保持されること（Au4f想定）."""
    amp, cen, fwhm, eta = 1000.0, 84.0, 1.0, 0.3
    split, ratio = 3.67, 0.75
    # ローレンツ裾は長いので面積照合には広い積分範囲を使う
    x = np.linspace(cen - 400, cen + 400, 800001)
    y = doublet_pseudo_voigt(x, amp, cen, fwhm, eta, split, ratio)
    # 主ピーク位置と副ピーク位置
    main_idx = np.argmax(y)
    assert x[main_idx] == pytest.approx(cen, abs=0.01)
    sub_region = (x > cen + split - 1) & (x < cen + split + 1)
    sub_max = np.max(y[sub_region])
    assert sub_max == pytest.approx(amp * ratio, rel=0.01)
    # 合計面積 = 単体面積×(1+ratio)
    total = np.trapezoid(y, x)
    assert doublet_area(amp, fwhm, eta, ratio) == pytest.approx(total, rel=2e-3)


def test_doublet_degenerates_to_singlet() -> None:
    """branch_ratio=0 でシングレットに一致."""
    x = np.linspace(90, 110, 2001)
    a = doublet_pseudo_voigt(x, 100.0, 100.0, 1.5, 0.5, 0.0, 0.0)
    b = pseudo_voigt(x, 100.0, 100.0, 1.5, 0.5)
    np.testing.assert_allclose(a, b)
