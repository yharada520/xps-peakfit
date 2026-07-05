"""背景モデルの単体テスト."""
import numpy as np
import pytest

from xps_peakfit.background import (
    shirley_background,
    shirley_from_peaks,
    tougaard_from_peaks,
)
from xps_peakfit.models import pseudo_voigt


def _synthetic(step_height: float = 500.0):
    """ピーク1本 + 階段背景の合成スペクトル."""
    x = np.arange(95.0, 110.0, 0.1)
    peak = pseudo_voigt(x, 5000.0, 102.0, 1.5, 0.3)
    bg_true = 1000.0 + step_height * np.cumsum(peak) / np.sum(peak)
    return x, peak, bg_true


def test_shirley_endpoints_and_monotonic() -> None:
    """古典Shirley: 端点一致・ピーク位置で単調増加."""
    x, peak, bg_true = _synthetic()
    y = peak + bg_true
    bg = shirley_background(x, y)
    assert bg[0] == pytest.approx(y[0])
    assert bg[-1] == pytest.approx(y[-1])
    assert np.all(np.diff(bg) >= -1e-9)  # 正のピークなら単調非減少
    # ピークを大きく超えないこと
    assert np.all(bg <= y + 1e-6)


def test_shirley_from_peaks_step_shape() -> None:
    """active Shirley: ピークの手前で0、通過後に一定値へ飽和."""
    x, peak, _ = _synthetic()
    k = 0.05
    bg = shirley_from_peaks(x, peak, k)
    assert bg[0] == 0.0
    total_area = np.trapezoid(peak, x)
    assert bg[-1] == pytest.approx(k * total_area, rel=1e-6)
    assert np.all(np.diff(bg) >= 0)


def test_tougaard_zero_before_peak_and_positive_after() -> None:
    """active Tougaard: ピークより低束縛エネルギー側でほぼ0、高側で正."""
    x, peak, _ = _synthetic()
    bg = tougaard_from_peaks(x, peak, b=2866.0)
    i_peak = int(np.argmax(peak))
    # 窓の低束縛エネルギー端では背景はごく小さい（ピーク裾の寄与のみ）
    assert bg[0] == 0.0
    assert np.all(bg[:10] < 0.02 * np.max(bg))
    assert np.max(bg) > 0
    # 背景最大はピークより高束縛エネルギー側
    assert np.argmax(bg) > i_peak


def test_tougaard_scales_linearly_with_b() -> None:
    x, peak, _ = _synthetic()
    bg1 = tougaard_from_peaks(x, peak, b=1000.0)
    bg2 = tougaard_from_peaks(x, peak, b=2000.0)
    np.testing.assert_allclose(bg2, 2.0 * bg1, rtol=1e-12)
