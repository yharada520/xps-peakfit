"""フィット範囲の自動検出.

急峻な背景上のピークでは「信号がノイズ床に落ちる点」が存在しないことが
多いため、次のヒューリスティックを採用する:

1. Savitzky-Golayで平滑化（初期値検出用途のみ。フィットには使わない）
2. ノイズσを2階差分のMADで頑健推定
3. prominence > n_sigma·σ のピークを検出
4. 最外ピーク中心 ± pad_factor×FWHM を窓とする

GUIではこの値を初期値としてユーザーが調整できる。
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter

from xps_peakfit.io import Spectrum

logger = logging.getLogger(__name__)


def estimate_noise_sigma(y: np.ndarray) -> float:
    """2階差分MADによるノイズ標準偏差の頑健推定."""
    d2 = y[:-2] - 2.0 * y[1:-1] + y[2:]
    mad = np.median(np.abs(d2 - np.median(d2)))
    return float(max(1.4826 * mad / np.sqrt(6.0), 1e-9))


def auto_range(
    spec: Spectrum,
    n_sigma: float = 8.0,
    pad_factor: float = 2.2,
    smooth_window: int = 11,
) -> tuple[float, float]:
    """フィット範囲 (emin, emax) を自動推定する.

    Args:
        spec: 対象スペクトル（広めの範囲を含むこと）
        n_sigma: ピーク検出のprominence閾値（ノイズσの倍数）
        pad_factor: 最外ピークからのマージン（FWHMの倍数）
        smooth_window: 平滑化窓（奇数）

    Returns:
        (emin, emax)。ピークが見つからない場合は全範囲を返す。
    """
    x, y = spec.energy, spec.intensity
    win = min(smooth_window, (len(y) // 2) * 2 - 1)
    win = max(win, 5)
    if win % 2 == 0:
        win += 1
    smoothed = savgol_filter(y, window_length=win, polyorder=min(3, win - 2))

    sigma = estimate_noise_sigma(y)
    peaks_idx, props = find_peaks(smoothed, prominence=n_sigma * sigma)
    if len(peaks_idx) == 0:
        logger.warning("ピークが検出できませんでした。全範囲を返します")
        return float(x[0]), float(x[-1])

    widths, _, _, _ = peak_widths(smoothed, peaks_idx, rel_height=0.5)
    step = spec.step
    fwhms = widths * step

    lows = x[peaks_idx] - pad_factor * fwhms
    highs = x[peaks_idx] + pad_factor * fwhms
    emin = float(max(np.min(lows), x[0]))
    emax = float(min(np.max(highs), x[-1]))
    logger.info("auto_range: peaks=%s -> window=(%.2f, %.2f)",
                np.round(x[peaks_idx], 2).tolist(), emin, emax)
    return emin, emax
