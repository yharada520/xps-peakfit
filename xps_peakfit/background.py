"""背景モデル: Shirley / Tougaard（active background 対応）.

active background では「現在のピーク成分和」から背景形状を計算し、
スケール係数のみをフィットパラメータとする。前引き方式で生じる
背景推定誤差→面積誤差の伝播を避けるための設計。
"""
from __future__ import annotations

import numpy as np

TOUGAARD_C_UNIVERSAL = 1643.0  # eV^2 (Tougaard universal cross-section)


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """累積台形積分（先頭0）."""
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum((y[:-1] + y[1:]) * 0.5 * np.diff(x))
    return out


def shirley_background(
    x: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-7
) -> np.ndarray:
    """古典的Shirley背景（端点固定・反復法）。表示・初期化用.

    x は束縛エネルギー昇順。背景は高束縛エネルギー側で高くなる。
    """
    y = np.asarray(y, dtype=float)
    y0, y1 = float(y[0]), float(y[-1])
    bg = np.linspace(y0, y1, len(y))
    for _ in range(max_iter):
        integral = _cumtrapz(y - bg, x)
        total = integral[-1]
        if abs(total) < 1e-12:
            break
        bg_new = y0 + (y1 - y0) * integral / total
        if np.max(np.abs(bg_new - bg)) < tol * max(abs(y1 - y0), 1.0):
            bg = bg_new
            break
        bg = bg_new
    return bg


def shirley_from_peaks(x: np.ndarray, peak_sum: np.ndarray, k: float) -> np.ndarray:
    """active Shirley: ピーク成分和の累積積分 × スケール k.

    bg(E) = k·∫_{E_min}^{E} peaks(E') dE'
    （非弾性散乱により、ピークより高束縛エネルギー側に階段状背景が生じる）
    """
    return k * _cumtrapz(peak_sum, x)


def tougaard_from_peaks(
    x: np.ndarray, peak_sum: np.ndarray, b: float, c: float = TOUGAARD_C_UNIVERSAL
) -> np.ndarray:
    """active Tougaard: universal cross-section による損失背景.

    bg(E) = Σ_{E'<E} B·T/(C+T²)² · peaks(E')·ΔE,  T = E - E'
    等間隔グリッドを仮定し離散畳み込みで計算する。
    """
    n = len(x)
    dx = float(np.median(np.diff(x)))
    t = np.arange(n) * dx  # T >= 0
    kernel = b * t / (c + t * t) ** 2
    kernel[0] = 0.0
    bg = np.convolve(peak_sum, kernel)[:n] * dx
    return bg
