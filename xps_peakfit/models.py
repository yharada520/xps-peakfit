"""ピーク形状モデル（擬Voigt・スピン軌道ダブレット）と面積計算."""
from __future__ import annotations

import numpy as np

_4LN2 = 4.0 * np.log(2.0)
# 高さ振幅・FWHM定義での単位面積係数
GAUSS_AREA_COEF = 0.5 * np.sqrt(np.pi / np.log(2.0))  # ≈ 1.06447
LORENTZ_AREA_COEF = np.pi / 2.0                        # ≈ 1.57080


def pseudo_voigt(
    x: np.ndarray, amp: float, cen: float, fwhm: float, eta: float
) -> np.ndarray:
    """擬Voigt関数（高さ振幅定義）.

    pv(x) = amp * [(1-eta)·G(x) + eta·L(x)]
    G, L はいずれもピーク高さ1・FWHM共通。eta=0でガウス、1でローレンツ。
    """
    dx2 = ((x - cen) / fwhm) ** 2
    g = np.exp(-_4LN2 * dx2)
    l = 1.0 / (1.0 + 4.0 * dx2)
    return amp * ((1.0 - eta) * g + eta * l)


def pv_area(amp: float, fwhm: float, eta: float) -> float:
    """擬Voigtの解析的面積.

    area = amp·fwhm·[eta·π/2 + (1-eta)·(1/2)√(π/ln2)]
    （legacy app_v5.py の面積式は約2倍過大だったため修正済み）
    """
    return float(amp * fwhm * (eta * LORENTZ_AREA_COEF + (1.0 - eta) * GAUSS_AREA_COEF))


def doublet_pseudo_voigt(
    x: np.ndarray,
    amp: float,
    cen: float,
    fwhm: float,
    eta: float,
    so_split: float,
    branch_ratio: float,
) -> np.ndarray:
    """スピン軌道ダブレット擬Voigt.

    主成分（例: 2p3/2, 4f7/2）を (amp, cen) に置き、副成分を
    cen + so_split（高束縛エネルギー側）に面積比 branch_ratio で配置。
    FWHM・etaは両成分共有。branch_ratio=0 でシングレットに退化。
    """
    main = pseudo_voigt(x, amp, cen, fwhm, eta)
    if branch_ratio <= 0.0:
        return main
    return main + pseudo_voigt(x, amp * branch_ratio, cen + so_split, fwhm, eta)


def doublet_area(amp: float, fwhm: float, eta: float, branch_ratio: float) -> float:
    """ダブレット合計面積（主+副）."""
    return pv_area(amp, fwhm, eta) * (1.0 + max(branch_ratio, 0.0))


def doniach_sunjic(
    x: np.ndarray, amp: float, cen: float, fwhm: float, asym: float
) -> np.ndarray:
    """Doniach-Šunjićラインシェイプ（高さ振幅定義、金属ピークの非対称裾）.

    asym=0 でローレンツ関数（HWHM=fwhm/2）に厳密に一致する。
    asym>0 で高束縛エネルギー側（エネルギー損失側）に裾が伸びる。
    注意: asym>0 では無限区間の面積が発散するため、面積は解析窓内の
    数値積分（numeric_area）で定義する。
    """
    gamma = 0.5 * fwhm
    u = (cen - x) / gamma  # 高束縛エネルギー側に裾を出す符号
    num = np.cos(0.5 * np.pi * asym + (1.0 - asym) * np.arctan(u))
    den = (1.0 + u * u) ** (0.5 * (1.0 - asym))
    peak = np.cos(0.5 * np.pi * asym)  # x=cen での値
    return amp * (num / den) / peak


def numeric_area(x: np.ndarray, curve: np.ndarray) -> float:
    """解析窓内の数値積分面積（Doniach-Šunjić等の発散形状用）."""
    return float(np.trapezoid(curve, x))
