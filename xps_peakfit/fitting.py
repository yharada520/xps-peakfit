"""MAP推定フィッティングエンジン.

- 生データにポアソン重み（1/√y）でフィット（平滑化データは使わない）
- 背景（Shirley/Tougaard + 直線）をモデルに内包する active background 方式
- 中心位置の事前分布をペナルティ残差として追加（MAP推定）
- 初期値ジッターによるマルチスタート + 失敗時 differential_evolution フォールバック
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import lmfit
import numpy as np

from xps_peakfit.background import shirley_from_peaks, tougaard_from_peaks
from xps_peakfit.io import Spectrum
from xps_peakfit.lines import LineShape, get_line
from xps_peakfit.models import doublet_area, doublet_pseudo_voigt

logger = logging.getLogger(__name__)

BACKGROUNDS = ("shirley", "tougaard", "linear")


@dataclass(frozen=True)
class Component:
    """フィット成分1つ（シングレット or スピン軌道ダブレット）.

    center_sigma=inf で無情報事前分布（拘束なし）。
    so_split/branch_ratio はラインDBから固定供給される物理定数。
    """

    name: str
    center: float
    center_sigma: float = np.inf
    so_split: float = 0.0
    branch_ratio: float = 0.0
    fwhm_bounds: tuple[float, float] = (0.6, 3.0)
    eta_bounds: tuple[float, float] = (0.0, 1.0)
    center_window: float = 2.5  # 中心の探索範囲 center±window (eV)
    # 同一グループ名の成分間でFWHM/ηを共有（同一ラインの化学状態は
    # 装置分解能・寿命幅がほぼ共通、というXPSの標準拘束）
    fwhm_group: str | None = None
    eta_group: str | None = None

    @classmethod
    def from_line(
        cls,
        line_key: str,
        state_name: str,
        *,
        name: str | None = None,
        center: float | None = None,
        center_sigma: float | None = None,
        **kwargs,
    ) -> "Component":
        """ラインDBの (ライン, 化学状態) から成分を生成.

        center/center_sigma を上書きすると「ゴースト線」のように
        形状（分裂・比率）だけDBから借りて位置を自由化できる。
        """
        line: LineShape = get_line(line_key)
        st = line.state(state_name)
        kwargs.setdefault("fwhm_group", line.key)
        kwargs.setdefault("eta_group", line.key)
        return cls(
            name=name or f"{line.key}_{state_name}",
            center=center if center is not None else st.center_eV,
            center_sigma=center_sigma if center_sigma is not None else st.sigma_eV,
            so_split=line.so_split_eV,
            branch_ratio=line.branch_ratio,
            **kwargs,
        )

    @property
    def pname(self) -> str:
        """lmfitパラメータ名として使える識別子."""
        return re.sub(r"\W", "_", self.name)

    @property
    def fwhm_pname(self) -> str:
        if self.fwhm_group:
            return f"grp_{re.sub(r'\\W', '_', self.fwhm_group)}_fwhm"
        return f"{self.pname}_fwhm"

    @property
    def eta_pname(self) -> str:
        if self.eta_group:
            return f"grp_{re.sub(r'\\W', '_', self.eta_group)}_eta"
        return f"{self.pname}_eta"


@dataclass
class FitResult:
    """フィット結果一式."""

    spectrum: Spectrum
    components: list[Component]
    background_kind: str
    params: lmfit.Parameters
    success: bool
    chi2: float          # データ項のみの重み付きχ²
    prior_penalty: float  # 事前分布ペナルティ Σ((cen-μ)/σ)²
    ndata: int
    nfree: int
    curves: dict[str, np.ndarray] = field(default_factory=dict)  # name→成分曲線
    background: np.ndarray | None = None
    model: np.ndarray | None = None

    @property
    def bic(self) -> float:
        """BIC = (χ²_weighted + 事前ペナルティ) + k·ln(n).

        事前分布を破る解（例: ゴースト位置の大幅な逸脱）がモデル比較で
        有利にならないよう、MAP目的関数値でランク付けする。
        """
        return self.chi2 + self.prior_penalty + self.nfree * np.log(self.ndata)

    @property
    def aic(self) -> float:
        return self.chi2 + 2.0 * self.nfree

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / max(self.ndata - self.nfree, 1)

    def peak_table(self) -> list[dict]:
        """成分ごとの中心・FWHM・高さ・面積（ダブレットは合計面積）."""
        rows: list[dict] = []
        for comp in self.components:
            p = self.params
            amp = p[f"{comp.pname}_amp"].value
            cen = p[f"{comp.pname}_cen"].value
            fwhm = p[comp.fwhm_pname].value
            eta = p[comp.eta_pname].value
            rows.append({
                "Component": comp.name,
                "Center_eV": round(cen, 3),
                "FWHM_eV": round(fwhm, 3),
                "Eta": round(eta, 3),
                "Height": round(amp, 1),
                "Area": round(doublet_area(amp, fwhm, eta, comp.branch_ratio), 1),
                "Doublet": comp.branch_ratio > 0,
            })
        total = sum(r["Area"] for r in rows)
        for r in rows:
            r["Area_pct"] = round(100.0 * r["Area"] / total, 2) if total > 0 else 0.0
        return rows


def _eval_peaks(
    params: lmfit.Parameters, x: np.ndarray, components: list[Component]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    total = np.zeros_like(x)
    curves: dict[str, np.ndarray] = {}
    for comp in components:
        p = comp.pname
        c = doublet_pseudo_voigt(
            x,
            params[f"{p}_amp"].value,
            params[f"{p}_cen"].value,
            params[comp.fwhm_pname].value,
            params[comp.eta_pname].value,
            comp.so_split,
            comp.branch_ratio,
        )
        curves[comp.name] = c
        total += c
    return total, curves


def _eval_background(
    params: lmfit.Parameters, x: np.ndarray, peak_sum: np.ndarray, kind: str
) -> np.ndarray:
    bg = params["bg_const"].value + params["bg_slope"].value * (x - x[0])
    if kind == "shirley":
        bg = bg + shirley_from_peaks(x, peak_sum, params["bg_scale"].value)
    elif kind == "tougaard":
        bg = bg + tougaard_from_peaks(x, peak_sum, params["bg_scale"].value)
    return bg


def _residual(
    params: lmfit.Parameters,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    components: list[Component],
    bg_kind: str,
) -> np.ndarray:
    peak_sum, _ = _eval_peaks(params, x, components)
    model = _eval_background(params, x, peak_sum, bg_kind) + peak_sum
    r = (model - y) * w
    # MAP: 中心位置の事前分布ペナルティ
    priors = [
        (params[f"{c.pname}_cen"].value - c.center) / c.center_sigma
        for c in components
        if np.isfinite(c.center_sigma)
    ]
    if priors:
        return np.concatenate([r, np.asarray(priors)])
    return r


def _build_params(
    x: np.ndarray,
    y: np.ndarray,
    components: list[Component],
    bg_kind: str,
    rng: np.random.Generator | None = None,
    jitter: float = 0.0,
) -> lmfit.Parameters:
    params = lmfit.Parameters()
    # 直線背景の初期値: 両端を結ぶ直線
    slope0 = (y[-1] - y[0]) / (x[-1] - x[0])
    params.add("bg_const", value=float(np.min(y)), min=0.0)
    params.add("bg_slope", value=float(slope0))
    if bg_kind in ("shirley", "tougaard"):
        params.add("bg_scale", value=1e-3 if bg_kind == "shirley" else 100.0, min=0.0)

    baseline = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
    net = np.clip(y - baseline, 0.0, None)
    amp_scale = max(float(np.max(net)), 1.0)

    for comp in components:
        p = comp.pname
        cen0 = comp.center
        fwhm0 = 0.5 * (comp.fwhm_bounds[0] + comp.fwhm_bounds[1])
        if rng is not None and jitter > 0.0:
            span = min(comp.center_sigma, comp.center_window) if np.isfinite(comp.center_sigma) else comp.center_window
            cen0 = cen0 + jitter * span * (2.0 * rng.random() - 1.0)
            fwhm0 = np.clip(
                fwhm0 * (1.0 + jitter * (rng.random() - 0.5)),
                comp.fwhm_bounds[0], comp.fwhm_bounds[1],
            )
        amp0 = max(float(np.interp(cen0, x, net)), 0.02 * amp_scale)
        if rng is not None and jitter > 0.0:
            amp0 *= 1.0 + jitter * (rng.random() - 0.5)
        params.add(f"{p}_amp", value=amp0, min=0.0)
        # 中心のハード境界: 事前分布があれば±2σに制限（窓端への逃げ込み防止）
        cwin = comp.center_window
        if np.isfinite(comp.center_sigma):
            cwin = min(cwin, 2.0 * comp.center_sigma)
        params.add(
            f"{p}_cen", value=float(np.clip(cen0, comp.center - cwin, comp.center + cwin)),
            min=comp.center - cwin, max=comp.center + cwin,
        )
        # FWHM/η: グループ共有パラメータは初出時のみ生成
        if comp.fwhm_pname not in params:
            params.add(comp.fwhm_pname, value=float(fwhm0),
                       min=comp.fwhm_bounds[0], max=comp.fwhm_bounds[1])
        if comp.eta_pname not in params:
            eta0 = float(np.clip(0.3, comp.eta_bounds[0], comp.eta_bounds[1]))
            vary_eta = comp.eta_bounds[1] - comp.eta_bounds[0] > 1e-9
            params.add(comp.eta_pname, value=eta0,
                       min=comp.eta_bounds[0], max=comp.eta_bounds[1], vary=vary_eta)
    return params


def estimate_noise_scale(y: np.ndarray) -> float:
    """ポアソン仮定に対するノイズスケール係数 s を2階差分から頑健推定.

    実効ノイズ σ_i = s·√y_i。データが平均化・スケール済みの場合 s≠1 になる。
    2階差分 d_i = y_{i-1} - 2y_i + y_{i+1} の分散はノイズ分散の6倍。
    """
    y = np.asarray(y, dtype=float)
    d2 = y[:-2] - 2.0 * y[1:-1] + y[2:]
    z = d2 / np.sqrt(6.0 * np.clip(y[1:-1], 1.0, None))
    mad = np.median(np.abs(z - np.median(z)))
    s = 1.4826 * mad
    return float(max(s, 1e-3))


def fit_components(
    spec: Spectrum,
    components: list[Component],
    background: str = "shirley",
    n_starts: int = 8,
    seed: int = 42,
    noise: str = "auto",
    agree_rtol: float = 1e-3,
    min_agree: int = 2,
) -> FitResult:
    """MAP推定による適応マルチスタートフィット.

    物理拘束モードでは最適化地形がほぼ単峰のため、独立スタートのうち
    min_agree 回が最良コストに一致（相対差 agree_rtol 以内）した時点で
    早期終了する。n_starts は上限として機能する。

    Args:
        spec: フィット対象スペクトル（範囲切り出し済みであること）
        components: フィット成分のリスト
        background: "shirley" / "tougaard" / "linear"
        n_starts: マルチスタート最大回数（1回目はジッターなし）
        noise: "auto"（2階差分でスケール校正）/ "poisson"（σ=√y）
        agree_rtol: 解一致とみなすコストの相対許容差
        min_agree: 早期終了に必要な一致スタート数
    """
    if background not in BACKGROUNDS:
        raise ValueError(f"background は {BACKGROUNDS} のいずれか: {background}")
    if not components:
        raise ValueError("components が空です")

    x, y = spec.energy, spec.intensity
    s = estimate_noise_scale(y) if noise == "auto" else 1.0
    w = 1.0 / (s * np.sqrt(np.clip(y, 1.0, None)))  # 校正済みポアソン重み

    rng = np.random.default_rng(seed)
    best: lmfit.minimizer.MinimizerResult | None = None
    costs: list[float] = []
    for i in range(max(n_starts, 1)):
        params = _build_params(
            x, y, components, background,
            rng=rng if i > 0 else None, jitter=0.6 if i > 0 else 0.0,
        )
        try:
            res = lmfit.minimize(
                _residual, params, args=(x, y, w, components, background),
                method="least_squares", max_nfev=20000,
            )
        except Exception:
            logger.debug("start %d failed", i, exc_info=True)
            continue
        costs.append(float(res.chisqr))
        if best is None or res.chisqr < best.chisqr:
            best = res
        # 早期終了判定: 最良コストに一致するスタートが min_agree 回
        best_cost = float(best.chisqr)
        n_agree = sum(
            1 for c in costs
            if (c - best_cost) <= agree_rtol * max(abs(best_cost), 1e-12)
        )
        if n_agree >= min_agree:
            logger.debug("early stop at start %d (%d agree)", i + 1, n_agree)
            break

    if best is None:
        raise RuntimeError("全マルチスタートでフィットが収束しませんでした")

    # 結果の組み立て（χ²データ項と事前ペナルティを分離して再計算）
    peak_sum, curves = _eval_peaks(best.params, x, components)
    bg = _eval_background(best.params, x, peak_sum, background)
    model = bg + peak_sum
    chi2 = float(np.sum(((model - y) * w) ** 2))
    prior_penalty = float(sum(
        ((best.params[f"{c.pname}_cen"].value - c.center) / c.center_sigma) ** 2
        for c in components if np.isfinite(c.center_sigma)
    ))
    nfree = sum(1 for p in best.params.values() if p.vary)

    return FitResult(
        spectrum=spec,
        components=list(components),
        background_kind=background,
        params=best.params,
        success=bool(best.success),
        chi2=chi2,
        prior_penalty=prior_penalty,
        ndata=len(x),
        nfree=nfree,
        curves=curves,
        background=bg,
        model=model,
    )
