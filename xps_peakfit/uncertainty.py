"""ベイズ不確かさ定量化（emcee、opt-in）.

BIC選択後のベストモデルに対してMCMCサンプリングを行い、
各パラメータと面積の信頼区間（16/50/84パーセンタイル）を返す。
残差ベクトルには事前分布ペナルティが含まれているため、
サンプリング対象は正しくMAP事後分布になる（篠塚2024の
レプリカ交換MCと同じ役割を、収束済み近傍のアフィン不変
サンプラーで軽量に代替する設計）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lmfit
import numpy as np

from xps_peakfit.fitting import (
    FitResult,
    _residual,
    _so_values,
    estimate_noise_scale,
)
from xps_peakfit.models import doublet_area

logger = logging.getLogger(__name__)


@dataclass
class BayesResult:
    """MCMC不確かさ定量化の結果."""

    param_ci: dict[str, tuple[float, float, float]]  # name → (p16, p50, p84)
    area_ci: dict[str, tuple[float, float, float]]   # 成分名 → 面積CI
    acceptance_fraction: float
    n_samples: int
    table: list[dict] = field(default_factory=list)


def _finite_bounds(params: lmfit.Parameters, scale: float = 10.0) -> lmfit.Parameters:
    """emcee用に無限境界を有限化（現在値の±scale倍を目安）.

    現在値が0近傍のパラメータ（例: 背景スケールが〜0に収束した場合）でも
    min==max とならないよう、スパンに下限を設ける。
    """
    p = params.copy()
    for par in p.values():
        if not par.vary:
            continue
        span = scale * max(abs(par.value), 1e-3)
        if not np.isfinite(par.min):
            par.min = par.value - span
        if not np.isfinite(par.max):
            par.max = par.value + span
        if par.max - par.min < 1e-9:
            par.max = par.min + max(abs(par.value), 1e-6)
    return p


def bayesian_uncertainty(
    result: FitResult,
    steps: int = 1500,
    burn: int = 500,
    thin: int = 5,
    nwalkers: int | None = None,
    seed: int = 42,
    noise: str = "auto",
) -> BayesResult:
    """ベストフィット近傍の事後分布をemceeでサンプリングし信頼区間を推定.

    Args:
        result: fit_components / select_model のベスト結果
        steps: MCMCステップ数（burn含む）
        burn: 焼きなまし破棄数
        thin: 間引き
        nwalkers: ウォーカー数（既定: 自由パラメータ数×4以上の偶数）

    Note:
        emcee が未インストールの場合は ImportError を送出する。
        目安: 90点スペクトル・15パラメータで20〜40秒。
    """
    import emcee  # noqa: F401  (依存確認のみ。lmfitが内部で使用)

    spec = result.spectrum
    x, y = spec.energy, spec.intensity
    s = estimate_noise_scale(y) if noise == "auto" else 1.0
    w = 1.0 / (s * np.sqrt(np.clip(y, 1.0, None)))

    params = _finite_bounds(result.params)
    nvary = sum(1 for p in params.values() if p.vary)
    if nwalkers is None:
        nwalkers = max(2 * nvary + 2, 4 * nvary)
        nwalkers += nwalkers % 2

    mcmc = lmfit.minimize(
        _residual, params,
        args=(x, y, w, result.components, result.background_kind),
        method="emcee",
        nan_policy="omit",
        burn=burn, steps=steps, thin=thin, nwalkers=nwalkers,
        is_weighted=True, seed=seed, progress=False,
    )
    chain = mcmc.flatchain  # DataFrame: 列=可変パラメータ
    n_samples = len(chain)

    def ci(arr: np.ndarray) -> tuple[float, float, float]:
        p16, p50, p84 = np.percentile(arr, [16, 50, 84])
        return float(p16), float(p50), float(p84)

    param_ci = {name: ci(chain[name].to_numpy()) for name in chain.columns}

    # 成分ごとの面積の事後分布（チェーンから直接計算）
    area_ci: dict[str, tuple[float, float, float]] = {}
    table: list[dict] = []
    for comp in result.components:
        p = comp.pname

        def col(pname: str, default: float) -> np.ndarray:
            if pname in chain.columns:
                return chain[pname].to_numpy()
            return np.full(n_samples, default)

        amp = col(f"{p}_amp", result.params[f"{p}_amp"].value)
        fwhm = col(comp.fwhm_pname, result.params[comp.fwhm_pname].value)
        eta = col(comp.eta_pname, result.params[comp.eta_pname].value)
        so0, br0 = _so_values(result.params, comp)
        br = col(f"{p}_br", br0)

        if comp.shape == "doniach":
            # DS面積は窓依存のためチェーン上ではHeightのCIを面積代理とする
            areas = amp
            area_label = "Height (DS: area proxy)"
        else:
            areas = np.array([
                doublet_area(a, f, e, b)
                for a, f, e, b in zip(amp, fwhm, eta, br)
            ])
            area_label = "Area"
        area_ci[comp.name] = ci(areas)

        cen_ci = param_ci.get(
            f"{p}_cen",
            (result.params[f"{p}_cen"].value,) * 3,
        )
        fwhm_ci = param_ci.get(
            comp.fwhm_pname, (result.params[comp.fwhm_pname].value,) * 3,
        )
        table.append({
            "Component": comp.name,
            "Center_eV": round(cen_ci[1], 3),
            "Center_err": round(0.5 * (cen_ci[2] - cen_ci[0]), 3),
            "FWHM_eV": round(fwhm_ci[1], 3),
            "FWHM_err": round(0.5 * (fwhm_ci[2] - fwhm_ci[0]), 3),
            area_label: round(area_ci[comp.name][1], 1),
            "Area_err": round(0.5 * (area_ci[comp.name][2] - area_ci[comp.name][0]), 1),
        })

    acc = float(np.mean(mcmc.acceptance_fraction))
    if acc < 0.1:
        logger.warning("emcee受容率が低すぎます (%.2f)。steps/burnの増加を検討", acc)
    return BayesResult(
        param_ci=param_ci, area_ci=area_ci,
        acceptance_fraction=acc, n_samples=n_samples, table=table,
    )
