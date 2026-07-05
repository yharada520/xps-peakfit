"""BICによるモデル（成分構成）選択.

候補となる成分集合を総当たりでフィットし、BIC最小の構成を選ぶ。
拘束ダブレットは自由パラメータが少ないため、同等の適合度なら
物理的に正しい構成がBIC上有利になる。
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from xps_peakfit.fitting import Component, FitResult, fit_components
from xps_peakfit.io import Spectrum

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """モデル選択の結果. results はBIC昇順."""

    best: FitResult
    results: list[FitResult] = field(default_factory=list)

    def summary(self) -> list[dict]:
        """BIC比較表（ΔBIC<10 は有力候補として併記推奨）.

        P_model はBIC重み exp(-ΔBIC/2) を全候補で正規化した近似事後確率。
        """
        bic0 = self.best.bic
        weights = np.array([np.exp(-0.5 * (r.bic - bic0)) for r in self.results])
        probs = weights / weights.sum()
        rows = []
        for r, pm in zip(self.results, probs):
            rows.append({
                "Components": " + ".join(c.name for c in r.components),
                "Background": r.background_kind,
                "N_components": len(r.components),
                "N_free_params": r.nfree,
                "Chi2_reduced": round(r.reduced_chi2, 3),
                "BIC": round(r.bic, 1),
                "dBIC": round(r.bic - bic0, 1),
                "P_model": round(float(pm), 4),
                "Success": r.success,
            })
        return rows

    def n_component_probabilities(self) -> dict[int, float]:
        """成分数nごとの近似事後確率（篠塚2024のK事後確率に相当）."""
        bic0 = self.best.bic
        weights = np.array([np.exp(-0.5 * (r.bic - bic0)) for r in self.results])
        probs = weights / weights.sum()
        out: dict[int, float] = {}
        for r, pm in zip(self.results, probs):
            n = len(r.components)
            out[n] = out.get(n, 0.0) + float(pm)
        return dict(sorted(out.items()))


def em_initial_centers(
    spec: Spectrum, n: int, n_iter: int = 150
) -> np.ndarray:
    """スペクトル適応EM（Matsumura/EMPeaks方式）による初期中心の推定.

    強度（線形背景除去後）を混合分布の重みとみなし、n成分ガウス混合を
    EM更新で推定する。微分不要・単調収束のため初期値生成として頑健。
    """
    x, y = spec.energy, spec.intensity
    baseline = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
    w = np.clip(y - baseline, 0.0, None)
    if w.sum() <= 0:
        # 信号なし → 等間隔フォールバック
        return x[0] + (x[-1] - x[0]) * (np.arange(n) + 0.5) / n
    w = w / w.sum()

    # 初期値: 重み累積分布の分位点
    cdf = np.cumsum(w)
    mu = np.interp((np.arange(n) + 0.5) / n, cdf, x)
    sig = np.full(n, max((x[-1] - x[0]) / (4.0 * n), spec.step))
    pi = np.full(n, 1.0 / n)

    for _ in range(n_iter):
        d = (x[None, :] - mu[:, None]) / sig[:, None]
        g = pi[:, None] * np.exp(-0.5 * d * d) / sig[:, None]
        g_sum = np.clip(g.sum(axis=0, keepdims=True), 1e-300, None)
        resp = (g / g_sum) * w[None, :]  # 責務×強度重み
        nk = np.clip(resp.sum(axis=1), 1e-12, None)
        mu_new = (resp * x[None, :]).sum(axis=1) / nk
        var = (resp * (x[None, :] - mu_new[:, None]) ** 2).sum(axis=1) / nk
        sig = np.sqrt(np.clip(var, spec.step ** 2, None))
        pi = nk / nk.sum()
        if np.max(np.abs(mu_new - mu)) < 1e-4:
            mu = mu_new
            break
        mu = mu_new
    return np.sort(mu)


def generic_candidates(
    spec: Spectrum,
    max_peaks: int = 5,
    fwhm_bounds: tuple[float, float] = (0.6, 3.0),
    init: str = "em",
) -> list[list[Component]]:
    """無拘束モード: n本の独立擬Voigt (n=1..max_peaks) の候補を生成.

    init="em" でスペクトル適応EMによる初期中心（既定）、
    init="grid" で等間隔初期配置。
    """
    x = spec.energy
    cands: list[list[Component]] = []
    for n in range(1, max_peaks + 1):
        if init == "em":
            centers = em_initial_centers(spec, n)
        else:
            centers = x[0] + (x[-1] - x[0]) * (np.arange(n) + 0.5) / n
        comps = []
        for i, cen in enumerate(centers):
            comps.append(Component(
                name=f"peak{i + 1}", center=float(cen),
                center_sigma=np.inf, fwhm_bounds=fwhm_bounds,
                center_window=float((x[-1] - x[0]) / 2),
            ))
        cands.append(comps)
    return cands


def subset_candidates(
    components: list[Component], min_size: int = 1, max_size: int | None = None
) -> list[list[Component]]:
    """物理拘束モード: 指定成分プールの部分集合を候補として列挙."""
    max_size = max_size or len(components)
    cands: list[list[Component]] = []
    for k in range(min_size, max_size + 1):
        for combo in combinations(components, k):
            cands.append(list(combo))
    return cands


def select_model(
    spec: Spectrum,
    candidates: list[list[Component]],
    background: str | Sequence[str] = "auto",
    n_starts: int = 8,
    seed: int = 42,
) -> SelectionResult:
    """候補構成×背景を総当たりフィットし、BIC最小を選択.

    background="auto" で shirley/tougaard の両方を候補に含め、
    背景モデルの選択もBICに委ねる（成分構成と同時に自動決定）。
    """
    if background == "auto":
        backgrounds: tuple[str, ...] = ("shirley", "tougaard")
    elif isinstance(background, str):
        backgrounds = (background,)
    else:
        backgrounds = tuple(background)

    results: list[FitResult] = []
    for comps in candidates:
        label = " + ".join(c.name for c in comps)
        for bg in backgrounds:
            try:
                res = fit_components(
                    spec, comps, background=bg, n_starts=n_starts, seed=seed
                )
                results.append(res)
                logger.info("candidate [%s|%s]: BIC=%.1f chi2r=%.3f",
                            label, bg, res.bic, res.reduced_chi2)
            except Exception:
                logger.exception("candidate [%s|%s] のフィットに失敗", label, bg)
    if not results:
        raise RuntimeError("全候補のフィットに失敗しました")
    results.sort(key=lambda r: r.bic)
    return SelectionResult(best=results[0], results=results)
