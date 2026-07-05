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
        """BIC比較表（ΔBIC<10 は有力候補として併記推奨）."""
        rows = []
        bic0 = self.best.bic
        for r in self.results:
            rows.append({
                "Components": " + ".join(c.name for c in r.components),
                "Background": r.background_kind,
                "N_components": len(r.components),
                "N_free_params": r.nfree,
                "Chi2_reduced": round(r.reduced_chi2, 3),
                "BIC": round(r.bic, 1),
                "dBIC": round(r.bic - bic0, 1),
                "Success": r.success,
            })
        return rows


def generic_candidates(
    spec: Spectrum,
    max_peaks: int = 5,
    fwhm_bounds: tuple[float, float] = (0.6, 3.0),
) -> list[list[Component]]:
    """無拘束モード: n本の独立擬Voigt (n=1..max_peaks) を等間隔初期配置."""
    x = spec.energy
    cands: list[list[Component]] = []
    for n in range(1, max_peaks + 1):
        comps = []
        for i in range(n):
            cen = x[0] + (x[-1] - x[0]) * (i + 0.5) / n
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
