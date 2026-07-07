"""系列データの一貫解析（オペランド・時間分解測定向け）.

スペクトルごとに独立にモデル選択すると、検出限界近傍で成分数が
切り替わり、ピーク位置・半値幅が系統的なふるまいを示さなくなる
（BIC Top-N から人が選ぶ方式の既知の課題）。本モジュールは:

1. 成分構成と背景を「系列合計BIC」で一度だけ選択（モデル切替の排除）
2. FWHM/η を高S/N基準スペクトルで確定し系列全体で固定（形状の一貫性）
3. 中心は前スペクトルの解をウォームスタートに追跡（物理事前分布は維持）

により、速度・精確さ・物理的傾向の3つを両立する。
成分の出現/消失は「面積→0」として連続的に表現される。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from xps_peakfit.fitting import Component, FitResult, fit_components
from xps_peakfit.io import Spectrum
from xps_peakfit.model_select import subset_candidates

logger = logging.getLogger(__name__)


@dataclass
class SeriesResult:
    """系列解析の結果一式."""

    composition: list[Component]
    background: str
    results: list[FitResult]
    labels: list[str] = field(default_factory=list)
    lock_info: dict = field(default_factory=dict)  # 固定したFWHM/η等

    def table(self) -> pd.DataFrame:
        """整形済みロング形式テーブル（index, 成分, 中心, FWHM, 面積...）."""
        rows = []
        for i, (res, lab) in enumerate(zip(self.results, self.labels)):
            for r in res.peak_table():
                rows.append({
                    "index": i, "label": lab,
                    "component": r["Component"],
                    "center_eV": r["Center_eV"],
                    "fwhm_eV": r["FWHM_eV"],
                    "area": r["Area"], "area_pct": r["Area_pct"],
                    "chi2r": res.reduced_chi2, "bg": res.background_kind,
                })
        return pd.DataFrame(rows)

    def smoothness(self) -> dict[str, float]:
        """系統性の指標: 成分ごとの中心位置の隣接差RMS (eV).

        小さいほど系列として滑らか（物理的傾向が読み取りやすい）。
        """
        df = self.table()
        out = {}
        for name, g in df.groupby("component"):
            c = g.sort_values("index")["center_eV"].to_numpy()
            if len(c) >= 3:
                out[name] = float(np.sqrt(np.mean(np.diff(c) ** 2)))
        return out


def select_series_model(
    specs: list[Spectrum],
    pool: list[Component],
    background: str = "auto",
    min_size: int = 1,
    subsample: int | None = None,
    n_starts: int = 4,
) -> tuple[list[Component], str]:
    """系列合計BICで成分構成×背景を一度だけ選択する.

    Args:
        subsample: Noneで全スペクトル、kで先頭/末尾を含むk本に間引いて選択
                   （大規模系列の高速化。既定は min(全数, 8)）
    """
    if subsample is None:
        subsample = min(len(specs), 8)
    idx = np.unique(np.linspace(0, len(specs) - 1, subsample).astype(int))
    sel_specs = [specs[i] for i in idx]

    backgrounds = ("shirley", "tougaard") if background == "auto" \
        else (background,)
    candidates = subset_candidates(pool, min_size=min_size)

    best: tuple[float, list[Component], str] | None = None
    for comps in candidates:
        for bg in backgrounds:
            total = 0.0
            ok = True
            for sp in sel_specs:
                try:
                    r = fit_components(sp, comps, background=bg,
                                       n_starts=n_starts)
                except Exception:
                    ok = False
                    break
                total += r.bic
            if not ok:
                continue
            label = " + ".join(c.name for c in comps)
            logger.info("series candidate [%s|%s]: ΣBIC=%.1f", label, bg, total)
            if best is None or total < best[0]:
                best = (total, list(comps), bg)
    if best is None:
        raise RuntimeError("系列モデル選択: 全候補が失敗しました")
    return best[1], best[2]


def fit_series(
    specs: list[Spectrum],
    pool: list[Component],
    background: str = "auto",
    labels: list[str] | None = None,
    min_size: int = 1,
    lock_shape: bool = True,
    shape_tol: float = 0.03,
    n_starts: int = 4,
    subsample: int | None = None,
) -> SeriesResult:
    """系列一貫フィット（構成固定＋形状ロック＋中心ウォームスタート）.

    Args:
        specs: 測定順に並んだスペクトル列（範囲切り出し済み）
        pool: 化学状態プール
        background: "auto"（系列選択時にBIC比較）/ 固定指定
        lock_shape: Trueで高S/N基準スペクトルのFWHM/ηを系列全体に固定
        shape_tol: 固定時の許容相対幅（±3%既定。完全固定ではなく微動許可）
    """
    labels = labels or [s.name or str(i) for i, s in enumerate(specs)]

    # Phase 1: 構成と背景を系列全体で1回だけ決める
    composition, bg = select_series_model(
        specs, pool, background=background, min_size=min_size,
        subsample=subsample, n_starts=n_starts)
    logger.info("series composition: %s | bg=%s",
                [c.name for c in composition], bg)

    # Phase 2: 形状ロック — 最高積算強度のスペクトルでFWHM/ηを確定
    lock_info: dict = {}
    comps = list(composition)
    if lock_shape:
        ref_i = int(np.argmax([float(np.sum(s.intensity)) for s in specs]))
        ref_fit = fit_components(specs[ref_i], comps, background=bg,
                                 n_starts=max(n_starts, 8))
        locked: list[Component] = []
        seen_groups: dict[str, tuple[float, float]] = {}
        for c in comps:
            fw = ref_fit.params[c.fwhm_pname].value
            et = ref_fit.params[c.eta_pname].value
            seen_groups[c.fwhm_pname] = (fw, et)
            locked.append(replace(
                c,
                fwhm_bounds=(fw * (1 - shape_tol), fw * (1 + shape_tol)),
                eta_bounds=(max(et - 0.02, 0.0), min(et + 0.02, 1.0)),
            ))
        comps = locked
        lock_info = {"ref_index": ref_i, "ref_label": labels[ref_i],
                     "shapes": seen_groups}
        logger.info("shape locked from spectrum #%d: %s", ref_i, seen_groups)

    # Phase 3: 測定順にウォームスタートで追跡
    results: list[FitResult] = []
    prev_centers: dict[str, float] | None = None
    for sp in specs:
        res = fit_components(sp, comps, background=bg, n_starts=n_starts,
                             init_centers=prev_centers)
        prev_centers = {
            c.name: res.params[f"{c.pname}_cen"].value for c in comps
        }
        results.append(res)

    return SeriesResult(composition=comps, background=bg, results=results,
                        labels=labels, lock_info=lock_info)
