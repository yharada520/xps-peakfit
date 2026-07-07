"""emcee信用区間の較正検証（simulation-based calibration, 論文用）.

代表条件（SiO2面積比14%・実測相当ノイズs=0.638）で合成スペクトルを
n_rep回生成し、各回で3成分フィット＋emcee MCMCを実行。68%信用区間に
真値が入る割合（被覆率）が名目値68%と整合するかを検証する。

2つのモード:
- calibrated（既定）: 各回の真値中心を事前分布 N(μ_prior, σ_prior) から
  抽選する。ベイズ的自己一貫性（Cook–Gelman–Rubin流）の検証で、
  手法が正しく較正されていれば被覆率は名目値に一致するはず。
- conflict: 真値中心を事前分布からずらした固定値（NG試料相当）にする。
  事前分布と試料が食い違う場合の頑健性（被覆率低下の程度）を測る。
  情報事前分布を使う以上、このモードでの被覆率低下は原理的に不可避。

実行: python -X utf8 benchmarks/validation_coverage.py [出力dir] [calibrated|conflict]
所要: 既定60回で6分前後
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.validation_montecarlo import (  # noqa: E402
    BRANCH, SO_SPLIT, TRUTH, X,
)
from xps_peakfit.background import tougaard_from_peaks  # noqa: E402
from xps_peakfit.fitting import Component, fit_components  # noqa: E402
from xps_peakfit.io import Spectrum  # noqa: E402
from xps_peakfit.lines import get_line  # noqa: E402
from xps_peakfit.models import doublet_area, doublet_pseudo_voigt  # noqa: E402
from xps_peakfit.uncertainty import bayesian_uncertainty  # noqa: E402

REPO = Path(__file__).parent.parent

R_SIO2 = 0.14
NOISE_S = 0.638  # 実測NGデータと同じノイズスケール
# 実運用と同一の境界を使う。FWHM/η境界はピーク↔背景縮退を抑える
# 正則化（事前分布の一部）であり、較正検証でも外さないのが正しい設計
FWHM_B = (1.2, 2.4)
ETA_B = (0.0, 0.6)
MIN_SEP = 0.9  # 抽選中心の最小間隔 (eV)。ラベル入れ替わりの防止


def _components() -> list[Component]:
    kw = dict(fwhm_bounds=FWHM_B, eta_bounds=ETA_B)
    return [Component.from_line("Si2p", s, **kw)
            for s in ("Si0", "SiOx", "SiO2")]


def _draw_truth(rng: np.random.Generator) -> dict:
    """事前分布から真値を抽選（calibratedモード用）.

    中心はガウス事前分布（ハード境界±2σに合わせてクリップ）、
    FWHM/ηは一様事前分布（境界の内側から余裕を持って抽選）。
    振幅・背景は固定（条件付き較正）。
    """
    line = get_line("Si2p")
    while True:
        cs = []
        ok = True
        for st in (line.state("Si0"), line.state("SiOx"), line.state("SiO2")):
            c = float(rng.normal(st.center_eV, st.sigma_eV))
            if abs(c - st.center_eV) > 2.0 * st.sigma_eV:  # ハード境界外は棄却
                ok = False
                break
            cs.append(c)
        if not ok:
            continue
        if cs[0] < cs[1] < cs[2] and min(np.diff(cs)) >= MIN_SEP \
                and X[0] + 1.0 < cs[0] and cs[2] < X[-1] - 1.0:
            break
    return dict(
        cens=tuple(cs),
        fwhm=float(rng.uniform(FWHM_B[0] + 0.1, FWHM_B[1] - 0.1)),
        eta=float(rng.uniform(ETA_B[0] + 0.05, ETA_B[1] - 0.05)),
    )


def _make_spectrum(truth: dict) -> tuple[np.ndarray, dict]:
    """抽選された真値でスペクトルを生成（振幅・背景はTRUTH共通）."""
    t = TRUTH
    a_main = t["amp_main"]
    a_siox = a_main * t["ratio_siox"]
    a_sio2 = (a_main + a_siox) * R_SIO2 / (1.0 - R_SIO2)
    cens, fw, et = truth["cens"], truth["fwhm"], truth["eta"]

    def peak(a: float, c: float) -> np.ndarray:
        return doublet_pseudo_voigt(X, a, c, fw, et, SO_SPLIT, BRANCH)

    peaks = peak(a_main, cens[0]) + peak(a_siox, cens[1]) + peak(a_sio2, cens[2])
    bg = (t["bg_const"] + t["bg_slope"] * (X - X[0])
          + tougaard_from_peaks(X, peaks, t["bg_tougaard_b"]))
    areas = {"area_sio2": doublet_area(a_sio2, fw, et, BRANCH)}
    return np.clip(peaks + bg, 1.0, None), areas


def run(out_dir: Path, mode: str = "calibrated",
        n_rep: int = 60, steps: int = 1200, burn: int = 400) -> None:
    rows = []
    t0 = time.perf_counter()
    for rep in range(n_rep):
        rng = np.random.default_rng(1000 + rep)
        if mode == "calibrated":
            truth = _draw_truth(rng)
        else:  # conflict: NG試料相当の固定真値（事前分布からのずれあり）
            truth = dict(cens=TRUTH["cen"], fwhm=TRUTH["fwhm"], eta=TRUTH["eta"])
        cens = truth["cens"]
        y_true, truth_areas = _make_spectrum(truth)
        y = y_true + rng.normal(0.0, NOISE_S * np.sqrt(y_true))
        spec = Spectrum(X, y)
        fit = fit_components(spec, _components(), background="tougaard", n_starts=4)
        try:
            br = bayesian_uncertainty(fit, steps=steps, burn=burn, thin=4)
        except Exception as e:
            print(f"rep {rep}: emcee失敗 {e}")
            continue

        row = dict(rep=rep, acc=br.acceptance_fraction)

        def record(tag: str, pname: str, truth_val: float) -> None:
            """被覆判定に加え、真値・事後中央値・事後幅を診断用に記録."""
            p16, p50, p84 = br.param_ci[pname]
            row[f"cov_{tag}"] = p16 <= truth_val <= p84
            row[f"truth_{tag}"] = truth_val
            row[f"est_{tag}"] = p50
            row[f"sig_{tag}"] = 0.5 * (p84 - p16)

        record("cen_si0", "Si2p_Si0_cen", cens[0])
        record("cen_siox", "Si2p_SiOx_cen", cens[1])
        record("cen_sio2", "Si2p_SiO2_cen", cens[2])
        record("fwhm", "grp_Si2p_fwhm", truth["fwhm"])
        # 面積（area_ci・真値ともダブレット込み面積で同一定義）
        p16, p50, p84 = br.area_ci["Si2p_SiO2"]
        row["cov_area_sio2"] = p16 <= truth_areas["area_sio2"] <= p84
        row["truth_area_sio2"] = truth_areas["area_sio2"]
        row["est_area_sio2"] = p50
        row["sig_area_sio2"] = 0.5 * (p84 - p16)
        rows.append(row)
        if (rep + 1) % 10 == 0:
            print(f"rep {rep + 1}/{n_rep} ({time.perf_counter() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"coverage_{mode}.csv", index=False)
    n = len(df)
    print(f"\n===== 68%信用区間の被覆率 [{mode}] "
          f"(n={n}, 名目68%, 二項95%CI≈±{1.96*np.sqrt(0.68*0.32/n)*100:.0f}%) =====")
    for tag in ("cen_si0", "cen_siox", "cen_sio2", "fwhm", "area_sio2"):
        err = df[f"est_{tag}"] - df[f"truth_{tag}"]
        sd_est = err.std()
        bias = err.mean()
        sig_mean = df[f"sig_{tag}"].mean()
        print(f"  {tag:<10}: 被覆 {df[f'cov_{tag}'].mean()*100:5.1f}%  "
              f"バイアス {bias:+9.3f}  推定SD {sd_est:9.3f}  "
              f"事後幅σ {sig_mean:9.3f}  (SD/σ = {sd_est/max(sig_mean,1e-12):.2f})")
    print(f"  emcee受容率中央値: {df['acc'].median():.2f}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "outputs" / "validation"
    mode = sys.argv[2] if len(sys.argv) > 2 else "calibrated"
    run(out, mode=mode)
