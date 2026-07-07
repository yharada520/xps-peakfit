"""モンテカルロ検証: 微弱成分の検出確率・偽陽性率・面積復元精度.

論文用の系統検証。NGデータを模した3成分＋Tougaard背景の真値モデルから
「SiO2相当の微弱成分の面積比 r」×「ノイズスケール s」のグリッドで
合成スペクトルを多数生成し、BICモデル選択の統計的性能を評価する。

- 検出確率: 真値 r>0 のとき3成分モデルがBIC選択される割合
- 偽陽性率: 真値 r=0 のとき誤って3成分が選択される割合
- 復元精度: 検出時の中心・面積の誤差分布

実行: python -X utf8 benchmarks/validation_montecarlo.py [出力ディレクトリ]
所要: 既定グリッド（6r×4s×30回）で10分前後
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from xps_peakfit.background import tougaard_from_peaks  # noqa: E402
from xps_peakfit.fitting import Component, fit_components  # noqa: E402
from xps_peakfit.io import Spectrum  # noqa: E402
from xps_peakfit.models import doublet_area, doublet_pseudo_voigt  # noqa: E402

SO_SPLIT, BRANCH = 0.61, 0.5  # Si2p ダブレット定数（フィットモデルと同一物理）

REPO = Path(__file__).parent.parent

# ---- 真値モデル（NGデータ規模を模した完全合成） ----
X = np.arange(98.0, 104.55, 0.1)
TRUTH = dict(
    cen=(99.9, 101.4, 102.7),   # Si0系 / SiOx / SiO2相当
    fwhm=2.3, eta=0.3,
    amp_main=26000.0,            # 主成分の高さ
    ratio_siox=0.22,             # SiOx面積 / 主成分面積
    bg_const=76000.0, bg_slope=-300.0, bg_tougaard_b=120.0,
)
FWHM_B = (1.2, 2.4)
ETA_B = (0.0, 0.6)


def make_truth(r_sio2: float) -> tuple[np.ndarray, dict]:
    """r_sio2 = SiO2面積 / 全ピーク面積 となる真値スペクトル（ノイズなし）."""
    t = TRUTH
    a_main = t["amp_main"]
    a_siox = a_main * t["ratio_siox"]
    # 面積比→高さ換算（FWHM/η共通なので高さ比=面積比）
    denom = 1.0 - r_sio2
    a_sio2 = (a_main + a_siox) * r_sio2 / denom if r_sio2 > 0 else 0.0
    # フィットモデルと同一物理（Si2pダブレット）で真値を生成する
    def peak(a: float, c: float) -> np.ndarray:
        return doublet_pseudo_voigt(X, a, c, t["fwhm"], t["eta"], SO_SPLIT, BRANCH)

    peaks = peak(a_main, t["cen"][0]) + peak(a_siox, t["cen"][1]) + peak(a_sio2, t["cen"][2])
    bg = (t["bg_const"] + t["bg_slope"] * (X - X[0])
          + tougaard_from_peaks(X, peaks, t["bg_tougaard_b"]))
    truth_areas = {
        "area_sio2": doublet_area(a_sio2, t["fwhm"], t["eta"], BRANCH),
        "area_siox": doublet_area(a_siox, t["fwhm"], t["eta"], BRANCH),
    }
    return np.clip(peaks + bg, 1.0, None), truth_areas


def candidates() -> tuple[list[Component], list[Component]]:
    kw = dict(fwhm_bounds=FWHM_B, eta_bounds=ETA_B)
    c2 = [Component.from_line("Si2p", "Si0", **kw),
          Component.from_line("Si2p", "SiOx", **kw)]
    c3 = c2 + [Component.from_line("Si2p", "SiO2", **kw)]
    return c2, c3


def run(out_dir: Path, r_grid=(0.0, 0.03, 0.06, 0.09, 0.14, 0.20),
        s_grid=(0.4, 0.7, 1.0, 1.5), n_rep: int = 30) -> None:
    rows = []
    t_start = time.perf_counter()
    for r in r_grid:
        y_true, truth_areas = make_truth(r)
        for s in s_grid:
            for rep in range(n_rep):
                rng = np.random.default_rng(hash((round(r * 100), round(s * 10), rep)) % 2**32)
                y = y_true + rng.normal(0.0, s * np.sqrt(y_true))
                spec = Spectrum(X, y)
                c2, c3 = candidates()
                f2 = fit_components(spec, c2, background="tougaard", n_starts=4)
                f3 = fit_components(spec, c3, background="tougaard", n_starts=4)
                detected = f3.bic < f2.bic
                dbic = f2.bic - f3.bic
                rec = dict(r_sio2=r, noise_s=s, rep=rep, detected=detected, dBIC=dbic)
                if detected:
                    tab = {t["Component"]: t for t in f3.peak_table()}
                    rec["cen_err_sio2"] = tab["Si2p_SiO2"]["Center_eV"] - TRUTH["cen"][2]
                    if truth_areas["area_sio2"] > 0:
                        rec["area_relerr_sio2"] = (
                            tab["Si2p_SiO2"]["Area"] / truth_areas["area_sio2"] - 1.0)
                rows.append(rec)
        print(f"r={r:.2f} done ({time.perf_counter() - t_start:.0f}s)")

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "mc_raw.csv", index=False)

    # 集計: 検出確率（r=0行は偽陽性率）
    summary = (df.groupby(["r_sio2", "noise_s"])
               .agg(detect_rate=("detected", "mean"),
                    dBIC_median=("dBIC", "median"),
                    cen_err_std=("cen_err_sio2", "std"),
                    area_relerr_mean=("area_relerr_sio2", "mean"),
                    area_relerr_std=("area_relerr_sio2", "std"),
                    n=("detected", "size"))
               .reset_index())
    summary.to_csv(out_dir / "mc_summary.csv", index=False)
    print(summary.to_string(index=False))

    # 図: 検出確率曲線
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for s in s_grid:
        sub = summary[summary["noise_s"] == s]
        axes[0].plot(sub["r_sio2"] * 100, sub["detect_rate"] * 100,
                     "o-", label=f"noise s={s}")
    axes[0].axhline(50, color="gray", ls=":", lw=0.8)
    axes[0].set_xlabel("True SiO2 area fraction (%)")
    axes[0].set_ylabel("Detection rate (%)")
    axes[0].set_title("Detection probability (r=0: false-positive rate)")
    axes[0].legend(fontsize=8)

    det = df[(df["detected"]) & (df["r_sio2"] > 0)]
    for s in s_grid:
        sub = det[det["noise_s"] == s]
        g = sub.groupby("r_sio2")["area_relerr_sio2"]
        axes[1].errorbar(g.mean().index * 100, g.mean() * 100,
                         yerr=g.std() * 100, fmt="o-", capsize=3,
                         label=f"noise s={s}")
    axes[1].axhline(0, color="gray", ls=":", lw=0.8)
    axes[1].set_xlabel("True SiO2 area fraction (%)")
    axes[1].set_ylabel("SiO2 area relative error (%)")
    axes[1].set_title("Area recovery (when detected)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "mc_detection.png", dpi=130)
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "outputs" / "validation"
    run(out)
