"""EMPeaks (spectrum-adapted EM) と xps_peakfit の直接ベンチマーク.

同一データ・同一エネルギー窓で、速度と分解品質（中心位置・幅・面積比）を比較する。

実行: python -X utf8 benchmarks/benchmark_empeaks.py [出力ディレクトリ]

比較設計の注意:
- EMPeaksは成分数Kが固定入力（モデル選択なし）のため、各データに
  「物理的な正解K」を与える（EMPeaks側に有利な条件）。
- EMPeaksの背景は混合成分（linear）。xps_peakfitはactive background。
- 幅の制約は両者おおむね同等になるよう γ∈[0.25, 1.25] ⇔ FWHM∈[0.5, 2.5] に設定。
- EMPeaksにはダブレット拘束がないため、Au4f/Cr2pではスピン軌道
  ペアを独立ピークとしてフィットさせ、分裂幅・面積比の再現性を見る。
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# EMPeaks 2.1.0 は scipy<1.14 の integrate.trapz/simps を参照するため互換シムを当てる
import scipy.integrate as _si

if not hasattr(_si, "trapz"):
    _si.trapz = np.trapezoid
if not hasattr(_si, "simps"):
    _si.simps = _si.simpson

sys.path.insert(0, str(Path(__file__).parent.parent))

from xps_peakfit import load_spectrum  # noqa: E402
from xps_peakfit.fitting import Component, fit_components  # noqa: E402
from xps_peakfit.io import Spectrum  # noqa: E402

REPO = Path(__file__).parent.parent
GAMMA_MIN, GAMMA_MAX = 0.25, 1.25  # EMPeaks γ(HWHM相当) ⇔ FWHM 0.5–2.5 eV
TRIAL = 8  # xps_peakfit の n_starts 上限と同数


@dataclass
class EmpeaksResult:
    time_s: float
    centers: list[float]
    fwhms: list[float]
    area_fracs: list[float]  # ピーク成分内で正規化した面積比
    rmse: float
    curve: np.ndarray
    config: str  # 成功した設定（頑健性の記録）
    failed_configs: list[str]


def _empeaks_once(x: np.ndarray, y: np.ndarray, k: int,
                  background: str) -> tuple[float, object]:
    from EMPeaks.PseudoVoigtMixture import PseudoVoigtMixtureModel

    # EMPeaksのsampling APIにはシード指定がないため、グローバルRNGを固定して
    # 再現性を確保する（これ自体が頑健性比較の知見）
    np.random.seed(0)
    model = PseudoVoigtMixtureModel(
        K=k, x_min=float(x[0]), x_max=float(x[-1]),
        gamma_min=GAMMA_MIN, gamma_max=GAMMA_MAX, background=background,
    )
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model.sampling(x, y, trial=TRIAL, max_iter=2000, r_eps=1e-8)
    return time.perf_counter() - t0, model


def run_empeaks(spec: Spectrum, k: int) -> EmpeaksResult:
    """設定フォールバック付きEMPeaks実行.

    生データ＋背景混合成分で失敗する場合（γ更新のブラケット探索が
    平坦背景で符号反転を見つけられずIndexError）、両端線形背景を
    前引きしたデータで再試行する（EMPeaks系論文の実運用に相当）。
    """
    x, y = spec.energy, spec.intensity
    y_sub = np.clip(y - np.interp(x, [x[0], x[-1]], [y[0], y[-1]]), 0.0, None)
    configs = [
        ("raw+linear_bg", y, "linear"),
        ("raw+uniform_bg", y, "uniform"),
        ("presub+uniform_bg", y_sub, "uniform"),
        ("presub+no_bg", y_sub, "none"),
    ]
    failed: list[str] = []
    for cfg_name, y_use, bg in configs:
        try:
            dt, model = _empeaks_once(x, y_use, k, bg)
        except Exception as e:  # EMPeaks内部エラーは頑健性データとして記録
            failed.append(f"{cfg_name}({type(e).__name__})")
            continue

        param = model.export_param()
        centers = [float(v) for v in param["x0"]]
        fwhms = [2.0 * float(g) for g in param["gamma"]]
        pi = np.asarray(param["pi"], dtype=float)
        pi_peaks = pi[:k]
        fracs = (pi_peaks / pi_peaks.sum()).tolist() if pi_peaks.sum() > 0 else [0.0] * k

        pdf = np.nan_to_num(model.predict(x), nan=0.0, posinf=0.0, neginf=0.0)
        scale = float(np.sum(y_use) / max(np.sum(pdf), 1e-300))  # pdf→強度スケール
        curve = pdf * scale
        if y_use is y_sub:  # 前引き構成は比較のため背景を足し戻す
            curve = curve + np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
        rmse = float(np.sqrt(np.mean((y - curve) ** 2)))
        return EmpeaksResult(dt, centers, fwhms, fracs, rmse, curve,
                             cfg_name, failed)
    raise RuntimeError(f"EMPeaks全設定で失敗: {failed}")


def run_ours(spec: Spectrum, components: list[Component],
             background: str = "tougaard"):
    t0 = time.perf_counter()
    res = fit_components(spec, components, background=background, n_starts=TRIAL)
    dt = time.perf_counter() - t0
    rmse = float(np.sqrt(np.mean((spec.intensity - res.model) ** 2)))
    return dt, res, rmse


def fmt(vals, nd=2):
    return "[" + ", ".join(f"{v:.{nd}f}" for v in vals) + "]"


def main(out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(fwhm_bounds=(0.5, 2.5), eta_bounds=(0.0, 0.6))

    cases = [
        # (タグ, ファイル, 窓, EMPeaksのK, xps_peakfit成分リスト, 我々の背景)
        ("Si2p_NG", "XPS_Si2p_siloxane_NG.csv", (98.0, 104.5), 3,
         [Component.from_line("Si2p", "Si0", **kw),
          Component.from_line("Si2p", "SiOx", **kw),
          Component.from_line("Si2p", "SiO2", **kw)], "tougaard"),
        ("Au4f", "XPS_Au4f.csv", (80.5, 91.5), 2,
         [Component.from_line("Au4f", "Au0", **kw)], "shirley"),
        ("C1s", "XPS_C1s.csv", (282.0, 291.5), 2,
         [Component.from_line("C1s", "C-C", **kw),
          Component.from_line("C1s", "C-O", **kw)], "shirley"),
        ("Cr2p", "XPS_Cr2p.csv", (548.0, 566.0), 2,
         [Component.from_line("Cr2p", "Cr0", name="Cr2p_shifted",
                              center=554.5, center_sigma=1.0,
                              fwhm_bounds=(1.0, 3.0), vary_so=True,
                              so_split_sigma=0.05)], "shirley"),
        ("Ni2p", "XPS_Ni2p.csv", (846.0, 866.0), 4,
         [Component.from_line("Ni2p", s, fwhm_bounds=(0.8, 3.5))
          for s in ("Ni0", "NiO", "Ni(OH)2", "Ni0_sat")], "shirley"),
        ("Si2p_reg", "XPS_Si2p.csv", (96.5, 106.0), 3,
         [Component.from_line("Si2p", s, **kw)
          for s in ("Si0", "SiOx", "SiO2")], "shirley"),
    ]

    # 実測定NGデータは非配布。存在しない環境では合成等価データで代替
    fallback = {"XPS_Si2p_siloxane_NG.csv": "XPS_Si2p_siloxane_synthetic.csv"}

    rows = []
    for tag, fname, window, k_emp, comps, bg in cases:
        path = REPO / "data" / fname
        if not path.exists() and fname in fallback:
            path = REPO / "data" / fallback[fname]
            print(f"[{tag}] 実データ非配布のため合成等価データを使用: {path.name}")
        spec = load_spectrum(path).crop(*window)
        try:
            emp = run_empeaks(spec, k_emp)
        except Exception as e:
            print(f"===== {tag}: EMPeaks全設定失敗 ({e}) =====")
            continue
        t_ours, ours, rmse_ours = run_ours(spec, comps, background=bg)

        print(f"===== {tag} (window {window}, K_EMPeaks={k_emp}) =====")
        fail_note = f"  失敗設定: {emp.failed_configs}" if emp.failed_configs else ""
        print(f"  EMPeaks    : {emp.time_s:6.2f}s  RMSE={emp.rmse:9.1f}  "
              f"[config={emp.config}]{fail_note}")
        print(f"    centers={fmt(sorted(emp.centers))}  FWHM={fmt(sorted(emp.fwhms))}")
        print(f"    area_frac={fmt(sorted(emp.area_fracs, reverse=True))}")
        print(f"  xps_peakfit: {t_ours:6.2f}s  RMSE={rmse_ours:9.1f}  "
              f"(bg={bg}, chi2r={ours.reduced_chi2:.2f})")
        for r in ours.peak_table():
            so = f" SO={r['SO_split_eV']}" if r["SO_split_eV"] else ""
            print(f"    {r['Component']:<16} cen={r['Center_eV']:>8} "
                  f"FWHM={r['FWHM_eV']:>6} area%={r['Area_pct']:>6}{so}")

        rows.append((tag, emp, t_ours, ours, rmse_ours))

        # 比較プロット（描画失敗はベンチ続行を妨げない）
        try:
            x, y = spec.energy, spec.intensity
            emp_curve = np.nan_to_num(emp.curve, nan=0.0, posinf=0.0, neginf=0.0)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
            axes[0].plot(x, y, "k.", ms=3, alpha=0.5)
            axes[0].plot(x, emp_curve, "r-", lw=1.5)
            axes[0].set_title(f"EMPeaks K={k_emp}, {emp.config} "
                              f"({emp.time_s:.2f}s, RMSE={emp.rmse:.0f})")
            axes[1].plot(x, y, "k.", ms=3, alpha=0.5)
            axes[1].plot(x, ours.model, "r-", lw=1.5)
            axes[1].plot(x, ours.background, "--", color="gray")
            for name, c in ours.curves.items():
                axes[1].fill_between(x, ours.background, ours.background + c,
                                     alpha=0.35, label=name)
            axes[1].legend(fontsize=7)
            axes[1].set_title(f"xps_peakfit ({t_ours:.2f}s, RMSE={rmse_ours:.0f})")
            for ax in axes:
                ax.invert_xaxis()
                ax.set_xlabel("Binding Energy (eV)")
            fig.suptitle(tag)
            fig.tight_layout()
            fig.savefig(out_dir / f"bench_{tag}.png", dpi=110)
            plt.close(fig)
        except Exception as e:
            print(f"  (プロット保存失敗: {type(e).__name__}: {e})")
            plt.close("all")

    print("\n===== サマリ（時間・RMSE） =====")
    print(f"{'case':<10} {'EMPeaks[s]':>10} {'ours[s]':>8} {'RMSE_EMP':>10} {'RMSE_ours':>10}")
    for tag, emp, t_ours, ours, rmse_ours in rows:
        print(f"{tag:<10} {emp.time_s:>10.2f} {t_ours:>8.2f} "
              f"{emp.rmse:>10.1f} {rmse_ours:>10.1f}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "outputs" / "benchmark"
    main(out)
