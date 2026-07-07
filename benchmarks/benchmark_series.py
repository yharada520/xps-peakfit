"""系列データ（ピークシフト追跡）でのEMPeaks公平比較.

EMPeaksの本領である「多数スペクトルのピークシフト追跡」タスクでの比較。
オペランド計測を模した合成系列（静止ピーク＋徐々にシフトするピーク）で、
追跡精度（真のシフト軌跡とのRMSE）と処理時間を比較する。

EMPeaks側はガウス混合（同法の原著2019で実証されたモデル）を使用。
xps_peakfit側は前スペクトルの解をウォームスタートに用いる系列モード相当。

実行: python -X utf8 benchmarks/benchmark_series.py [出力ディレクトリ]
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

# EMPeaks 2.1.0 の scipy 互換シム
import scipy.integrate as _si

if not hasattr(_si, "trapz"):
    _si.trapz = np.trapezoid
if not hasattr(_si, "simps"):
    _si.simps = _si.simpson

sys.path.insert(0, str(Path(__file__).parent.parent))

from xps_peakfit.fitting import Component, fit_components  # noqa: E402
from xps_peakfit.io import Spectrum  # noqa: E402
from xps_peakfit.models import pseudo_voigt  # noqa: E402

REPO = Path(__file__).parent.parent

X = np.arange(95.0, 106.0, 0.1)
N_SPECTRA = 25
CEN_STATIC = 98.5
SHIFT_TRUE = np.linspace(100.0, 100.8, N_SPECTRA)  # 徐々に高BE側へ
FWHM_TRUE = 1.5


def make_series(seed: int = 7) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    series = []
    for i in range(N_SPECTRA):
        peaks = (pseudo_voigt(X, 4000.0, CEN_STATIC, FWHM_TRUE, 0.0)
                 + pseudo_voigt(X, 6000.0, SHIFT_TRUE[i], FWHM_TRUE, 0.0))
        y = rng.poisson(peaks + 800.0).astype(float)
        series.append(y)
    return series


def run_empeaks_series(series: list[np.ndarray]) -> tuple[float, np.ndarray]:
    from EMPeaks.GaussianMixture import GaussianMixtureModel

    centers = []
    t0 = time.perf_counter()
    for y in series:
        np.random.seed(0)
        model = GaussianMixtureModel(K=2, x_min=float(X[0]), x_max=float(X[-1]),
                                     background="uniform")
        with contextlib.redirect_stdout(io.StringIO()):
            model.sampling(X, y, trial=3, max_iter=1000, r_eps=1e-7)
        param = model.export_param()
        mu = sorted(float(v) for v in param["mu"])
        centers.append(mu[-1])  # 高BE側=シフトするピーク
    return time.perf_counter() - t0, np.array(centers)


def run_ours_series(series: list[np.ndarray]) -> tuple[float, np.ndarray]:
    centers = []
    prev = (CEN_STATIC, 100.0)
    t0 = time.perf_counter()
    for y in series:
        spec = Spectrum(X, y)
        comps = [
            Component(name="static", center=prev[0], center_sigma=np.inf,
                      fwhm_bounds=(0.8, 2.5), center_window=1.5),
            Component(name="shifting", center=prev[1], center_sigma=np.inf,
                      fwhm_bounds=(0.8, 2.5), center_window=1.5),
        ]
        res = fit_components(spec, comps, background="linear", n_starts=2)
        c_static = res.params["static_cen"].value
        c_shift = res.params["shifting_cen"].value
        prev = (c_static, c_shift)  # ウォームスタート（系列モード）
        centers.append(c_shift)
    return time.perf_counter() - t0, np.array(centers)


def main(out_dir: Path) -> None:
    series = make_series()
    t_emp, cen_emp = run_empeaks_series(series)
    t_ours, cen_ours = run_ours_series(series)

    def rmse(a: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - SHIFT_TRUE) ** 2)))

    def rmse_rel(a: np.ndarray) -> float:
        """相対シフト（初点基準）の追跡誤差。EMPeaksの本来の用途に対応."""
        return float(np.sqrt(np.mean(
            ((a - a[0]) - (SHIFT_TRUE - SHIFT_TRUE[0])) ** 2)))

    print(f"===== ピークシフト追跡（{N_SPECTRA}スペクトル系列, "
          f"真のシフト {SHIFT_TRUE[0]:.1f}→{SHIFT_TRUE[-1]:.1f} eV） =====")
    print(f"  EMPeaks (GMM, K=2)   : {t_emp:6.2f}s "
          f"({t_emp / N_SPECTRA * 1000:6.1f} ms/spec)  "
          f"絶対RMSE={rmse(cen_emp) * 1000:6.1f} meV  "
          f"相対シフトRMSE={rmse_rel(cen_emp) * 1000:5.1f} meV")
    print(f"  xps_peakfit (warm)   : {t_ours:6.2f}s "
          f"({t_ours / N_SPECTRA * 1000:6.1f} ms/spec)  "
          f"絶対RMSE={rmse(cen_ours) * 1000:6.1f} meV  "
          f"相対シフトRMSE={rmse_rel(cen_ours) * 1000:5.1f} meV")
    rmse_emp, rmse_ours = rmse(cen_emp), rmse(cen_ours)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(SHIFT_TRUE, "k-", lw=1, label="truth")
    ax.plot(cen_emp, "s--", ms=4, label=f"EMPeaks (RMSE {rmse_emp*1000:.0f} meV)")
    ax.plot(cen_ours, "o--", ms=4, label=f"xps_peakfit (RMSE {rmse_ours*1000:.0f} meV)")
    ax.set_xlabel("Spectrum index")
    ax.set_ylabel("Tracked center (eV)")
    ax.set_title("Peak-shift tracking (EMPeaks home-turf task)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "series_tracking.png", dpi=130)
    print(f"saved -> {out_dir / 'series_tracking.png'}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "outputs" / "validation"
    main(out)
