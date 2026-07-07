"""ケーススタディ: c面GaN初期酸化のO1s系列自動分解（Spring-8 BL23SU, hv=730 eV）.

データ: data2/*.csv（ヘッダなし・第1列KE。非配布・ローカルのみ）
各スペクトルを独立に「化学状態プール×背景」のBIC自動選択で分解し、
成分面積の時間発展（酸化カイネティクス）を無人で抽出する。

O1s化学状態プール（Sumiya et al., J. Phys. Chem. C 124 (2020) 25282 の
帰属に準拠。DFMD計算による裏付けあり）:
- O-O  : 分子性吸着酸素   ~530 eV
- Ga-O : 解離性吸着(Ga-O) ~531 eV
- N-O  : N-O結合          ~532 eV

実験条件（同文献・原田ら先端計測シンポジウム2022 P2-9より）:
O2/Heビーム照射（並進エネルギー2.26 eV, 200°C）中にO1sを逐次測定。
hv=730 eV, パスエネルギー5.0 eV（分解能250 meV）。

実行: python -X utf8 benchmarks/casestudy_gan_oxidation.py [出力ディレクトリ]
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from xps_peakfit import load_spectrum  # noqa: E402
from xps_peakfit.fitting import Component  # noqa: E402
from xps_peakfit.model_select import select_model, subset_candidates  # noqa: E402

REPO = Path(__file__).parent.parent
HV = 730.0
WINDOW = (526.0, 537.0)
FWHM_B = (0.9, 2.2)
ETA_B = (0.0, 0.6)


def pool() -> list[Component]:
    kw = dict(fwhm_bounds=FWHM_B, eta_bounds=ETA_B, fwhm_group="O1s",
              eta_group="O1s")
    return [
        Component(name="O-O", center=530.0, center_sigma=0.5, **kw),
        Component(name="Ga-O", center=531.0, center_sigma=0.5, **kw),
        Component(name="N-O", center=532.2, center_sigma=0.6, **kw),
    ]


def main(out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted((REPO / "data2").glob("*_dsp.csv"))
    if not files:
        sys.exit("data2/ にデータがありません（非配布データ・ローカル専用）")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.perf_counter()
    for f in files:
        step = int(re.search(r"(\d{4})_dsp$", f.stem).group(1))  # 例: 0005→5
        spec = load_spectrum(f, hv=HV).crop(*WINDOW)
        sel = select_model(spec, subset_candidates(pool(), min_size=1),
                           background="auto", n_starts=8)
        best = sel.best
        nprob = sel.n_component_probabilities()
        table = {r["Component"]: r for r in best.peak_table()}
        row = dict(
            file=f.name, step=step,
            n_comp=len(best.components), bg=best.background_kind,
            chi2r=round(best.reduced_chi2, 3),
            P_n=";".join(f"{k}:{v:.2f}" for k, v in nprob.items()),
        )
        for name in ("O-O", "Ga-O", "N-O"):
            r = table.get(name)
            row[f"{name}_cen"] = r["Center_eV"] if r else np.nan
            row[f"{name}_area"] = r["Area"] if r else 0.0
        rows.append(row)
        comps = " + ".join(c.name for c in best.components)
        print(f"{f.name} step={step:>4} n={row['n_comp']} bg={row['bg']:<8} "
              f"chi2r={row['chi2r']:<6} [{comps}]")

    df = pd.DataFrame(rows).sort_values("step")
    df.to_csv(out_dir / "gan_o1s_series.csv", index=False)
    print(f"\n16スペクトル解析: {time.perf_counter() - t0:.1f}s")

    # カイネティクス図
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for name, mk in (("O-O", "o"), ("Ga-O", "s"), ("N-O", "^")):
        axes[0].plot(df["step"], df[f"{name}_area"], mk + "-", ms=5, label=name)
    axes[0].plot(df["step"],
                 df[[f"{n}_area" for n in ("O-O", "Ga-O", "N-O")]].sum(axis=1),
                 "k--", lw=1, label="total")
    axes[0].set_xlabel("Oxidation step (file index)")
    axes[0].set_ylabel("O1s component area (a.u.)")
    axes[0].set_title("GaN initial oxidation kinetics (auto-decomposed)")
    axes[0].legend(fontsize=8)

    axes[1].plot(df["step"], df["Ga-O_cen"], "o-", ms=5, label="Ga-O center")
    axes[1].set_xlabel("Oxidation step (file index)")
    axes[1].set_ylabel("Ga-O center (eV)")
    axes[1].set_title("Chemical shift evolution")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gan_kinetics.png", dpi=130)

    # 代表スペクトルの分解図（初期・中期・最終）
    fig2, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
    picks = [files[0], files[len(files) // 2], files[-1]]
    for ax, f in zip(axs, picks):
        spec = load_spectrum(f, hv=HV).crop(*WINDOW)
        sel = select_model(spec, subset_candidates(pool(), min_size=1),
                           background="auto", n_starts=8)
        b = sel.best
        x, y = b.spectrum.energy, b.spectrum.intensity
        ax.plot(x, y, "k.", ms=3, alpha=0.5)
        ax.plot(x, b.model, "r-", lw=1.3)
        ax.plot(x, b.background, "--", color="gray")
        for name, c in b.curves.items():
            ax.fill_between(x, b.background, b.background + c, alpha=0.35,
                            label=name)
        ax.invert_xaxis()
        ax.legend(fontsize=7)
        ax.set_title(f"{f.stem[-8:]} (chi2r={b.reduced_chi2:.2f}, bg={b.background_kind})")
        ax.set_xlabel("Binding Energy (eV)")
    fig2.tight_layout()
    fig2.savefig(out_dir / "gan_decompositions.png", dpi=130)
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "outputs" / "gan"
    main(out)
