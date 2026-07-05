"""バッチCLI: CSVを解析して結果CSV・プロットPNG・BIC比較表を出力する.

使用例:
    # 物理拘束モード（Si2p全状態 + Auゴースト）
    python -m xps_peakfit.cli data/XPS_Si2p_siloxane_NG.csv \\
        --window 98 104.5 --background tougaard \\
        --line Si2p --ghost Au4f@99.8:0.5

    # 無拘束モード（最大5本の擬Voigt）
    python -m xps_peakfit.cli data/XPS_Si2p.csv --auto-range --generic 5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xps_peakfit.autorange import auto_range
from xps_peakfit.fitting import Component, FitResult
from xps_peakfit.io import load_spectrum
from xps_peakfit.lines import get_line
from xps_peakfit.model_select import (
    generic_candidates,
    select_model,
    subset_candidates,
)

logger = logging.getLogger(__name__)


def parse_ghost(spec_str: str) -> Component:
    """'Au4f@99.8:0.5' 形式のゴースト指定をComponentに変換."""
    try:
        line_key, rest = spec_str.split("@", 1)
        center_s, sigma_s = rest.split(":", 1)
        line = get_line(line_key)
        state = line.states[0].name
        return Component.from_line(
            line_key, state, name=f"{line_key}_ghost",
            center=float(center_s), center_sigma=float(sigma_s),
        )
    except (ValueError, KeyError) as e:
        raise argparse.ArgumentTypeError(
            f"ゴースト指定は 'Line@center:sigma' 形式（例: Au4f@99.8:0.5）: {e}"
        ) from e


def build_pool(
    line_keys: list[str],
    ghosts: list[Component],
    fwhm_bounds: tuple[float, float],
    eta_max: float,
) -> list[Component]:
    kw = dict(fwhm_bounds=fwhm_bounds, eta_bounds=(0.0, eta_max))
    pool: list[Component] = []
    for key in line_keys:
        line = get_line(key)
        for st in line.states:
            pool.append(Component.from_line(key, st.name, **kw))
    for g in ghosts:
        pool.append(Component(
            name=g.name, center=g.center, center_sigma=g.center_sigma,
            so_split=g.so_split, branch_ratio=g.branch_ratio,
            fwhm_group=g.fwhm_group, eta_group=g.eta_group, **kw,
        ))
    return pool


def save_outputs(result: FitResult, sel_summary: list[dict], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.peak_table()).to_csv(out_dir / f"{stem}_peaks.csv", index=False)
    pd.DataFrame(sel_summary).to_csv(out_dir / f"{stem}_bic.csv", index=False)

    x, y = result.spectrum.energy, result.spectrum.intensity
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True, height_ratios=[3, 1]
    )
    ax.plot(x, y, "k.", ms=3, alpha=0.5, label="raw")
    ax.plot(x, result.model, "r-", lw=1.5, label="fit")
    ax.plot(x, result.background, "--", color="gray",
            label=f"bg ({result.background_kind})")
    for name, c in result.curves.items():
        ax.fill_between(x, result.background, result.background + c,
                        alpha=0.35, label=name)
    ax.set_ylabel("Intensity (a.u.)")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.set_title(f"{stem}: {' + '.join(c.name for c in result.components)} "
                 f"(BIC={result.bic:.1f}, chi2r={result.reduced_chi2:.2f})")
    axr.axhline(0, color="gray", lw=0.5)
    axr.plot(x, y - result.model, "b.", ms=3)
    axr.set_xlabel("Binding Energy (eV)")
    axr.set_ylabel("residual")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_fit.png", dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="XPS自動ピーク分離（BIC本数選択 + MAP擬Voigtフィット）")
    ap.add_argument("csv", nargs="+", help="入力CSV（energy/int列）")
    ap.add_argument("--window", nargs=2, type=float, metavar=("EMIN", "EMAX"),
                    help="フィット範囲 (eV)")
    ap.add_argument("--auto-range", action="store_true",
                    help="フィット範囲を自動検出")
    ap.add_argument("--background", default="auto",
                    choices=("auto", "shirley", "tougaard", "linear"),
                    help="auto: shirley/tougaardの両方をBICで比較して自動選択")
    ap.add_argument("--line", action="append", default=[],
                    help="ラインDBキー（例: Si2p）。全化学状態が候補プールに入る")
    ap.add_argument("--ghost", action="append", default=[], type=parse_ghost,
                    help="ゴースト成分 'Line@center:sigma'（例: Au4f@99.8:0.5）")
    ap.add_argument("--generic", type=int, default=0, metavar="N",
                    help="無拘束モード: 最大N本の独立擬Voigt")
    ap.add_argument("--fwhm", nargs=2, type=float, default=(1.2, 2.4),
                    metavar=("MIN", "MAX"), help="FWHM範囲 (eV)")
    ap.add_argument("--eta-max", type=float, default=0.6,
                    help="擬Voigt混合比ηの上限")
    ap.add_argument("--n-starts", type=int, default=8, help="マルチスタート回数")
    ap.add_argument("--min-components", type=int, default=1,
                    help="候補構成の最小成分数")
    ap.add_argument("--out", type=Path, default=Path("outputs"), help="出力先")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.line and not args.ghost and args.generic <= 0:
        ap.error("--line / --ghost / --generic のいずれかを指定してください")

    for path_str in args.csv:
        path = Path(path_str)
        spec = load_spectrum(path)
        if args.window:
            window = (args.window[0], args.window[1])
        elif args.auto_range:
            window = auto_range(spec)
            print(f"[auto-range] {path.name}: window = ({window[0]:.2f}, {window[1]:.2f}) eV")
        else:
            window = (float(spec.energy[0]), float(spec.energy[-1]))
        sub = spec.crop(*window)

        if args.generic > 0:
            candidates = generic_candidates(sub, max_peaks=args.generic,
                                            fwhm_bounds=tuple(args.fwhm))
        else:
            pool = build_pool(args.line, args.ghost, tuple(args.fwhm), args.eta_max)
            candidates = subset_candidates(pool, min_size=args.min_components)

        sel = select_model(sub, candidates, background=args.background,
                           n_starts=args.n_starts)
        best = sel.best
        print(f"\n=== {path.name} | window=({window[0]:g}, {window[1]:g}) eV "
              f"| bg={args.background} ===")
        print(f"BEST: {' + '.join(c.name for c in best.components)} "
              f"| bg={best.background_kind} "
              f"(BIC={best.bic:.1f}, chi2r={best.reduced_chi2:.2f})")
        for r in best.peak_table():
            print(f"  {r['Component']:<16} cen={r['Center_eV']:>8.3f} eV  "
                  f"FWHM={r['FWHM_eV']:>5.2f}  area%={r['Area_pct']:>6.2f}")
        runners = [row for row in sel.summary()[1:] if row["dBIC"] < 10.0]
        if runners:
            print("有力な代替候補 (dBIC<10):")
            for row in runners:
                print(f"  dBIC={row['dBIC']:>5.1f}  {row['Components']} "
                      f"[{row['Background']}]")

        save_outputs(best, sel.summary(), args.out, path.stem)
        print(f"出力: {args.out / (path.stem + '_peaks.csv')} ほか")
    return 0


if __name__ == "__main__":
    sys.exit(main())
