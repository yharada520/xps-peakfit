"""XPS自動ピーク分離 Streamlit GUI.

起動: streamlit run app_streamlit.py
"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from xps_peakfit import __version__
from xps_peakfit.autorange import auto_range
from xps_peakfit.fitting import Component, FitResult
from xps_peakfit.io import Spectrum, spectrum_from_dataframe  # noqa: F401 (型注釈用)
from xps_peakfit.lines import LINE_REGISTRY, get_line
from xps_peakfit.model_select import (
    generic_candidates,
    select_model,
    subset_candidates,
)

st.set_page_config(page_title="XPS Peak Fit", layout="wide")
st.title("XPS自動ピーク分離")
st.caption(f"xps_peakfit v{__version__} — BIC本数選択 + MAP擬Voigtフィット "
           "+ active background (Shirley/Tougaard)")


@st.cache_data(show_spinner=False)
def _load(file_bytes: bytes, name: str) -> Spectrum:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return spectrum_from_dataframe(df, name=name)


def _fit_figure(result: FitResult) -> plt.Figure:
    x, y = result.spectrum.energy, result.spectrum.intensity
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 6),
                                  sharex=True, height_ratios=[3, 1])
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
    axr.axhline(0, color="gray", lw=0.5)
    axr.plot(x, y - result.model, "b.", ms=3)
    axr.set_xlabel("Binding Energy (eV)")
    axr.set_ylabel("residual")
    fig.tight_layout()
    return fig


# ---------------- サイドバー ----------------
with st.sidebar:
    st.header("1. データ")
    up = st.file_uploader("CSV（energy/int 列）", type=["csv"])
    if up is not None:
        spec = _load(up.getvalue(), up.name)
    else:
        from pathlib import Path
        samples = sorted(Path("data").glob("*.csv")) if Path("data").is_dir() else []
        if not samples:
            st.info("CSVをアップロードしてください")
            st.stop()
        sample = st.selectbox("またはサンプルデータを選択",
                              [p.name for p in samples])
        spec = _load((Path("data") / sample).read_bytes(), Path(sample).stem)
    st.success(f"{spec.name}: {len(spec.energy)}点 "
               f"[{spec.energy[0]:.1f}, {spec.energy[-1]:.1f}] eV")

    st.header("2. フィット範囲")
    auto_win = auto_range(spec)
    e_lo, e_hi = float(spec.energy[0]), float(spec.energy[-1])
    window = st.slider(
        "範囲 (eV) — 初期値は自動検出", min_value=e_lo, max_value=e_hi,
        value=(max(auto_win[0], e_lo), min(auto_win[1], e_hi)),
        step=float(spec.step),
    )

    st.header("3. モデル")
    mode = st.radio("モード", ["物理拘束（ラインDB）", "無拘束（汎用擬Voigt）"])
    background = st.selectbox(
        "背景", ["auto", "tougaard", "shirley", "linear"],
        help="auto: shirley/tougaardの両方をBICで比較して自動選択。"
             "金属基板上の急峻な背景にはTougaardが選ばれやすい",
    )

    pool: list[Component] = []
    max_generic = 5
    fwhm_b = st.slider("FWHM範囲 (eV)", 0.4, 4.0, (1.2, 2.4), 0.1)
    shape = st.selectbox(
        "ピーク形状", ["pvoigt", "doniach"],
        help="doniach: 金属ピーク向け非対称Doniach-Šunjić。η枠が非対称度(推奨上限0.3)になる",
    )
    eta_label = "η上限（ローレンツ混合比）" if shape == "pvoigt" else "非対称度上限"
    eta_max = st.slider(eta_label, 0.0, 1.0, 0.6 if shape == "pvoigt" else 0.25, 0.05)
    vary_so = st.checkbox(
        "スピン軌道定数の微変動を許可（タイト事前分布）", value=False,
        help="分裂幅±0.02 eV・分岐比±0.02のガウス事前分布付きで微調整（篠塚2024方式）",
    )
    kw = dict(fwhm_bounds=fwhm_b, eta_bounds=(0.0, eta_max),
              shape=shape, vary_so=vary_so)

    if mode.startswith("物理"):
        keys = st.multiselect("ライン", sorted(LINE_REGISTRY),
                              default=["Si2p"] if "Si2p" in LINE_REGISTRY else [])
        for key in keys:
            line = get_line(key)
            states = st.multiselect(
                f"{key} の化学状態", [s.name for s in line.states],
                default=[s.name for s in line.states], key=f"st_{key}",
            )
            for sname in states:
                pool.append(Component.from_line(key, sname, **kw))
        with st.expander("ゴースト成分を追加"):
            g_on = st.checkbox("有効化")
            g_line = st.selectbox("形状の借用元ライン", sorted(LINE_REGISTRY),
                                  help="スピン軌道分裂・面積比をDBから借用")
            g_cen = st.number_input("中心の事前平均 (eV)", value=99.8, step=0.1)
            g_sig = st.number_input("事前σ (eV)", value=0.5, step=0.1, min_value=0.1)
            if g_on:
                gl = get_line(g_line)
                pool.append(Component.from_line(
                    g_line, gl.states[0].name, name=f"{g_line}_ghost",
                    center=float(g_cen), center_sigma=float(g_sig), **kw,
                ))
        min_comp = st.number_input("最小成分数", 1, max(len(pool), 1), 1)
    else:
        max_generic = st.number_input("最大ピーク本数", 1, 8, 5)

    n_starts = st.number_input("マルチスタート回数", 1, 32, 8)
    run = st.button("解析実行", type="primary",
                    disabled=(mode.startswith("物理") and not pool))

# ---------------- メイン ----------------
sub = spec.crop(*window)
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("スペクトル")
    fig0, ax0 = plt.subplots(figsize=(7, 3.2))
    ax0.plot(spec.energy, spec.intensity, "k-", lw=0.8, alpha=0.5)
    ax0.plot(sub.energy, sub.intensity, "b-", lw=1.2)
    ax0.axvspan(window[0], window[1], alpha=0.08, color="blue")
    ax0.set_xlabel("Binding Energy (eV)")
    ax0.set_ylabel("Intensity")
    ax0.invert_xaxis()
    st.pyplot(fig0, clear_figure=True)

if run:
    if mode.startswith("物理"):
        candidates = subset_candidates(pool, min_size=int(min_comp))
    else:
        candidates = generic_candidates(sub, max_peaks=int(max_generic),
                                        fwhm_bounds=fwhm_b)
    with st.spinner(f"{len(candidates)}構成をフィット中..."):
        sel = select_model(sub, candidates, background=background,
                           n_starts=int(n_starts))
    st.session_state["sel"] = sel

if "sel" in st.session_state:
    sel = st.session_state["sel"]
    best = sel.best

    with col1:
        st.subheader("ベストフィット")
        st.pyplot(_fit_figure(best), clear_figure=True)

    with col2:
        st.subheader("ピークパラメータ")
        peaks_df = pd.DataFrame(best.peak_table())
        st.dataframe(peaks_df, use_container_width=True)
        st.metric("BIC", f"{best.bic:.1f}",
                  help="χ²+事前ペナルティ+パラメータ数ペナルティ")
        st.metric("χ²_reduced", f"{best.reduced_chi2:.3f}")

        st.subheader("BIC比較（モデル選択）")
        nprobs = sel.n_component_probabilities()
        st.write("成分数の近似事後確率: "
                 + ", ".join(f"**n={n}**: {p:.1%}" for n, p in nprobs.items()))
        bic_df = pd.DataFrame(sel.summary())
        st.dataframe(bic_df, use_container_width=True)
        n_alt = (bic_df["dBIC"] < 10).sum() - 1
        if n_alt > 0:
            st.warning(f"ΔBIC<10 の代替候補が{n_alt}件あります。"
                       "本数の断定には注意してください。")

        st.download_button("ピークCSVをダウンロード",
                           peaks_df.to_csv(index=False).encode(),
                           file_name=f"{spec.name}_peaks.csv")
        st.download_button("BIC比較CSVをダウンロード",
                           bic_df.to_csv(index=False).encode(),
                           file_name=f"{spec.name}_bic.csv")

        with st.expander("ベイズ不確かさ定量化（emcee, 数十秒）"):
            steps = st.number_input("MCMCステップ数", 500, 10000, 1500, 500)
            if st.button("MCMC実行"):
                from xps_peakfit.uncertainty import bayesian_uncertainty
                with st.spinner("emceeサンプリング中..."):
                    br = bayesian_uncertainty(best, steps=int(steps))
                st.caption(f"受容率 {br.acceptance_fraction:.2f} / "
                           f"サンプル数 {br.n_samples}")
                st.dataframe(pd.DataFrame(br.table), use_container_width=True)
