"""スペクトルの読み込みと前処理."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 列名の柔軟判定（小文字・空白除去後に照合）
_ENERGY_ALIASES = {"energy", "bindingenergy", "be", "e", "ev"}
_INTENSITY_ALIASES = {"int", "intensity", "counts", "cps", "y"}


@dataclass(frozen=True)
class Spectrum:
    """XPSスペクトル1本分（束縛エネルギー昇順で保持）."""

    energy: np.ndarray
    intensity: np.ndarray
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        e = np.asarray(self.energy, dtype=float)
        i = np.asarray(self.intensity, dtype=float)
        if e.shape != i.shape or e.ndim != 1:
            raise ValueError("energy と intensity は同じ長さの1次元配列が必要です")
        if len(e) < 5:
            raise ValueError("データ点数が少なすぎます (>=5 点必要)")
        order = np.argsort(e)
        object.__setattr__(self, "energy", e[order])
        object.__setattr__(self, "intensity", i[order])

    def crop(self, emin: float, emax: float) -> "Spectrum":
        """束縛エネルギー範囲 [emin, emax] を切り出す."""
        mask = (self.energy >= emin) & (self.energy <= emax)
        if mask.sum() < 5:
            raise ValueError(f"範囲 [{emin}, {emax}] eV 内のデータ点が不足しています")
        return Spectrum(self.energy[mask], self.intensity[mask], self.name, dict(self.meta))

    @property
    def step(self) -> float:
        """エネルギー刻みの中央値 (eV)."""
        return float(np.median(np.diff(self.energy)))


def _is_headerless(path: Path) -> bool:
    """先頭行が全て数値ならヘッダなしCSVと判定."""
    with open(path, encoding="utf-8-sig") as f:
        first = f.readline()
    try:
        [float(tok) for tok in first.replace("\t", ",").split(",") if tok.strip()]
        return True
    except ValueError:
        return False


def load_spectrum(path: str | Path, hv: float | None = None) -> Spectrum:
    """CSVからスペクトルを読み込む。列名は energy/int 系のエイリアスを柔軟判定.

    Args:
        path: CSVパス。ヘッダ行がない（先頭行が数値のみの）ファイルにも対応
        hv: 光子エネルギー (eV)。指定すると第1列を運動エネルギーとみなし
            束縛エネルギー BE = hv - KE に変換する（シンクロトロン計測用）
    """
    path = Path(path)
    try:
        if _is_headerless(path):
            df = pd.read_csv(path, header=None, names=["energy", "int"],
                             usecols=[0, 1])
        else:
            df = pd.read_csv(path)
    except Exception:
        logger.exception("CSV読み込みに失敗: %s", path)
        raise
    spec = spectrum_from_dataframe(df, name=path.stem)
    if hv is not None:
        spec = Spectrum(hv - spec.energy, spec.intensity, name=spec.name,
                        meta={**spec.meta, "hv_eV": float(hv), "axis": "KE->BE"})
    return spec


def spectrum_from_dataframe(df: pd.DataFrame, name: str = "") -> Spectrum:
    """DataFrameからスペクトルを構築（列名エイリアス柔軟判定）."""
    e_col = i_col = None
    for c in df.columns:
        lc = str(c).strip().lower().replace(" ", "").replace("\t", "")
        if e_col is None and lc in _ENERGY_ALIASES:
            e_col = c
        elif i_col is None and lc in _INTENSITY_ALIASES:
            i_col = c
    if e_col is None or i_col is None:
        # フォールバック: 数値列の先頭2列
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) >= 2:
            e_col = e_col or num_cols[0]
            i_col = i_col or (num_cols[1] if num_cols[1] != e_col else num_cols[0])
        else:
            raise ValueError(
                f"エネルギー/強度の列が見つかりません: {list(df.columns)}"
            )

    sub = df[[e_col, i_col]].dropna()
    return Spectrum(sub[e_col].to_numpy(float), sub[i_col].to_numpy(float), name=name)
