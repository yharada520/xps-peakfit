"""XPSラインデータベース（スピン軌道分裂・化学シフト事前分布）.

@register デコレータによるレジストリパターン。元素非依存の汎用設計で、
Si専用等のハードコードを排除する。値は標準的な文献値（Al Kα, C1s=284.8 eV基準）。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChemicalState:
    """化学状態1つ分の事前分布: 中心位置 μ ± σ (eV)."""

    name: str
    center_eV: float
    sigma_eV: float


@dataclass(frozen=True)
class LineShape:
    """1本の光電子ラインの物理定数.

    so_split_eV:  スピン軌道分裂幅（副成分は高束縛エネルギー側）
    branch_ratio: 副成分/主成分の面積比（例: 2p1/2:2p3/2 = 1:2 → 0.5）
    シングレット（s軌道等）は so_split_eV=0, branch_ratio=0。
    """

    element: str
    orbital: str
    so_split_eV: float
    branch_ratio: float
    states: tuple[ChemicalState, ...] = field(default_factory=tuple)

    @property
    def key(self) -> str:
        return f"{self.element}{self.orbital}"

    def state(self, name: str) -> ChemicalState:
        for s in self.states:
            if s.name == name:
                return s
        raise KeyError(f"{self.key} に化学状態 '{name}' は登録されていません: "
                       f"{[s.name for s in self.states]}")


LINE_REGISTRY: dict[str, LineShape] = {}


def register(line: LineShape) -> LineShape:
    """ラインをレジストリに登録するデコレータ的ヘルパー."""
    LINE_REGISTRY[line.key] = line
    return line


def get_line(key: str) -> LineShape:
    if key not in LINE_REGISTRY:
        raise KeyError(f"未登録のライン: {key}（登録済み: {sorted(LINE_REGISTRY)}）")
    return LINE_REGISTRY[key]


# =========================
# 初期登録ライン
# =========================
register(LineShape(
    element="Si", orbital="2p", so_split_eV=0.61, branch_ratio=0.5,
    states=(
        ChemicalState("Si0", 99.4, 0.4),        # 元素状Si
        ChemicalState("SiOx", 101.5, 0.8),      # サブオキサイド/シロキサン
        ChemicalState("SiO2", 103.3, 0.7),
    ),
))

register(LineShape(
    element="Au", orbital="4f", so_split_eV=3.67, branch_ratio=0.75,
    states=(
        ChemicalState("Au0", 84.0, 0.3),
    ),
))

register(LineShape(
    element="C", orbital="1s", so_split_eV=0.0, branch_ratio=0.0,
    states=(
        ChemicalState("C-C", 284.8, 0.3),
        ChemicalState("C-O", 286.3, 0.4),
        ChemicalState("C=O", 288.0, 0.5),
        ChemicalState("O-C=O", 289.0, 0.5),
    ),
))

register(LineShape(
    element="O", orbital="1s", so_split_eV=0.0, branch_ratio=0.0,
    states=(
        ChemicalState("metal-oxide", 530.1, 0.5),
        ChemicalState("SiO2/organic", 532.5, 0.7),
    ),
))
