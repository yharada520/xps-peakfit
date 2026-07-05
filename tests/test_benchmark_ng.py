"""実データベンチマーク: Au電極上シロキサンのSi2p（最難関データ）.

正解（ドメイン知識）:
- 99–103 eV に3〜4本
- 最高束縛エネルギー側 (102–103 eV) が SiO2
- ~101 eV が SiOx（シロキサン）
- それ以外は Au ゴーストまたは元素状 Si2p
"""
from pathlib import Path

import pytest

from xps_peakfit import load_spectrum, select_model
from xps_peakfit.fitting import Component
from xps_peakfit.model_select import subset_candidates

DATA = Path(__file__).parent.parent / "data"

FWHM_B = (1.2, 2.4)
ETA_B = (0.0, 0.6)


def _pool() -> list[Component]:
    kw = dict(fwhm_bounds=FWHM_B, eta_bounds=ETA_B)
    return [
        Component.from_line("Au4f", "Au0", name="Au4f_ghost",
                            center=99.8, center_sigma=0.5, **kw),
        Component.from_line("Si2p", "Si0", **kw),
        Component.from_line("Si2p", "SiOx", **kw),
        Component.from_line("Si2p", "SiO2", **kw),
    ]


@pytest.mark.slow
def test_ng_data_selects_physical_solution() -> None:
    """NGデータ: Tougaard背景でSiOx/SiO2を含む3〜4成分解が選択されること."""
    spec = load_spectrum(DATA / "XPS_Si2p_siloxane_NG.csv").crop(98.0, 104.5)
    sel = select_model(spec, subset_candidates(_pool(), min_size=2),
                       background="tougaard", n_starts=8)
    best = sel.best
    assert 3 <= len(best.components) <= 4

    table = {r["Component"]: r for r in best.peak_table()}
    assert "Si2p_SiO2" in table, f"SiO2成分が未選択: {list(table)}"
    assert "Si2p_SiOx" in table, f"SiOx成分が未選択: {list(table)}"
    assert 102.0 <= table["Si2p_SiO2"]["Center_eV"] <= 103.5
    assert 100.8 <= table["Si2p_SiOx"]["Center_eV"] <= 101.8
    # 残りの主成分（ゴースト or Si0）は 99–100.5 eV
    others = [r for n, r in table.items() if n not in ("Si2p_SiO2", "Si2p_SiOx")]
    assert others and all(99.0 <= r["Center_eV"] <= 100.5 for r in others)
    assert best.reduced_chi2 < 1.5


@pytest.mark.slow
def test_regular_data_sio2_dominant() -> None:
    """通常データ: SiO2主体（>90%）の3成分解に収束すること（回帰確認）."""
    spec = load_spectrum(DATA / "XPS_Si2p.csv").crop(96.5, 106.0)
    sel = select_model(spec, subset_candidates(_pool()[1:], min_size=1),
                       background="shirley", n_starts=8)
    table = {r["Component"]: r for r in sel.best.peak_table()}
    assert "Si2p_SiO2" in table
    assert table["Si2p_SiO2"]["Area_pct"] > 90.0
    assert 103.0 <= table["Si2p_SiO2"]["Center_eV"] <= 104.5
