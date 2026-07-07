"""GaN初期酸化系列（BL23SU, 公開データ）の回帰テスト.

化学状態の帰属は Sumiya et al., J. Phys. Chem. C 124 (2020) 25282 に準拠:
O-O（分子性吸着, ~530 eV）/ Ga-O（解離性吸着, ~531 eV）/ N-O（~532 eV）。
"""
from pathlib import Path

import pytest

from xps_peakfit import load_spectrum, select_model
from xps_peakfit.fitting import Component
from xps_peakfit.model_select import subset_candidates

DATA2 = Path(__file__).parent.parent / "data2"
HV = 730.0
WINDOW = (526.0, 537.0)


def _pool() -> list[Component]:
    kw = dict(fwhm_bounds=(0.9, 2.2), eta_bounds=(0.0, 0.6),
              fwhm_group="O1s", eta_group="O1s")
    return [
        Component(name="O-O", center=530.0, center_sigma=0.5, **kw),
        Component(name="Ga-O", center=531.0, center_sigma=0.5, **kw),
        Component(name="N-O", center=532.2, center_sigma=0.6, **kw),
    ]


@pytest.mark.slow
def test_gan_final_spectrum_three_states() -> None:
    """最終スペクトル(step340): 3状態分解、Ga-Oが最大成分."""
    spec = load_spectrum(DATA2 / "1804100340_dsp.csv", hv=HV).crop(*WINDOW)
    sel = select_model(spec, subset_candidates(_pool(), min_size=1),
                       background="auto")
    table = {r["Component"]: r for r in sel.best.peak_table()}
    assert len(table) == 3
    assert 529.7 <= table["O-O"]["Center_eV"] <= 530.4
    assert 530.7 <= table["Ga-O"]["Center_eV"] <= 531.5
    assert table["Ga-O"]["Area"] > table["O-O"]["Area"]  # 後期は解離吸着が優位
    assert sel.best.reduced_chi2 < 1.5


@pytest.mark.slow
def test_gan_initial_spectrum_molecular_dominant() -> None:
    """初期スペクトル(step5): 分子性吸着O-Oが主体、Ga-Oは未出現.

    Sumiya et al.の「分子性吸着→解離性吸着」の遷移描像と整合すること。
    """
    spec = load_spectrum(DATA2 / "1804100005_dsp.csv", hv=HV).crop(*WINDOW)
    sel = select_model(spec, subset_candidates(_pool(), min_size=1),
                       background="auto")
    table = {r["Component"]: r for r in sel.best.peak_table()}
    assert "O-O" in table
    assert table["O-O"]["Area_pct"] > 80.0
    assert "Ga-O" not in table  # 解離性吸着はまだBIC検出されない
