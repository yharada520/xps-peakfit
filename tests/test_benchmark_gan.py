"""GaN初期酸化系列（BL23SU, 公開データ）の回帰テスト."""
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
        Component(name="O_GaO", center=530.6, center_sigma=0.5, **kw),
        Component(name="O_OH", center=532.0, center_sigma=0.6, **kw),
        Component(name="O_H2O", center=533.3, center_sigma=0.6, **kw),
    ]


@pytest.mark.slow
def test_gan_final_spectrum_three_states() -> None:
    """最終スペクトル(step340): 3状態分解、Ga-O中心が530 eV近傍."""
    spec = load_spectrum(DATA2 / "1804100340_dsp.csv", hv=HV).crop(*WINDOW)
    sel = select_model(spec, subset_candidates(_pool(), min_size=1),
                       background="auto")
    table = {r["Component"]: r for r in sel.best.peak_table()}
    assert len(table) == 3
    assert 529.7 <= table["O_GaO"]["Center_eV"] <= 530.6
    assert 530.8 <= table["O_OH"]["Center_eV"] <= 531.8
    assert sel.best.reduced_chi2 < 1.5


@pytest.mark.slow
def test_gan_initial_spectrum_gao_dominant() -> None:
    """初期スペクトル(step5): Ga-O主体（OHはまだ小さい）."""
    spec = load_spectrum(DATA2 / "1804100005_dsp.csv", hv=HV).crop(*WINDOW)
    sel = select_model(spec, subset_candidates(_pool(), min_size=1),
                       background="auto")
    table = {r["Component"]: r for r in sel.best.peak_table()}
    assert "O_GaO" in table
    assert table["O_GaO"]["Area_pct"] > 60.0
