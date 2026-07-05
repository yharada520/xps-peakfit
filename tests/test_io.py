"""io モジュールの単体テスト."""
import numpy as np
import pandas as pd
import pytest

from xps_peakfit.io import Spectrum, load_spectrum


def test_spectrum_sorts_ascending() -> None:
    s = Spectrum(np.array([103.0, 101.0, 102.0, 100.0, 99.0]),
                 np.array([3.0, 1.0, 2.0, 0.0, 5.0]))
    assert np.all(np.diff(s.energy) > 0)
    assert s.intensity[0] == 5.0


def test_crop_and_step() -> None:
    e = np.arange(90.0, 110.0, 0.1)
    s = Spectrum(e, np.ones_like(e))
    c = s.crop(95.0, 105.0)
    assert c.energy[0] >= 95.0 and c.energy[-1] <= 105.0
    assert c.step == pytest.approx(0.1)


def test_load_flexible_columns(tmp_path) -> None:
    p = tmp_path / "t.csv"
    pd.DataFrame({"Binding Energy": [1.0, 2, 3, 4, 5],
                  "Counts": [10.0, 20, 30, 20, 10]}).to_csv(p, index=False)
    s = load_spectrum(p)
    assert len(s.energy) == 5
    assert s.name == "t"


def test_load_real_data() -> None:
    s = load_spectrum("data/XPS_Si2p_siloxane_NG.csv")
    assert len(s.energy) == 201
    assert s.energy[0] == pytest.approx(90.0)
    assert s.energy[-1] == pytest.approx(110.0)
