"""ヘッダなしCSVと運動エネルギー→束縛エネルギー変換のテスト."""
import numpy as np
import pytest

from xps_peakfit.io import load_spectrum


@pytest.fixture()
def headerless_ke_csv(tmp_path):
    """Spring-8系データ形式: ヘッダなし、第1列KE昇順."""
    p = tmp_path / "ke.csv"
    ke = np.arange(192.0, 205.1, 0.1)
    y = 1000.0 + 50.0 * np.exp(-((ke - 199.5) / 1.0) ** 2)
    p.write_text("\n".join(f"{k:.1f},{v:.1f}" for k, v in zip(ke, y)),
                 encoding="utf-8")
    return p


def test_headerless_csv_loads(headerless_ke_csv) -> None:
    s = load_spectrum(headerless_ke_csv)
    assert len(s.energy) == 131
    assert s.energy[0] == pytest.approx(192.0)  # 変換なし=そのまま


def test_hv_converts_ke_to_be(headerless_ke_csv) -> None:
    s = load_spectrum(headerless_ke_csv, hv=730.0)
    # BE = 730 - KE: 205→525, 192→538。昇順ソートで [525, 538]
    assert s.energy[0] == pytest.approx(525.0)
    assert s.energy[-1] == pytest.approx(538.0)
    assert s.meta["hv_eV"] == 730.0
    # KE 199.5のピークは BE 530.5 に現れる
    assert s.energy[np.argmax(s.intensity)] == pytest.approx(530.5, abs=0.15)


def test_hv_with_headered_be_data() -> None:
    """既存のBEデータはhv未指定でそのまま（回帰）."""
    s = load_spectrum("data/XPS_Si2p.csv")
    assert s.energy[0] == pytest.approx(90.0)
