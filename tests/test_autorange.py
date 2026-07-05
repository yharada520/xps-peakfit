"""範囲自動検出の単体テスト."""
import numpy as np

from xps_peakfit.autorange import auto_range, estimate_noise_sigma
from xps_peakfit.io import Spectrum, load_spectrum
from xps_peakfit.models import pseudo_voigt


def test_noise_sigma_estimation() -> None:
    rng = np.random.default_rng(1)
    y = 1000.0 + rng.normal(0, 50.0, 2000)
    est = estimate_noise_sigma(y)
    assert 40.0 < est < 60.0


def test_auto_range_synthetic_peak() -> None:
    """孤立ピークで、窓がピークを覆いかつ全範囲より狭いこと."""
    rng = np.random.default_rng(2)
    x = np.arange(80.0, 120.0, 0.1)
    y = 500.0 + pseudo_voigt(x, 5000.0, 100.0, 1.5, 0.3) + rng.normal(0, 20, len(x))
    emin, emax = auto_range(Spectrum(x, y))
    assert emin < 98.0 < 102.0 < emax
    assert emin > 90.0 and emax < 110.0


def test_auto_range_flat_noise_returns_full() -> None:
    """ピークなし → 全範囲を返す."""
    rng = np.random.default_rng(3)
    x = np.arange(0.0, 20.0, 0.1)
    y = 100.0 + rng.normal(0, 5.0, len(x))
    emin, emax = auto_range(Spectrum(x, y))
    assert emin == x[0] and emax == x[-1]


def test_auto_range_ng_data_covers_si2p() -> None:
    """NGデータで、窓がSi2p領域 (99–103) を覆うこと."""
    spec = load_spectrum("data/XPS_Si2p_siloxane_NG.csv")
    emin, emax = auto_range(spec)
    assert emin <= 99.0
    assert emax >= 103.0
