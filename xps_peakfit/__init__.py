"""xps_peakfit: XPS自動ピーク分離パッケージ.

擬Voigtピーク + active background (Shirley/Tougaard) + MAP推定 + BIC本数選択。
"""
from xps_peakfit.io import Spectrum, load_spectrum
from xps_peakfit.models import pseudo_voigt, doublet_pseudo_voigt, pv_area
from xps_peakfit.lines import LINE_REGISTRY, LineShape, get_line
from xps_peakfit.fitting import Component, FitResult, fit_components
from xps_peakfit.model_select import SelectionResult, select_model

__version__ = "0.1.0"

__all__ = [
    "Spectrum", "load_spectrum",
    "pseudo_voigt", "doublet_pseudo_voigt", "pv_area",
    "LINE_REGISTRY", "LineShape", "get_line",
    "Component", "FitResult", "fit_components",
    "SelectionResult", "select_model",
    "__version__",
]
