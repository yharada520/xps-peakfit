import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit
from math import sqrt, pi

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def triple_gaussian(x, *params):
    return sum(gaussian(x, params[i], params[i+1], params[i+2]) for i in range(0, 9, 3))

def shirley_background(x, y):
    i0, i1 = 0, -1
    y0, y1 = y[i0], y[i1]
    bg = np.linspace(y0, y1, len(y))
    for _ in range(10):
        integral = cumtrapz(y - bg, x, initial=0)
        norm = integral[-1] if integral[-1] != 0 else 1
        bg_new = y0 + (y1 - y0) * integral / norm
        if np.allclose(bg, bg_new, atol=1e-3):
            break
        bg = bg_new
    return bg

def process_file(filepath):
    df = pd.read_csv(filepath)
    df_cut = df[(df["energy"] >= 96.5) & (df["energy"] <= 105)].copy()
    df_cut.sort_values("energy", inplace=True)

    energy = df_cut["energy"].values
    intensity = df_cut["int"].values
    smoothed = savgol_filter(intensity, window_length=11, polyorder=3)
    background = shirley_background(energy, smoothed)
    corrected = smoothed - background

    initial_guess = [5000, 99.0, 0.9, 10000, 101.0, 0.9, 15000, 103.0, 0.9]
    sigma_min = 2.0 / 2.355
    sigma_max = 2.2 / 2.355
    bounds = (
        [0, 98.5, sigma_min, 0, 100.5, sigma_min, 0, 102.5, sigma_min],
        [np.inf, 99.5, sigma_max, np.inf, 101.5, sigma_max, np.inf, 103.5, sigma_max]
    )

    popt, _ = curve_fit(triple_gaussian, energy, corrected, p0=initial_guess, bounds=bounds)

    results = []
    for i in range(3):
        amp = popt[i * 3]
        cen = popt[i * 3 + 1]
        wid = popt[i * 3 + 2]
        fwhm = wid * 2.355
        area = amp * wid * np.sqrt(2 * np.pi)
        results.append({"Peak": f"Peak {i+1}", "Center": cen, "FWHM": fwhm, "Height": amp, "Area": area})

    df_result = pd.DataFrame(results)
    out_csv = filepath.with_suffix("").name + "_fit_results.csv"
    df_result.to_csv(filepath.parent / out_csv, index=False)

# 実行ディレクトリ内の ./spectra フォルダをスキャン
data_dir = Path("./spectra")
for file in data_dir.glob("*.csv"):
    process_file(file)

print("✅ バッチ処理完了しました！")
