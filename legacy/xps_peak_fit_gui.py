import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from math import sqrt, pi
import os
from datetime import datetime

def cumulative_trapezoid(y, x):
    y = np.asarray(y)
    x = np.asarray(x)
    dx = np.diff(x)
    mid = (y[:-1] + y[1:]) * 0.5
    out = np.empty(y.shape, dtype=float)
    out[0] = 0.0
    out[1:] = np.cumsum(mid * dx)
    return out

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def triple_gaussian(x, *params):
    return sum(gaussian(x, params[i], params[i+1], params[i+2]) for i in range(0, 9, 3))

def shirley_background(x, y, iters=50, tol=1e-5):
    i0, i1 = 0, -1
    y0, y1 = y[i0], y[i1]
    bg = np.linspace(y0, y1, len(y))
    for _ in range(iters):
        integral = cumulative_trapezoid(y - bg, x)
        norm = integral[-1] if abs(integral[-1]) > 1e-12 else 1.0
        bg_new = y0 + (y1 - y0) * (integral / norm)
        if np.allclose(bg, bg_new, atol=tol, rtol=0):
            break
        bg = bg_new
    return bg

def process_file(filepath):
    try:
        df = pd.read_csv(filepath)
        if not set(['energy', 'int']).issubset(df.columns):
            raise ValueError("CSV must have columns: 'energy', 'int'")
        # Cut to 96.5–105 eV
        df_cut = df[(df["energy"] >= 96.5) & (df["energy"] <= 105)].copy()
        if df_cut.empty:
            raise ValueError("No data points within [96.5, 105.0] eV.")
        df_cut.sort_values("energy", inplace=True)

        energy = df_cut["energy"].values
        intensity = df_cut["int"].values

        # S-G smoothing (robust window handling)
        win = min(len(intensity)//2*2 - 1, 31)
        win = max(5, win)
        if win % 2 == 0: win += 1
        poly = 3 if win >= 7 else 2

        smoothed = savgol_filter(intensity, window_length=win, polyorder=poly)
        background = shirley_background(energy, smoothed)
        corrected = smoothed - background

        # FWHM 2.0–2.2 eV → sigma bounds
        sigma_min = 2.0 / 2.355
        sigma_max = 2.2 / 2.355

        initial_guess = [5000, 99.0, 0.9,
                         10000, 101.0, 0.9,
                         15000, 103.0, 0.9]

        bounds = (
            [0, 98.5, sigma_min, 0, 100.5, sigma_min, 0, 102.5, sigma_min],
            [np.inf, 99.5, sigma_max, np.inf, 101.5, sigma_max, np.inf, 103.5, sigma_max]
        )

        popt, _ = curve_fit(triple_gaussian, energy, corrected, p0=initial_guess, bounds=bounds, maxfev=200000)

        fit_curve = triple_gaussian(energy, *popt)
        peaks = [gaussian(energy, *popt[i:i+3]) for i in range(0, 9, 3)]

        # Plot & save
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(os.path.dirname(filepath), f"fitting_plot_{ts}.png")
        plt.figure()
        plt.plot(energy, corrected, label="Corrected (smoothed - Shirley)")
        plt.plot(energy, fit_curve, label="Fit (3x Gaussian)", linestyle="--")
        for i, p in enumerate(peaks):
            plt.plot(energy, p, label=f"Peak {i+1}")
        plt.xlabel("Binding Energy (eV)")
        plt.ylabel("Intensity (a.u.)")
        plt.title("XPS Si2p: 3-Gaussian Fit (Au ghost, SiO, SiO2)")
        plt.legend()
        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()

        # Save results
        results = []
        labels = ["Au_ghost_~99eV", "SiO_(siloxane)_~101eV", "SiO2_~103eV"]
        for idx, label in enumerate(labels):
            amp = popt[idx*3]
            cen = popt[idx*3+1]
            wid = popt[idx*3+2]
            fwhm = wid * 2.355
            area = amp * wid * np.sqrt(2*np.pi)
            results.append({"Peak": label, "Center_eV": cen, "FWHM_eV": fwhm, "Height": amp, "Area": area})

        out_csv = os.path.join(os.path.dirname(filepath), f"fitting_result_gui_{ts}.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False)

        messagebox.showinfo("完了", f"解析完了:\n結果CSV: {out_csv}\nプロットPNG: {plot_path}")
    except Exception as e:
        messagebox.showerror("エラー", f"解析に失敗しました: {e}")

def select_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if filepath:
        process_file(filepath)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("XPS Si2p Peak Fitting GUI (Fixed)")
    btn = tk.Button(root, text="スペクトルファイルを選択", command=select_file)
    btn.pack(padx=20, pady=20)
    root.mainloop()
