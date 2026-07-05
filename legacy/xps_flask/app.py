
import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from math import sqrt
import numpy as np

# ----- Flask setup -----
app = Flask(__name__)
app.secret_key = "change-me"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- Math/helpers -----
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

def process_spectrum(df, energy_min=96.5, energy_max=105.0):
    if not set(["energy","int"]).issubset(df.columns):
        raise ValueError("CSV must contain columns: 'energy', 'int'")

    df_cut = df[(df["energy"] >= energy_min) & (df["energy"] <= energy_max)].copy()
    if df_cut.empty:
        raise ValueError(f"No data points within [{energy_min}, {energy_max}] eV.")
    df_cut.sort_values("energy", inplace=True)

    energy = df_cut["energy"].values
    intensity = df_cut["int"].values

    # S-G smoothing
    win = min(len(intensity)//2*2 - 1, 31)  # ensure odd and <=31
    win = max(5, win)
    if win % 2 == 0:
        win += 1
    poly = 3 if win >= 7 else 2

    smoothed = savgol_filter(intensity, window_length=win, polyorder=poly)
    background = shirley_background(energy, smoothed)
    corrected = smoothed - background

    # FWHM 2.0–2.2 eV ⇒ sigma
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

    # Collect params
    labels = ["Au_ghost_~99eV", "SiO_(siloxane)_~101eV", "SiO2_~103eV"]
    results = []
    for idx, label in enumerate(labels):
        amp = popt[idx*3]
        cen = popt[idx*3+1]
        wid = popt[idx*3+2]
        fwhm = wid * 2.355
        area = amp * wid * np.sqrt(2*np.pi)
        results.append({"Peak": label, "Center_eV": float(cen), "FWHM_eV": float(fwhm),
                        "Height": float(amp), "Area": float(area)})

    return energy, corrected, fit_curve, peaks, results

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/fit", methods=["POST"])
def fit():
    if "file" not in request.files:
        flash("CSVファイルを選択してください。")
        return redirect(url_for("index"))
    f = request.files["file"]
    if f.filename == "":
        flash("CSVファイルを選択してください。")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(f)
        energy, corrected, fit_curve, peaks, results = process_spectrum(df)

        # Save artifacts
        tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        csv_name = f"fitting_result_{tag}.csv"
        png_name = f"fitting_plot_{tag}.png"
        csv_path = os.path.join(OUTPUT_DIR, csv_name)
        png_path = os.path.join(OUTPUT_DIR, png_name)

        pd.DataFrame(results).to_csv(csv_path, index=False)

        # Plot
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(energy, corrected, label="Corrected (smoothed - Shirley)")
        plt.plot(energy, fit_curve, label="Fit (3x Gaussian)", linestyle="--")
        for i, p in enumerate(peaks):
            plt.plot(energy, p, label=f"Peak {i+1}")
        plt.xlabel("Binding Energy (eV)")
        plt.ylabel("Intensity (a.u.)")
        plt.title("XPS Si2p: 3-Gaussian Fit (Au ghost, SiO, SiO2)")
        plt.legend(); plt.gca().invert_xaxis(); plt.tight_layout()
        plt.savefig(png_path, dpi=160); plt.close()

        return render_template("result.html",
                               table=results,
                               csv_name=csv_name,
                               png_name=png_name)
    except Exception as e:
        flash(f"解析に失敗しました: {e}")
        return redirect(url_for("index"))

@app.route("/download/<path:fname>")
def download(fname):
    return send_from_directory(OUTPUT_DIR, fname, as_attachment=True)

@app.route("/view/<path:fname>")
def view(fname):
    # For <img src="..."> embedding
    return send_from_directory(OUTPUT_DIR, fname, as_attachment=False)

if __name__ == "__main__":
    # Run: FLASK_APP=app.py flask run --reload
    app.run(host="0.0.0.0", port=5000, debug=True)
