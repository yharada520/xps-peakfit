
import os, uuid, io, base64
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

app = Flask(__name__)
app.secret_key = "change-me"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def cumulative_trapezoid(y, x):
    y = np.asarray(y); x = np.asarray(x)
    dx = np.diff(x)
    mid = (y[:-1] + y[1:]) * 0.5
    out = np.empty(y.shape, dtype=float)
    out[0] = 0.0
    out[1:] = np.cumsum(mid * dx)
    return out

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def n_gaussian(x, *params):
    s = 0.0
    for i in range(0, len(params), 3):
        s += gaussian(x, params[i], params[i+1], params[i+2])
    return s

def shirley_background(x, y, iters=50, tol=1e-5):
    i0, i1 = 0, -1
    y0, y1 = y[i0], y[i1]
    bg = np.linspace(y0, y1, len(y))
    for _ in range(iters):
        integral = cumulative_trapezoid(y - bg, x)
        norm = integral[-1] if abs(integral[-1]) > 1e-12 else 1.0
        bg_new = y0 + (y1 - y0) * (integral / norm)
        if np.allclose(bg, bg_new, atol=tol, rtol=0): break
        bg = bg_new
    return bg

def shirley_background_with_endpoints(x, y, i0=0, i1=-1, iters=50, tol=1e-5):
    y = np.asarray(y); x = np.asarray(x)
    n = len(y)
    if i1 < 0: i1 = n + i1
    i0 = max(0, min(i0, n-1)); i1 = max(0, min(i1, n-1))
    if i0 > i1: i0, i1 = i1, i0
    y0, y1 = y[i0], y[i1]
    sub_x = x[i0:i1+1]; sub_y = y[i0:i1+1]
    bg = np.linspace(y0, y1, len(sub_y))
    for _ in range(iters):
        dx = np.diff(sub_x)
        mid = (sub_y[:-1] - bg[:-1] + sub_y[1:] - bg[1:]) * 0.5
        integral = np.empty_like(sub_y, dtype=float); integral[0] = 0.0
        integral[1:] = np.cumsum(mid * dx)
        norm = integral[-1] if abs(integral[-1]) > 1e-12 else 1.0
        bg_new = y0 + (y1 - y0) * (integral / norm)
        if np.allclose(bg, bg_new, atol=tol, rtol=0): break
        bg = bg_new
    full_bg = np.empty_like(y, dtype=float)
    full_bg[:i0] = bg[0]; full_bg[i0:i1+1] = bg; full_bg[i1+1:] = bg[-1]
    return full_bg

def shirley_background_adaptive(x, y, iters=50, tol=1e-5, neg_eps=-1e-9):
    bg0 = shirley_background(x, y, iters=iters, tol=tol)
    corr0 = y - bg0
    min_idx = int(np.argmin(corr0))
    if corr0[min_idx] >= neg_eps:
        return bg0
    n = len(y)
    if (n - 1 - min_idx) <= min_idx:
        bg1 = shirley_background_with_endpoints(x, y, i0=0, i1=min_idx, iters=iters, tol=tol)
    else:
        bg1 = shirley_background_with_endpoints(x, y, i0=min_idx, i1=-1, iters=iters, tol=tol)
    return bg1

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")

def find_fit_window(corrected, eps):
    n = len(corrected)
    k = int(np.argmax(corrected))
    i = k
    left = 0
    while i >= 0:
        if corrected[i] <= eps:
            left = min(n-1, i+1)
            break
        i -= 1
    j = k
    right = n-1
    while j < n:
        if corrected[j] <= eps:
            right = max(0, j-1)
            break
        j += 1
    if right - left + 1 < 5:
        mask = corrected > eps
        idx = np.where(mask)[0]
        if len(idx) >= 5:
            left, right = idx[0], idx[-1]
        else:
            raise ValueError("有効なフィット範囲が見つかりません（背景除去後がゼロ近傍に落ちています）。")
    return left, right

def process_spectrum(df, energy_min=96.5, energy_max=103.0):
    if not set(["energy","int"]).issubset(df.columns):
        raise ValueError("CSV must contain 'energy','int'")
    df_cut = df[(df["energy"]>=energy_min) & (df["energy"]<=energy_max)].copy()
    if df_cut.empty: raise ValueError("No points in range.")
    df_cut.sort_values("energy", inplace=True)
    energy = df_cut["energy"].values
    raw = df_cut["int"].values

    win = min(len(raw)//2*2 - 1, 21); win = max(5, win)
    if win % 2 == 0: win += 1
    poly = 2 if win < 9 else 3
    smoothed = savgol_filter(raw, window_length=win, polyorder=poly)

    background = shirley_background_adaptive(energy, smoothed)
    corrected = smoothed - background

    eps = max(1e-6, 2e-3 * float(np.max(corrected)) if np.max(corrected) > 0 else 1e-6)
    s, e = find_fit_window(corrected, eps)
    e_energy = energy[s:e+1]
    e_corr   = corrected[s:e+1]

    # ---- 3-Gaussian model ----
    sigma_min, sigma_max = 1.5/2.355, 1.8/2.355
    p0 = [5000, 97.5, 1.7/2.355,
          10000, 101.0, 1.7/2.355,
          15000, 102.0, 1.7/2.355]
    bounds = ([0, 97.0, sigma_min, 0, 101.0, sigma_min, 0, 102.0, sigma_min],
              [1e12, 98.0, sigma_max, 1e12, 101.9, sigma_max, 1e12, 102.5, sigma_max])
    popt, _ = curve_fit(n_gaussian, e_energy, e_corr, p0=p0, bounds=bounds, maxfev=300000)
    fit_curve = n_gaussian(e_energy, *popt)
    peaks = [gaussian(e_energy, *popt[i:i+3]) for i in range(0, len(popt), 3)]

    labels = ["Au_ghost_~97.5eV", "SiO_(siloxane)_~101eV", "SiO2_~102eV"]
    rows = []
    for idx,lbl in enumerate(labels):
        amp, cen, wid = popt[idx*3], popt[idx*3+1], popt[idx*3+2]
        fwhm = wid*2.355; area = float(amp * wid * np.sqrt(2*np.pi))
        rows.append({"Peak": lbl, "Center_eV": float(cen), "FWHM_eV": float(fwhm),
                     "Height": float(amp), "Area": area})

    plt.figure()
    plt.plot(energy, raw, label="Raw")
    plt.plot(energy, smoothed, label="Smoothed", linestyle="--")
    plt.plot(energy, background, label="Adaptive Shirley BG")
    plt.xlabel("Binding Energy (eV)"); plt.ylabel("Intensity (a.u.)")
    plt.title("Raw / Smoothed / Background")
    plt.legend(); plt.gca().invert_xaxis(); plt.tight_layout()
    img1 = fig_to_base64()

    plt.figure()
    plt.plot(energy, corrected, label="Corrected (full)", alpha=0.35)
    plt.plot(e_energy, e_corr, label="Corrected (fit region)")
    plt.plot(e_energy, fit_curve, label=f"Fit ({len(popt)//3}x Gaussian)", linestyle="--")
    for i,p in enumerate(peaks):
        plt.plot(e_energy, p, label=f"Peak {i+1}")
    plt.xlabel("Binding Energy (eV)"); plt.ylabel("Intensity (a.u.)")
    plt.title("Corrected / Fit / Components (3-Gaussian)")
    plt.legend(); plt.gca().invert_xaxis(); plt.tight_layout()
    img2 = fig_to_base64()

    return rows, img1, img2

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/fit", methods=["POST"])
def fit():
    if "file" not in request.files or request.files["file"].filename == "":
        flash("CSVファイルを選択してください。"); return redirect(url_for("index"))
    f = request.files["file"]
    try:
        df = pd.read_csv(f)
        rows, img_rawbg, img_fit = process_spectrum(df)
        tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        csv_name = f"fitting_result_{tag}.csv"
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, csv_name), index=False)
        return render_template("result.html", table=rows, img_rawbg=img_rawbg, img_fit=img_fit, csv_name=csv_name)
    except Exception as e:
        flash(f"解析に失敗しました: {e}")
        return redirect(url_for("index"))

@app.route("/download/<path:fname>")
def download(fname):
    return send_from_directory(OUTPUT_DIR, fname, as_attachment=True)

@app.route("/view/<path:fname>")
def view(fname):
    return send_from_directory(OUTPUT_DIR, fname, as_attachment=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
