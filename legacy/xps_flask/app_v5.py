# app_v5.py
# XPS Si2p Peak Fitting (GP residual + robust LS) / Flask GUI+API
# env layout:
#   templates/ index.html, result.html
#   uploads/   (auto-created)
#   output/    (CSV results)
# default port: 5000

import io, os, json, base64, time, traceback
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename

from scipy.signal import savgol_filter
from scipy.optimize import least_squares

# --- scikit-learn (GPR) ---
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor


# =========================
# Paths / Flask
# =========================
BASE_DIR     = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
UPLOAD_DIR   = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR   = os.path.join(BASE_DIR, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=None)
app.secret_key = "xps-gp-fit"
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64MB


# =========================
# Utils
# =========================
def fig_to_base64() -> str:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close()
    return b64

def load_xps_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    for c in df.columns:
        lc = c.strip().lower().replace(" ", "").replace("\t", "")
        if lc in ["energy", "bindingenergy", "be", "e"]:
            rename[c] = "energy"
        elif lc in ["int", "intensity", "counts", "y"]:
            rename[c] = "int"
    df = df.rename(columns=rename)
    if not {"energy", "int"}.issubset(df.columns):
        raise ValueError("CSVに 'energy' と 'int' カラムが見つかりません。")
    return df[["energy", "int"]].copy()


# =========================
# Peak models
# =========================
def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-0.5 * ((x - cen) / sigma) ** 2)

def n_gaussian(x, *params):
    s = 0.0
    for i in range(0, len(params), 3):
        s += gaussian(x, params[i], params[i+1], params[i+2])
    return s

def pseudo_voigt(x, amp, cen, fwhm, eta):
    sigma = fwhm / 2.354820045
    g = amp * np.exp(-(x - cen)**2 / (2*sigma**2))
    l = amp * (0.5*fwhm)**2 / ((x - cen)**2 + (0.5*fwhm)**2)
    return eta * l + (1 - eta) * g

def n_pvoigt(x, *params):
    s = 0.0
    for i in range(0, len(params), 4):
        s += pseudo_voigt(x, params[i], params[i+1], params[i+2], params[i+3])
    return s


# =========================
# Shirley background
# =========================
def shirley_background(energy, intensity, max_iter=200, tol=1e-6):
    y = intensity.astype(float)
    i_low, i_high = y[0], y[-1]
    if i_low < i_high:
        i_low, i_high = i_high, i_low
    bg = np.linspace(i_low, i_high, len(y))
    for _ in range(max_iter):
        prev = bg.copy()
        resid = y - bg
        integ = np.cumsum(resid[::-1])[::-1]
        K = (i_low - i_high) / (integ[0] + 1e-12)
        bg = i_high + K * integ
        if np.max(np.abs(bg - prev)) < tol:
            break
    return bg

def shirley_background_adaptive(energy, intensity):
    bg = shirley_background(energy, intensity)
    win = max(7, min(51, (len(bg)//2)*2-1))
    return savgol_filter(bg, window_length=win, polyorder=3)


# =========================
# Fit window
# =========================
def find_fit_window(y: np.ndarray, eps: float, pad: int = 5) -> Tuple[int, int]:
    i_max = int(np.argmax(y))
    left = i_max
    while left > 0 and y[left] > eps:
        left -= 1
    right = i_max
    while right < len(y) - 1 and y[right] > eps:
        right += 1
    return max(0, left - pad), min(len(y) - 1, right + pad)


# =========================
# Robust LS + multi-start
# =========================
def fit_peaks_robust(x, y, model, p0, lb, ub, n_starts=12,
                     loss='soft_l1', f_scale=1.0, max_nfev=20000, rng_seed=42):
    best = None
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_starts):
        jitter = 0.20 * (ub - lb) * (rng.random(len(p0)) - 0.5)
        seed = np.clip(p0 + jitter, lb, ub)
        fun = lambda p: model(x, *p) - y
        res = least_squares(fun, seed, bounds=(lb, ub),
                            loss=loss, f_scale=f_scale, max_nfev=max_nfev)
        if (best is None) or (res.cost < best.cost):
            best = res
    return best.x, best


# =========================
# GP residual
# =========================
def gpr_residual(x, y_resid, alpha_floor=1e-10, length_scale_init=0.3):
    X = x.reshape(-1, 1)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=length_scale_init,
                                       length_scale_bounds=(1e-2, 5.0)) \
             + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha_floor,
                                  normalize_y=True, n_restarts_optimizer=3)
    gp.fit(X, y_resid)
    mu, _ = gp.predict(X, return_std=True)
    return mu


# =========================
# Core process
# =========================
@dataclass
class FitParams:
    energy_min: float = 96.5
    energy_max: float = 105.0
    fwhm_min: float = 1.7
    fwhm_max: float = 2.1
    eps_pct: float = 0.2
    smooth_maxwin: int = 21
    poly: int = 3
    c1: float = 97.6
    c2: float = 99.9
    c3: float = 102.4
    c1_min: float = 97.0
    c2_min: float = 99.5
    c3_min: float = 101.8
    c1_max: float = 98.2
    c2_max: float = 100.3
    c3_max: float = 102.7
    use_pvoigt: bool = False
    use_gp: bool = True
    max_alt_iter: int = 3
    tol_rel: float = 1e-3

def process_spectrum_gp(df: pd.DataFrame, params: FitParams):
    # cut
    df_cut = df[(df["energy"] >= params.energy_min) & (df["energy"] <= params.energy_max)].copy().sort_values("energy")
    energy = df_cut["energy"].values
    raw = df_cut["int"].values
    
    # smoothing + background
    win = min(len(raw)//2*2 - 1, params.smooth_maxwin)
    win = max(5, win)
    if win % 2 == 0:
        win += 1
    polyorder = max(2, min(int(params.poly), 5))  # ★ ここでint化
    smoothed = savgol_filter(raw, window_length=int(win), polyorder=polyorder)
    background = shirley_background_adaptive(energy, smoothed)
    corrected_all = smoothed - background

    # fit window
    eps = max(1e-6, (params.eps_pct/100.0) * float(np.max(corrected_all)) if np.max(corrected_all) > 0 else 1e-6)
    s, e = find_fit_window(corrected_all, eps)
    x = energy[s:e+1]
    y = corrected_all[s:e+1]

    # model/initial
    centers = [params.c1, params.c2, params.c3]
    cmins   = [params.c1_min, params.c2_min, params.c3_min]
    cmaxs   = [params.c1_max, params.c2_max, params.c3_max]

    if params.use_pvoigt:
        p0 = [0.35*np.max(y), centers[0], 0.5*(params.fwhm_min+params.fwhm_max), 0.3,
              0.65*np.max(y), centers[1], 0.5*(params.fwhm_min+params.fwhm_max), 0.3,
              0.85*np.max(y), centers[2], 0.5*(params.fwhm_min+params.fwhm_max), 0.3]
        lb = [0, cmins[0], params.fwhm_min, 0.0,
              0, cmins[1], params.fwhm_min, 0.0,
              0, cmins[2], params.fwhm_min, 0.0]
        ub = [1e12, cmaxs[0], params.fwhm_max, 1.0,
              1e12, cmaxs[1], params.fwhm_max, 1.0,
              1e12, cmaxs[2], params.fwhm_max, 1.0]
        model = n_pvoigt
    else:
        sigma_min, sigma_max = params.fwhm_min/2.355, params.fwhm_max/2.355
        p0 = [0.35*np.max(y), centers[0], (0.5*(params.fwhm_min+params.fwhm_max))/2.355,
              0.65*np.max(y), centers[1], (0.5*(params.fwhm_min+params.fwhm_max))/2.355,
              0.85*np.max(y), centers[2], (0.5*(params.fwhm_min+params.fwhm_max))/2.355]
        lb = [0, cmins[0], sigma_min,  0, cmins[1], sigma_min,  0, cmins[2], sigma_min]
        ub = [1e12, cmaxs[0], sigma_max, 1e12, cmaxs[1], sigma_max, 1e12, cmaxs[2], sigma_max]
        model = n_gaussian

    resid_gp = np.zeros_like(y)
    prev_cost = np.inf
    pcur = np.array(p0, dtype=float)

    for _ in range(int(params.max_alt_iter)):
        y_eff = y - resid_gp
        popt, res = fit_peaks_robust(x, y_eff, model, pcur, np.array(lb), np.array(ub),
                                     n_starts=12, loss='soft_l1', f_scale=1.0)
        fit_curve = model(x, *popt)
        resid = y - fit_curve
        resid_gp = gpr_residual(x, resid) if params.use_gp else np.zeros_like(y)

        if abs(prev_cost - res.cost) / (prev_cost + 1e-12) < params.tol_rel:
            pcur = popt
            break
        prev_cost = res.cost
        pcur = popt

    popt = pcur
    y_fit = model(x, *popt) + resid_gp

    # components + areas
    labels = ["Au_ghost", "SiO_siloxane", "SiO2"]
    table = []
    comps = []
    if params.use_pvoigt:
        for k in range(3):
            A, C, F, eta = popt[4*k:4*k+4]
            comps.append(pseudo_voigt(x, A, C, F, eta))
            area = float(np.pi*A*F*(eta + (1-eta)/np.sqrt(4*np.log(2))))
            table.append(dict(Peak=labels[k], Center_eV=float(C), FWHM_eV=float(F),
                              Height=float(A), Area=area))
    else:
        for k in range(3):
            A, C, S = popt[3*k:3*k+3]
            comps.append(gaussian(x, A, C, S))
            fwhm = float(S*2.355)
            area = float(A*S*np.sqrt(2*np.pi))
            table.append(dict(Peak=labels[k], Center_eV=float(C), FWHM_eV=fwhm,
                              Height=float(A), Area=area))

    # Plot1
    plt.figure(figsize=(7,4.5))
    plt.plot(energy, raw, label="Raw")
    plt.plot(energy, smoothed, "--", label="Smoothed")
    plt.plot(energy, background, label="Adaptive Shirley BG")
    plt.xlabel("Binding Energy (eV)"); plt.ylabel("Intensity (a.u.)")
    plt.title("Raw / Smoothed / Background")
    plt.legend(); plt.gca().invert_xaxis()
    img1 = fig_to_base64()

    # Plot2
    plt.figure(figsize=(7,4.5))
    plt.plot(energy, corrected_all, alpha=0.35, label="Corrected (full)")
    plt.plot(x, y, label="Corrected (fit region)")
    plt.plot(x, y_fit, "--", label=f"Fit ({'pVoigt' if params.use_pvoigt else 'Gaussian'} + {'GP' if params.use_gp else 'noGP'})")
    for i, c in enumerate(comps):
        plt.plot(x, c, label=f"Peak {i+1}")
    if params.use_gp:
        plt.plot(x, resid_gp, ":", label="GP residual")
    plt.xlabel("Binding Energy (eV)"); plt.ylabel("Intensity (a.u.)")
    plt.title("Corrected / Fit / Components (GP-assisted)")
    plt.legend(); plt.gca().invert_xaxis()
    img2 = fig_to_base64()

    used = dict(
        energy_min=params.energy_min, energy_max=params.energy_max,
        fwhm_min=params.fwhm_min, fwhm_max=params.fwhm_max,
        eps_pct=params.eps_pct, smooth_maxwin=params.smooth_maxwin, poly=params.poly,
        c1=params.c1, c2=params.c2, c3=params.c3,
        c1_min=params.c1_min, c2_min=params.c2_min, c3_min=params.c3_min,
        c1_max=params.c1_max, c2_max=params.c2_max, c3_max=params.c3_max,
        use_pvoigt=params.use_pvoigt, use_gp=params.use_gp
    )
    return table, img1, img2, used


# =========================
# Web UI
# =========================
@app.route("/", methods=["GET"])
def index():
    # テンプレが見つからない場合は早期に分かるようにする
    if not os.path.exists(os.path.join(TEMPLATE_DIR, "index.html")):
        return "templates/index.html が見つかりません。配置してください。", 500
    return render_template("index.html", defaults=FitParams().__dict__)


def _params_from_form(form) -> FitParams:
    # どの項目を整数として受け取るか
    INT_FIELDS = {"smooth_maxwin", "poly", "max_alt_iter"}
    BOOL_FIELDS = {"use_gp", "use_pvoigt"}

    kwargs = {}
    for k, v in form.items():
        if k in BOOL_FIELDS:
            kwargs[k] = (v == "on")
        elif k in INT_FIELDS:
            try:
                kwargs[k] = int(float(v))
            except Exception:
                pass
        else:
            try:
                kwargs[k] = float(v)
            except Exception:
                pass

    # チェックボックスが未送信の時は False を補完
    for b in BOOL_FIELDS:
        kwargs.setdefault(b, False)

    # dataclass に該当キーのみ渡す
    return FitParams(**{k: v for k, v in kwargs.items() if hasattr(FitParams, k)})


@app.route("/fit", methods=["POST"])
def fit_form():
    try:
        if "file" not in request.files or request.files["file"].filename == "":
            flash("CSVファイルを選択してください。")
            return redirect(url_for("index"))

        f = request.files["file"]
        fname = secure_filename(f.filename)
        save_path = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{fname}")
        f.save(save_path)

        params = _params_from_form(request.form)
        df = load_xps_csv(save_path)
        table, img1, img2, used = process_spectrum_gp(df, params)

        # save CSV
        out_name = f"fit_{int(time.time())}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        pd.DataFrame(table).to_csv(out_path, index=False)

        if not os.path.exists(os.path.join(TEMPLATE_DIR, "result.html")):
            return "templates/result.html が見つかりません。配置してください。", 500

        return render_template(
            "result.html",
            table=table,
            img_rawbg=f"data:image/png;base64,{img1}",
            img_fit=f"data:image/png;base64,{img2}",
            used=used,
            csv_name=out_name
        )
    except Exception as e:
        app.logger.exception("fit_form failed")
        flash(f"解析に失敗しました: {e}")
        return redirect(url_for("index"))

@app.route("/download")
def download():
    fname = request.args.get("fname")
    if not fname:
        return "fname required", 400
    path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(path):
        return "file not found", 404
    return send_file(path, as_attachment=True, download_name=fname, mimetype="text/csv")


# =========================
# JSON API (optional)
# =========================
@app.route("/schema", methods=["GET"])
def schema():
    return jsonify(FitParams().__dict__)

@app.route("/fit_json", methods=["POST"])
def fit_json():
    data = request.get_json(force=True, silent=True) or {}
    csv_path = data.get("csv_path")
    if not csv_path:
        return jsonify({"ok": False, "error": "csv_path is required"}), 400
    p = FitParams(**{k: v for k, v in (data.get("params") or {}).items() if hasattr(FitParams, k)})
    try:
        df = load_xps_csv(csv_path)
        table, img1, img2, used = process_spectrum_gp(df, p)
        return jsonify({
            "ok": True,
            "peaks": table,
            "plot_raw_bg_png_base64": img1,
            "plot_fit_png_base64": img2,
            "used_params": used
        })
    except Exception as e:
        app.logger.exception("fit_json failed")
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/healthz")
def healthz():
    return "ok", 200


# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="XPS GP-assisted peak fitting (GUI+API)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)  # ← 既定を5000に戻す
    args = parser.parse_args()
    print(f"[app_v5] http://{args.host}:{args.port}  templates={TEMPLATE_DIR}  output={OUTPUT_DIR}")
    app.run(host=args.host, port=args.port, debug=False)
