import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XPS Si2p Peak Fitting GUI (v2)")
        self.geometry("780x520")

        frm = tk.Frame(self)
        frm.pack(fill="x", padx=10, pady=10)

        self.btn_open = tk.Button(frm, text="CSVを開く", command=self.select_file)
        self.btn_open.pack(side="left")

        self.btn_save_dir = tk.Button(frm, text="保存先フォルダを選択", command=self.select_dir)
        self.btn_save_dir.pack(side="left", padx=8)

        self.lbl_dir = tk.Label(frm, text="保存先: (未選択→CSVと同じ)")
        self.lbl_dir.pack(side="left", padx=8)

        # Results table
        cols = ("Peak", "Center_eV", "FWHM_eV", "Height", "Area")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=8)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)

        # Status
        self.status = tk.StringVar(value="準備完了")
        tk.Label(self, textvariable=self.status, anchor="w").pack(fill="x", padx=10, pady=(0,10))

        self.save_dir = None
        self.last_outputs = None  # (csv_path, png_path)

    def select_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.save_dir = d
            self.lbl_dir.config(text=f"保存先: {d}")

    def select_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if filepath:
            self.process_file(filepath)

    def process_file(self, filepath):
        try:
            df = pd.read_csv(filepath)
            if not set(['energy','int']).issubset(df.columns):
                raise ValueError("CSV must have columns: 'energy', 'int'")

            df_cut = df[(df["energy"] >= 96.5) & (df["energy"] <= 105)].copy()
            if df_cut.empty:
                raise ValueError("No data points within [96.5, 105.0] eV.")
            df_cut.sort_values("energy", inplace=True)

            energy = df_cut["energy"].values
            intensity = df_cut["int"].values

            win = min(len(intensity)//2*2 - 1, 31); win = max(5, win)
            if win % 2 == 0: win += 1
            poly = 3 if win >= 7 else 2

            smoothed = savgol_filter(intensity, window_length=win, polyorder=poly)
            background = shirley_background(energy, smoothed)
            corrected = smoothed - background

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

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = self.save_dir if self.save_dir else os.path.dirname(filepath)
            csv_path = os.path.join(base_dir, f"fitting_result_gui_{ts}.csv")
            png_path = os.path.join(base_dir, f"fitting_plot_{ts}.png")

            # Save plot
            plt.figure()
            plt.plot(energy, corrected, label="Corrected (smoothed - Shirley)")
            plt.plot(energy, fit_curve, label="Fit (3x Gaussian)", linestyle="--")
            for i, p in enumerate(peaks):
                plt.plot(energy, p, label=f"Peak {i+1}")
            plt.xlabel("Binding Energy (eV)"); plt.ylabel("Intensity (a.u.)")
            plt.title("XPS Si2p: 3-Gaussian Fit (Au ghost, SiO, SiO2)")
            plt.legend(); plt.gca().invert_xaxis(); plt.tight_layout()
            plt.savefig(png_path, dpi=160); plt.close()

            # Save params
            labels = ["Au_ghost_~99eV", "SiO_(siloxane)_~101eV", "SiO2_~103eV"]
            rows = []
            for idx, label in enumerate(labels):
                amp = popt[idx*3]; cen = popt[idx*3+1]; wid = popt[idx*3+2]
                fwhm = wid * 2.355; area = amp * wid * np.sqrt(2*np.pi)
                rows.append({"Peak": label, "Center_eV": cen, "FWHM_eV": fwhm, "Height": amp, "Area": area})

            pd.DataFrame(rows).to_csv(csv_path, index=False)

            # Display in table
            for i in self.tree.get_children():
                self.tree.delete(i)
            for r in rows:
                self.tree.insert("", "end", values=(r["Peak"],
                                                    f"{r['Center_eV']:.3f}",
                                                    f"{r['FWHM_eV']:.3f}",
                                                    f"{r['Height']:.1f}",
                                                    f"{r['Area']:.1f}"))
            self.last_outputs = (csv_path, png_path)
            self.status.set(f"完了: CSV={csv_path} / PNG={png_path}")

            # Ask user to save-as (optional). If cancel, files remain at base_dir.
            if messagebox.askyesno("保存", "別名で保存しますか？（任意）"):
                # CSV
                new_csv = filedialog.asksaveasfilename(defaultextension=".csv",
                                                       initialfile=os.path.basename(csv_path),
                                                       filetypes=[("CSV","*.csv")])
                if new_csv:
                    try:
                        os.replace(csv_path, new_csv)
                        csv_path = new_csv
                    except Exception as e:
                        messagebox.showwarning("保存警告", f"CSVの保存に失敗: {e}")
                # PNG
                new_png = filedialog.asksaveasfilename(defaultextension=".png",
                                                       initialfile=os.path.basename(png_path),
                                                       filetypes=[("PNG","*.png")])
                if new_png:
                    try:
                        os.replace(png_path, new_png)
                        png_path = new_png
                    except Exception as e:
                        messagebox.showwarning("保存警告", f"PNGの保存に失敗: {e}")
                self.last_outputs = (csv_path, png_path)
                self.status.set(f"保存先: CSV={csv_path} / PNG={png_path}")

        except Exception as e:
            messagebox.showerror("エラー", f"解析に失敗しました: {e}")

if __name__ == "__main__":
    App().mainloop()
