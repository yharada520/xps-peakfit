import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os

# --- 計算用関数（ロジック部分はそのまま継承） ---
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def multi_gaussian_linear(x, *params):
    y = params[0] * x + params[1]  # Linear BG
    n_peaks = (len(params) - 2) // 3
    for i in range(n_peaks):
        amp = params[2 + 3*i]
        cen = params[2 + 3*i + 1]
        wid = params[2 + 3*i + 2]
        y += gaussian(x, amp, cen, wid)
    return y

def calculate_bic(y_true, y_pred, n_params):
    n = len(y_true)
    rss = np.sum((y_true - y_pred)**2)
    if rss == 0: return np.inf
    bic = n * np.log(rss / n) + n_params * np.log(n)
    return bic

# --- GUIクラス ---
class XPSAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("XPS Peak Auto-Fitter (BIC Optimization)")
        self.root.geometry("1100x700")

        # データ保持用
        self.df = None
        self.filename = ""

        # --- レイアウト作成 ---
        
        # 1. コントロールパネル (上部)
        control_frame = ttk.LabelFrame(root, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # ファイル選択ボタン
        self.btn_load = ttk.Button(control_frame, text="1. Load CSV File", command=self.load_file)
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        # ファイル名表示ラベル
        self.lbl_file = ttk.Label(control_frame, text="No file loaded")
        self.lbl_file.pack(side=tk.LEFT, padx=5)

        # ピーク最大数設定
        ttk.Label(control_frame, text="| Max Peaks:").pack(side=tk.LEFT, padx=(20, 5))
        self.spin_max_peaks = ttk.Spinbox(control_frame, from_=1, to=10, width=5)
        self.spin_max_peaks.set(5) # デフォルト
        self.spin_max_peaks.pack(side=tk.LEFT, padx=5)

        # 実行ボタン
        self.btn_run = ttk.Button(control_frame, text="2. Run Analysis", command=self.run_analysis, state=tk.DISABLED)
        self.btn_run.pack(side=tk.LEFT, padx=20)

        # 2. メインエリア (左右分割)
        paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 左側：テキスト結果表示
        result_frame = ttk.LabelFrame(paned_window, text="Fitting Results", padding=5, width=350)
        paned_window.add(result_frame, weight=1)
        
        self.txt_result = tk.Text(result_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.txt_result.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(result_frame, command=self.txt_result.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_result['yscrollcommand'] = scrollbar.set

        # 右側：グラフ表示
        plot_frame = ttk.LabelFrame(paned_window, text="Plot", padding=5)
        paned_window.add(plot_frame, weight=3)

        # MatplotlibのFigureを作成
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Canvasを配置
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ツールバー（ズームとか保存用）
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        """テキストボックスに文字を出力"""
        self.txt_result.insert(tk.END, message + "\n")
        self.txt_result.see(tk.END)

    def load_file(self):
        """ファイル選択ダイアログ"""
        f_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not f_path:
            return

        try:
            self.df = pd.read_csv(f_path)
            self.filename = os.path.basename(f_path)
            self.lbl_file.config(text=f"Loaded: {self.filename}")
            self.btn_run.config(state=tk.NORMAL)
            self.log(f"--- Loaded {self.filename} ---")
            
            # 簡単なデータチェックとプロット
            self.plot_raw_data()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def get_xy_data(self):
        """列名を柔軟に判定してX, Yデータを返す"""
        cols = [c.lower() for c in self.df.columns]
        
        # X軸 (Energy or BE)
        if 'energy' in cols: x_col = self.df.columns[cols.index('energy')]
        elif 'be' in cols: x_col = self.df.columns[cols.index('be')]
        else: x_col = self.df.columns[0]
        
        # Y軸 (Intensity or Int)
        if 'intensity' in cols: y_col = self.df.columns[cols.index('intensity')]
        elif 'int' in cols: y_col = self.df.columns[cols.index('int')]
        else: y_col = self.df.columns[1]
        
        return self.df[x_col].values, self.df[y_col].values, x_col, y_col

    def plot_raw_data(self):
        """解析前の生データをプロット"""
        x, y, x_name, y_name = self.get_xy_data()
        self.ax.clear()
        self.ax.plot(x, y, 'ko', markersize=2, alpha=0.5, label='Raw Data')
        self.ax.set_xlabel(x_name)
        self.ax.set_ylabel(y_name)
        self.ax.set_title(f"Raw Data: {self.filename}")
        self.ax.grid(True)
        self.canvas.draw()

    def run_analysis(self):
        """BICによる自動フィッティング実行"""
        if self.df is None: return
        
        self.log(f"\n[Start Analysis] Target: {self.filename}")
        self.btn_run.config(state=tk.DISABLED)
        self.root.update() # 画面更新

        x_data, y_data, x_name, y_name = self.get_xy_data()
        max_peaks = int(self.spin_max_peaks.get())
        
        best_result = None
        min_bic = np.inf

        # 線形BG初期値
        slope_init = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0])
        intercept_init = y_data[0] - slope_init * x_data[0]

        # 本数ループ
        for n in range(1, max_peaks + 1):
            self.log(f"Trying {n} peak(s)...")
            
            # ピーク検出で初期値を決める
            peaks_idx, _ = find_peaks(y_data, height=np.max(y_data)*0.1, distance=len(x_data)//(n+1))
            
            p0 = [slope_init, intercept_init]
            bounds_min = [-np.inf, -np.inf]
            bounds_max = [np.inf, np.inf]
            
            current_peak_count = 0
            for idx in peaks_idx:
                if current_peak_count >= n: break
                p0.extend([y_data[idx], x_data[idx], 1.0])
                bounds_min.extend([0, np.min(x_data), 0.1])
                bounds_max.extend([np.inf, np.max(x_data), (np.max(x_data)-np.min(x_data))/2])
                current_peak_count += 1
            
            while current_peak_count < n:
                center_guess = np.min(x_data) + (np.max(x_data) - np.min(x_data)) * (current_peak_count + 0.5) / n
                p0.extend([np.max(y_data)/2, center_guess, 1.0])
                bounds_min.extend([0, np.min(x_data), 0.1])
                bounds_max.extend([np.inf, np.max(x_data), (np.max(x_data)-np.min(x_data))/2])
                current_peak_count += 1

            try:
                popt, pcov = curve_fit(multi_gaussian_linear, x_data, y_data, p0=p0, bounds=(bounds_min, bounds_max), maxfev=10000)
                y_fit = multi_gaussian_linear(x_data, *popt)
                bic = calculate_bic(y_data, y_fit, len(popt))
                
                if bic < min_bic:
                    min_bic = bic
                    best_result = {'n': n, 'popt': popt, 'y_fit': y_fit, 'bic': bic}
            except RuntimeError:
                self.log(f"  -> Failed to fit {n} peaks.")

        # 結果表示
        if best_result:
            self.log(f"\n=== BEST RESULT FOUND ===")
            self.log(f"Optimal Peaks: {best_result['n']}")
            self.log(f"BIC Score: {best_result['bic']:.2f}")
            
            params = best_result['popt']
            self.ax.clear()
            self.ax.plot(x_data, y_data, 'ko', markersize=3, alpha=0.4, label='Raw Data')
            self.ax.plot(x_data, best_result['y_fit'], 'r-', linewidth=2, label='Total Fit')
            
            bg = params[0] * x_data + params[1]
            self.ax.plot(x_data, bg, 'k--', alpha=0.6, label='Background')
            
            self.log(f"Background: y = {params[0]:.2e}x + {params[1]:.2e}")
            
            for i in range(best_result['n']):
                amp = params[2 + 3*i]
                cen = params[2 + 3*i + 1]
                wid = params[2 + 3*i + 2]
                
                self.log(f"Peak {i+1}: {cen:.2f} eV (Wid: {wid:.2f}, Amp: {amp:.0f})")
                
                p_params = [0, 0, amp, cen, wid]
                y_p = multi_gaussian_linear(x_data, *p_params)
                self.ax.fill_between(x_data, bg, bg + y_p, alpha=0.4, label=f'Peak {i+1}')

            self.ax.set_xlabel(x_name)
            self.ax.set_ylabel(y_name)
            self.ax.set_title(f"Fit Result: {best_result['n']} Peaks (BIC Best)")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()
            
        else:
            self.log("Analysis Failed.")
            messagebox.showwarning("Warning", "Fitting could not converge.")

        self.btn_run.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = XPSAnalyzerApp(root)
    root.mainloop()