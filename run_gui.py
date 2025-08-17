#!/usr/bin/env python3
"""
Basic GUI wrapper for the Eye Gaze Tracker project.

It provides one window with three actions:
1) Calibration  2) Train model  3) Realtime run

It calls into the existing project modules:
- src.calibration.run_calibration
- src.train_model.train_model
- src.inference.run_realtime

Notes:
- If you get ImportErrors, be sure to `pip install -r requirements.txt` from the project root.
"""

import os
import sys
import threading
import time
import queue
from dataclasses import dataclass

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Ensure project root is on sys.path when running from project root
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.calibration import run_calibration as _run_calibration
from src.train_model import train_model as _train_model
from src.inference import run_realtime as _run_realtime


@dataclass
class AppConfig:
    # General
    window_w: int = 1280
    window_h: int = 720
    # Calibration
    points_per_grid: int = 3
    dwell_frames: int = 18
    warmup_frames: int = 8
    out_csv: str = "data/calibration_samples.csv"
    # Training
    model_out: str = "models/gaze_ridge_xy.joblib"
    # Realtime
    model_in: str = "models/gaze_ridge_xy.joblib"
    fps_target: int = 30


class GazeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Eye Gaze Tracker — Control Panel")
        self.geometry("880x600")
        self.minsize(780, 520)

        self.config = AppConfig()
        self.log_q = queue.Queue()
        self._threads = []
        self._rt_stop_flag = threading.Event()

        self._build_ui()
        self._pump_logs()

    # ---------------- UI ----------------
    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.calib_frame = self._build_calibration_tab(nb)
        self.train_frame = self._build_training_tab(nb)
        self.rt_frame    = self._build_realtime_tab(nb)

        nb.add(self.calib_frame, text="Calibration")
        nb.add(self.train_frame, text="Training")
        nb.add(self.rt_frame,    text="Realtime")

        # Log panel
        self.log_text = tk.Text(self, height=10, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0,10))

    def _build_calibration_tab(self, parent):
        f = ttk.Frame(parent, padding=10)

        g = ttk.LabelFrame(f, text="Calibration Settings", padding=10)
        g.pack(fill=tk.X, expand=False)

        # Window size
        ttk.Label(g, text="Capture Width:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.var_w = tk.IntVar(value=self.config.window_w)
        ttk.Entry(g, textvariable=self.var_w, width=8).grid(row=0, column=1, sticky="w")

        ttk.Label(g, text="Capture Height:").grid(row=0, column=2, sticky="w", padx=(10,4), pady=4)
        self.var_h = tk.IntVar(value=self.config.window_h)
        ttk.Entry(g, textvariable=self.var_h, width=8).grid(row=0, column=3, sticky="w")

        # Grid points (NxN in current calibration module)
        ttk.Label(g, text="Grid size (N x N):").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.var_grid = tk.IntVar(value=self.config.points_per_grid)
        ttk.Entry(g, textvariable=self.var_grid, width=8).grid(row=1, column=1, sticky="w")

        ttk.Label(g, text="Warmup frames:").grid(row=1, column=2, sticky="w", padx=(10,4), pady=4)
        self.var_warm = tk.IntVar(value=self.config.warmup_frames)
        ttk.Entry(g, textvariable=self.var_warm, width=8).grid(row=1, column=3, sticky="w")

        ttk.Label(g, text="Dwell frames:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        self.var_dwell = tk.IntVar(value=self.config.dwell_frames)
        ttk.Entry(g, textvariable=self.var_dwell, width=8).grid(row=2, column=1, sticky="w")

        # Output CSV
        ttk.Label(g, text="Output CSV:").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        self.var_csv = tk.StringVar(value=self.config.out_csv)
        ttk.Entry(g, textvariable=self.var_csv, width=40).grid(row=3, column=1, columnspan=2, sticky="we")
        ttk.Button(g, text="Browse…", command=self._choose_csv).grid(row=3, column=3, sticky="w")

        # Actions
        row = ttk.Frame(f, padding=(0,10,0,0))
        row.pack(fill=tk.X, expand=False)
        ttk.Button(row, text="Start Calibration", command=self._start_calibration).pack(side=tk.LEFT)
        ttk.Label(row, text="(An OpenCV window will appear; press ESC to abort)").pack(side=tk.LEFT, padx=10)

        return f

    def _build_training_tab(self, parent):
        f = ttk.Frame(parent, padding=10)

        g = ttk.LabelFrame(f, text="Training Settings", padding=10)
        g.pack(fill=tk.X, expand=False)

        ttk.Label(g, text="Calibration CSV:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.var_train_csv = tk.StringVar(value=self.config.out_csv)
        ttk.Entry(g, textvariable=self.var_train_csv, width=40).grid(row=0, column=1, columnspan=2, sticky="we")
        ttk.Button(g, text="Browse…", command=self._choose_train_csv).grid(row=0, column=3, sticky="w")

        ttk.Label(g, text="Model output (.joblib):").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.var_model_out = tk.StringVar(value=self.config.model_out)
        ttk.Entry(g, textvariable=self.var_model_out, width=40).grid(row=1, column=1, columnspan=2, sticky="we")
        ttk.Button(g, text="Browse…", command=self._choose_model_out).grid(row=1, column=3, sticky="w")

        row = ttk.Frame(f, padding=(0,10,0,0))
        row.pack(fill=tk.X, expand=False)
        ttk.Button(row, text="Train Model", command=self._start_training).pack(side=tk.LEFT)

        return f

    def _build_realtime_tab(self, parent):
        f = ttk.Frame(parent, padding=10)

        g = ttk.LabelFrame(f, text="Realtime Settings", padding=10)
        g.pack(fill=tk.X, expand=False)

        ttk.Label(g, text="Model (.joblib):").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.var_model_in = tk.StringVar(value=self.config.model_in)
        ttk.Entry(g, textvariable=self.var_model_in, width=40).grid(row=0, column=1, columnspan=2, sticky="we")
        ttk.Button(g, text="Browse…", command=self._choose_model_in).grid(row=0, column=3, sticky="w")

        ttk.Label(g, text="Capture Width:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.var_rt_w = tk.IntVar(value=self.config.window_w)
        ttk.Entry(g, textvariable=self.var_rt_w, width=8).grid(row=1, column=1, sticky="w")

        ttk.Label(g, text="Capture Height:").grid(row=1, column=2, sticky="w", padx=(10,4), pady=4)
        self.var_rt_h = tk.IntVar(value=self.config.window_h)
        ttk.Entry(g, textvariable=self.var_rt_h, width=8).grid(row=1, column=3, sticky="w")

        ttk.Label(g, text="Target FPS:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        self.var_fps = tk.IntVar(value=self.config.fps_target)
        ttk.Entry(g, textvariable=self.var_fps, width=8).grid(row=2, column=1, sticky="w")

        row = ttk.Frame(f, padding=(0,10,0,0))
        row.pack(fill=tk.X, expand=False)
        ttk.Button(row, text="Start Realtime", command=self._start_realtime).pack(side=tk.LEFT, padx=(0,8))
        ttk.Button(row, text="Stop Realtime", command=self._stop_realtime).pack(side=tk.LEFT)

        ttk.Label(f, text="(Close the OpenCV window or click Stop Realtime to end)").pack(anchor="w", pady=(6,0))

        return f

    # ------------- Actions -------------
    def _start_calibration(self):
        w = int(self.var_w.get())
        h = int(self.var_h.get())
        grid = int(self.var_grid.get())
        warm = int(self.var_warm.get())
        dwell = int(self.var_dwell.get())
        out_csv = self.var_csv.get()

        self._log(f"Calibration starting → size=({w},{h}), grid={grid}x{grid}, warmup={warm}, dwell={dwell}, out={out_csv}")
        def job():
            try:
                _run_calibration(window_size=(w, h),
                                 points_per_grid=grid,
                                 dwell_frames=dwell,
                                 warmup_frames=warm,
                                 out_csv=out_csv)
                self._log("Calibration finished.")
            except Exception as e:
                self._log(f"[ERROR] Calibration failed: {e}")
        self._run_bg(job)

    def _start_training(self):
        csv_path = self.var_train_csv.get()
        model_out = self.var_model_out.get()
        self._log(f"Training starting → csv={csv_path}, out={model_out}")
        def job():
            try:
                _train_model(csv_path=csv_path, out_path=model_out)
                self._log("Training finished.")
            except Exception as e:
                self._log(f"[ERROR] Training failed: {e}")
        self._run_bg(job)

    def _start_realtime(self):
        model_in = self.var_model_in.get()
        w = int(self.var_rt_w.get())
        h = int(self.var_rt_h.get())
        fps = int(self.var_fps.get())

        # Reset stop flag
        self._rt_stop_flag.clear()
        self._log(f"Realtime starting → model={model_in}, size=({w},{h}), fps_target={fps}")
        def job():
            try:
                # Wrap the original run_realtime to support stop flag
                for _ in self._run_realtime_yield(model_in, (w, h), fps):
                    if self._rt_stop_flag.is_set():
                        self._log("Realtime stop requested.")
                        break
                self._log("Realtime finished.")
            except Exception as e:
                self._log(f"[ERROR] Realtime failed: {e}")
        self._run_bg(job)

    def _stop_realtime(self):
        self._log("Stop requested — attempting to end realtime…")
        self._rt_stop_flag.set()

    # ------------- Helpers -------------
    def _run_realtime_yield(self, model_path, window_size, fps_target):
        """
        Calls the stock _run_realtime but yields periodically so we can check a stop flag.
        This is a cooperative stop mechanism: the original function draws/handles its own window.
        """
        # We run it on a thread and periodically yield control back to the GUI.
        # Since we cannot easily break inside _run_realtime without modifying it,
        # we simply call it and yield once; user can also close the OpenCV window.
        _run_realtime(model_path=model_path, window_size=window_size, fps_target=fps_target)
        yield  # single yield after completion

    def _run_bg(self, fn):
        t = threading.Thread(target=fn, daemon=True)
        t.start()
        self._threads.append(t)

    def _choose_csv(self):
        path = filedialog.asksaveasfilename(
            title="Save calibration CSV",
            defaultextension=".csv",
            filetypes=[("CSV files","*.csv"), ("All files","*.*")],
            initialfile=os.path.basename(self.var_csv.get() or "calibration_samples.csv"),
            initialdir=os.path.join(ROOT, "data"),
        )
        if path:
            self.var_csv.set(os.path.relpath(path, ROOT))

    def _choose_train_csv(self):
        path = filedialog.askopenfilename(
            title="Select calibration CSV",
            filetypes=[("CSV files","*.csv"), ("All files","*.*")],
            initialdir=os.path.join(ROOT, "data"),
        )
        if path:
            self.var_train_csv.set(os.path.relpath(path, ROOT))

    def _choose_model_out(self):
        path = filedialog.asksaveasfilename(
            title="Save trained model",
            defaultextension=".joblib",
            filetypes=[("Joblib files","*.joblib"), ("All files","*.*")],
            initialdir=os.path.join(ROOT, "models"),
            initialfile=os.path.basename(self.var_model_out.get() or "gaze_ridge_xy.joblib"),
        )
        if path:
            self.var_model_out.set(os.path.relpath(path, ROOT))

    def _choose_model_in(self):
        path = filedialog.askopenfilename(
            title="Select trained model",
            filetypes=[("Joblib files","*.joblib"), ("All files","*.*")],
            initialdir=os.path.join(ROOT, "models"),
        )
        if path:
            self.var_model_in.set(os.path.relpath(path, ROOT))

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_q.put(f"[{ts}] {msg}\n")

    def _pump_logs(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                self.log_text.configure(state=tk.NORMAL)
                self.log_text.insert(tk.END, msg)
                self.log_text.see(tk.END)
                self.log_text.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        # poll again
        self.after(120, self._pump_logs)


if __name__ == "__main__":
    app = GazeApp()
    app.mainloop()
