import os
import csv
import cv2
import time
import numpy as np
from .feature_extraction import FeatureExtractor
from .utils import ensure_dir, normalize_coords

def _grid_points(cols=3, rows=3, margin=0.15):
    xs = np.linspace(margin, 1.0 - margin, cols)
    ys = np.linspace(margin, 1.0 - margin, rows)
    grid = [(x, y) for y in ys for x in xs]  # row-major
    return grid

def run_calibration(window_size=(1280, 720), points_per_grid=3,
                    dwell_frames=18, warmup_frames=8, out_csv="data/calibration_samples.csv"):
    w, h = window_size
    ensure_dir("data")
    points = _grid_points(points_per_grid, points_per_grid, margin=0.15)

    fe = FeatureExtractor()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    samples = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    try:
        for i, (nx, ny) in enumerate(points, start=1):
            cx, cy = int(nx * w), int(ny * h)
            # Warmup phase: show point, let user move eyes to it
            for f in range(warmup_frames):
                ret, frame = cap.read()
                if not ret: continue
                _draw_target(frame, (cx, cy), f"Point {i}/{len(points)} — get ready")
                cv2.imshow("Calibration", frame)
                if cv2.waitKey(1) == 27:  # ESC
                    _cleanup(cap, fe)
                    return

            # Dwell phase: collect features while user looks at the point
            collected = 0
            while collected < dwell_frames:
                ret, frame = cap.read()
                if not ret: continue
                feat, dbg = fe.features_from_frame(frame)
                _draw_target(frame, (cx, cy), f"Point {i}/{len(points)} — hold gaze")
                if feat is not None:
                    samples.append({
                        "feat": feat.tolist(),
                        "xy": [nx, ny]
                    })
                    collected += 1

                cv2.imshow("Calibration", frame)
                if cv2.waitKey(1) == 27:
                    _cleanup(cap, fe)
                    return

            # brief pause between targets
            _flash_ack(cap, (cx, cy), w, h)

        # Save CSV: 4 feat columns + 2 targets
        with open(out_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["f1","f2","f3","f4","tx","ty"])
            for s in samples:
                wr.writerow(s["feat"] + s["xy"])

        print(f"Saved {len(samples)} samples to {out_csv}")

    finally:
        _cleanup(cap, fe)

def _draw_target(frame, center, msg):
    cx, cy = center
    cv2.circle(frame, (cx, cy), 14, (255, 255, 255), -1)
    cv2.circle(frame, (cx, cy), 28, (255, 255, 255), 2)
    cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

def _flash_ack(cap, center, w, h, ms=250):
    t0 = time.time()
    while (time.time() - t0) * 1000 < ms:
        ret, frame = cap.read()
        if not ret: continue
        cx, cy = center
        cv2.circle(frame, (cx, cy), 40, (0, 255, 0), 3)
        cv2.imshow("Calibration", frame)
        cv2.waitKey(1)

def _cleanup(cap, fe):
    try: cap.release()
    except: pass
    try: fe.close()
    except: pass
    cv2.destroyAllWindows()
