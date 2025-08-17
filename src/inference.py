import cv2
import time
import numpy as np
from joblib import load
from .feature_extraction import FeatureExtractor
from .utils import ema, denormalize_coords

def run_realtime(model_path="models/gaze_ridge_xy.joblib",
                 window_size=(1280, 720), fps_target=30):
    w, h = window_size
    pack = load(model_path)
    mx, my = pack["model_x"], pack["model_y"]

    fe = FeatureExtractor()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    pred_smooth = None
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret: continue

            feat, dbg = fe.features_from_frame(frame)
            if feat is not None:
                nx = float(mx.predict(feat.reshape(1, -1))[0])
                ny = float(my.predict(feat.reshape(1, -1))[0])
                pred = np.array([nx, ny], dtype=np.float32)
                pred_smooth = ema(pred_smooth, pred, alpha=0.25)
                px, py = denormalize_coords(pred_smooth[0], pred_smooth[1], w, h)

                _draw_hud(frame, (px, py), dbg)

            cv2.imshow("Realtime Gaze", frame)
            if cv2.waitKey(1) == 27:  # ESC
                break

            # basic FPS cap
            dt = time.time() - t0
            wait = max(0, (1.0 / fps_target) - dt)
            if wait > 0:
                time.sleep(wait)
    finally:
        try: cap.release()
        except: pass
        try: fe.close()
        except: pass
        cv2.destroyAllWindows()

def _draw_hud(frame, pt, dbg):
    px, py = pt
    cv2.circle(frame, (px, py), 10, (0, 255, 0), -1)
    cv2.putText(frame, "Predicted gaze", (px + 12, py - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    if dbg is not None:
        lc, rc = dbg["lc"], dbg["rc"]
        cv2.circle(frame, (int(lc[0]), int(lc[1])), 4, (255, 255, 0), -1)
        cv2.circle(frame, (int(rc[0]), int(rc[1])), 4, (255, 255, 0), -1)
        cv2.putText(frame, "Eye centers", (int(rc[0])+10, int(rc[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
