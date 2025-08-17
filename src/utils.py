import os
import time
import numpy as np

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def now_ms() -> int:
    return int(time.time() * 1000)

def ema(prev, new, alpha=0.3):
    if prev is None:
        return new
    return (1 - alpha) * prev + alpha * new

def normalize_coords(x, y, w, h):
    return x / float(w), y / float(h)

def denormalize_coords(nx, ny, w, h):
    return int(nx * w), int(ny * h)

def feature_stack(samples):
    """Convert list of dicts with 'feat' and 'xy' to arrays."""
    X = np.array([s["feat"] for s in samples], dtype=np.float32)
    y = np.array([s["xy"]   for s in samples], dtype=np.float32)
    return X, y
