import os
import csv
import numpy as np
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_model(csv_path="data/calibration_samples.csv",
                out_path="models/gaze_ridge_xy.joblib",
                test_size=0.2, random_state=42):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    X = data[:, :4]
    y = data[:, 4:6]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Two independent regressors (x and y) wrapped in a simple class
    model_x = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    model_y = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])

    model_x.fit(Xtr, ytr[:, 0])
    model_y.fit(Xtr, ytr[:, 1])

    r2x = model_x.score(Xte, yte[:, 0])
    r2y = model_y.score(Xte, yte[:, 1])
    print(f"Eval R^2 — x: {r2x:.3f}, y: {r2y:.3f}")

    dump({"model_x": model_x, "model_y": model_y}, out_path)
    print(f"Saved model to {out_path}")
