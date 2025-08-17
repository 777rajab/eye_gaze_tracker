from src.calibration import run_calibration

if __name__ == "__main__":
    run_calibration(
        window_size=(1280, 720),   # change if you like
        points_per_grid=3,         # 3x3 grid
        dwell_frames=18,           # ~0.6s at 30 FPS
        warmup_frames=8,           # small delay when point appears
        out_csv="data/calibration_samples.csv"
    )
