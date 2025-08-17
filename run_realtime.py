from src.inference import run_realtime

if __name__ == "__main__":
    run_realtime(
        model_path="models/gaze_ridge_xy.joblib",
        window_size=(1280, 720),
        fps_target=30
    )
