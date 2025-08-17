from src.train_model import train_model

if __name__ == "__main__":
    train_model(
        csv_path="data/calibration_samples.csv",
        out_path="models/gaze_ridge_xy.joblib",
        test_size=0.2,
        random_state=42
    )
