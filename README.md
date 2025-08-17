# 👁Eye Gaze Tracker

A Python-based **real-time eye gaze tracking system** using **OpenCV**, **MediaPipe**, and **scikit-learn**.  
The project lets you calibrate gaze points, train a regression model, and run real-time inference through a **GUI control panel** or through the **command line**.

---

## Features
- **Calibration** with a grid of points  
- **Feature extraction** from eye landmarks using MediaPipe FaceMesh  
- **Supervised training** with Ridge Regression  
- **Real-time gaze prediction** with smoothing  
- **Tkinter GUI** to control calibration, training, and real-time modes  

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Dependencies (from [`requirements.txt`](requirements.txt)):

* `opencv-python`
* `mediapipe`
* `numpy`
* `scikit-learn`
* `joblib`

---

## Usage

### 1. Run the GUI

From the project root:

```bash
py run_gui.py
```

The GUI has 3 tabs:

1. **Calibration** → Runs calibration and saves samples to `data/calibration_samples.csv`
2. **Training** → Trains a Ridge regression model and saves it to `models/gaze_ridge_xy.joblib`
3. **Realtime** → Runs live gaze tracking with the trained model

Press **ESC** to stop calibration or real-time windows.

---

### 2. Command-line

**Calibration**

```bash
py run_calibration.py
```

This opens a calibration window and saves gaze samples to `data/calibration_samples.csv`.

**Training**

```bash
py train.py
```

This trains a Ridge regression model and saves it under `models/gaze_ridge_xy.joblib`.

**Realtime Inference**

```bash
py run_realtime.py
```

This opens a webcam feed with gaze predictions overlaid.

---

## Project Structure

```
eye-gaze-tracker/
│
├── requirements.txt         
├── run_gui.py               
├── run_calibration.py       
├── train.py                 
├── run_realtime.py          
│
├── src/                     
│   ├── calibration.py       
│   ├── feature_extraction.py
│   ├── inference.py         
│   ├── train_model.py       
│   └── utils.py             
│
├── data/                    
├── models/                  
└── README.md                
```

---

## How it Works

1. **Calibration Phase**

   * User looks at grid points on screen.
   * Eye features are extracted and normalized by interocular distance.
   * Features + known point coordinates are stored in a CSV.

2. **Training Phase**

   * Ridge Regression models learn mappings from features → (x, y) coordinates.
   * Models are serialized with Joblib.

3. **Realtime Phase**

   * Webcam frames are processed in real time.
   * Eye features are extracted, passed to trained models, and smoothed.
   * Predicted gaze point is drawn on the screen.

---

## Notes

* Ensure **good lighting** and sit directly facing the webcam for best accuracy.
* The **calibration grid** can be adjusted in code (e.g., 3×3 vs 5×5).
* Recalibrate if predictions drift.
* Empty `data/` and `models/` folders are included — these will fill when you run calibration and training.

---

## Future Improvements

* Switch to **non-linear models** (MLP, neural nets) for better accuracy.
* Use **more landmarks** (nose, mouth, cheeks) to handle head pose.
* Implement **yaw, pitch, roll compensation**.
* Provide a modern GUI with **PyQt** or web-based dashboard.
