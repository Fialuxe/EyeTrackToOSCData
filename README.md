# EyeTrackToOSCData

A comprehensive gaze tracking solution that uses computer vision (L2CS-Net) to estimate eye gaze and head pose from a webcam and broadcasts the data via OSC (Open Sound Control) for use in other applications (such as VR, VTubing, or experimental setups).

## Features

- **Webcam-based Gaze Tracking**: Uses robust face detection (RetinaFace/MediaPipe) and L2CS-Net for high-accuracy pitch and yaw estimation.
- **Advanced Calibration**: 
  - 25-point GPR (Gaussian Process Regression) calibration with Lissajous head-movement correction (`gaze_tracker.py`).
  - 9-point Ridge Regression calibration (`L2csGazeOSC.py`).
- **Data Smoothing**: Implements Kalman and Exponential Moving Average (EMA) filters to reduce jitter and provide smooth, continuous gaze coordinates.
- **OSC Broadcasting**: Sends normalized screen coordinates and blink/face-loss detection over UDP to any local or remote application.

## Requirements

Ensure you have the following installed (preferably in a 64-bit Python environment for the webcam scripts):

- Python 3.8+ (64-bit)
- `opencv-python` (`cv2`)
- `torch` (PyTorch)
- `l2cs`
- `python-osc`
- `scikit-learn`
- `joblib`
- `numpy`

> **Note on Infrared Tracking**: Scripts located in `scr_infrared` may require a 32-bit Python environment depending on the specific eye-tracking camera SDK being used.

## Usage

### 1. High-Accuracy Tracker (`gaze_tracker.py`)

This script provides highly accurate gaze tracking by combining Gaussian Process Regression with Lissajous head-movement compensation.

**Step 1: Calibration**
```bash
python scr_webcam/gaze_tracker.py --calibrate
```
*Instructions: Look at the on-screen dots and press `SPACE` to record at each point. Afterwards, follow the moving Lissajous dot naturally with your head to calibrate head-movement correction.*

**Step 2: Run with OSC**
```bash
python scr_webcam/gaze_tracker.py --osc-port 8000 --preview
```

### 2. Hybrid Tracker (`L2csGazeOSC.py`)

An alternative pipeline using a 9-point Ridge Regression calibration grid.

**Step 1: Calibration**
```bash
python scr_webcam/L2csGazeOSC.py --calibrate --preview
```

**Step 2: Run with OSC**
```bash
python scr_webcam/L2csGazeOSC.py --osc-port 8000 --preview
```

## OSC Output Format

When running with an active OSC port (e.g., `--osc-port 8000`), the scripts send the following messages to the target IP (default: `127.0.0.1`):

- **Gaze Coordinates**: 
  ```text
  /gaze <float_x> <float_y> <float_z>
  ```
  `x` and `y` are normalized screen coordinates clamped between `0.0` and `1.0`. `z` is currently always `0.0`.

- **Face Loss / Blink Event**: 
  ```text
  /gaze/blink <float_value>
  ```
  Sent when the face tracker loses the face (which often corresponds to a blink or turning away). The value is typically `1.0`.