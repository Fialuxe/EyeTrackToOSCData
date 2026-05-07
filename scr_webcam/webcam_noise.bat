
@echo off
echo "Starting Calibration and Executing..."
echo "Each command may take up to a few minutes."

py webcam_gaze_tracker.py --calibrate --device cuda 

if %errorlevel% neq 0 (
    echo "Calibration failed. Please report to the person in charge of the experiment."
    pause
    exit /b
)

echo "Starting Preview..."
py webcam_gaze_tracker.py --preview --osc-port 8000 --device cuda 

pause