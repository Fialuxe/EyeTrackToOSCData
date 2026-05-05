import argparse
import time
import csv
import cv2
import numpy as np
import os

try:
    from eyetrax.gaze import GazeEstimator
    from eyetrax.calibration import run_9_point_calibration
    from eyetrax.utils.video import camera, iter_frames
    from eyetrax.utils.screen import get_screen_size
    from eyetrax.filters import make_kalman, KalmanSmoother, NoSmoother
except ImportError:
    raise SystemExit("eyetrax not installed.")

def setup_kalman_smoother(gaze_estimator, camera_index):
    # Setup Kalman Filter with tuned parameters
    kalman = make_kalman()
    
    # Manual tuning for strong smoothing (Process < Measurement)
    # processNoiseCov: Trust model (smoothness) more
    cv2.setIdentity(kalman.processNoiseCov, 1e-5) 
    # measurementNoiseCov: Trust measurement (jittery input) less
    cv2.setIdentity(kalman.measurementNoiseCov, 1e-1)
    
    smoother = KalmanSmoother(kalman)
    
    # Optional: Run automatic tuning if desired, but we overwrite for demonstration
    # try:
    #     smoother.tune(gaze_estimator, camera_index=camera_index)
    # except Exception as e:
    #     print(f"Smoother tuning skipped: {e}")
        
    return smoother

def run_survey():
    parser = argparse.ArgumentParser(description="Record gaze data for survey.")
    parser.add_argument("--mode", choices=["kalman", "none"], default="none", help="Filtering mode")
    parser.add_argument("--output", default="survey_data.csv", help="Output CSV filename")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    output_filename = args.output
    mode = args.mode
    camera_index = args.camera

    # Initialize components
    gaze_estimator = GazeEstimator()
    
    print("=== Starting Calibration ===")
    run_9_point_calibration(gaze_estimator, camera_index=camera_index)
    print("=== Calibration Complete ===")

    screen_width, screen_height = get_screen_size()
    if screen_width == 0:
        screen_width, screen_height = 1920, 1080 # Fallback

    # Setup Smoother
    if mode == "kalman":
        smoother = setup_kalman_smoother(gaze_estimator, camera_index)
        print("Model: Kalman Filter (Tuned)")
    else:
        smoother = NoSmoother()
        print("Model: No Filter (Raw)")

    print(f"Recording to: {output_filename}")
    print("Press SPACE to Start Standardized Test")
    print("  - The target will move from Left to Right.")
    print("  - Follow the green circle.")
    print("Press ESC to Quit")

    # Open CSV
    with open(output_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "gaze_x", "gaze_y", "target_x", "target_y", "mode"])

        state = "IDLE" # IDLE, RUNNING, FINISHED
        start_time = 0
        duration = 4.0 # seconds for full sweep
        
        with camera(camera_index) as cap:
            for frame in iter_frames(cap):
                current_time = time.time()
                
                features, blink = gaze_estimator.extract_features(frame)
                
                # Make a black background for the stimulus (fullscreen effect)
                # We overlay the camera feed in corner or just use black screen with target
                stimulus_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
                
                # --- Gaze Processing ---
                current_gaze = None
                if features is not None and not blink:
                    gaze_point = gaze_estimator.predict(np.array([features]))[0]
                    raw_x, raw_y = gaze_point
                    
                    # Apply Filter
                    pred_x, pred_y = smoother.step(raw_x, raw_y)
                    
                    if pred_x is not None and pred_y is not None:
                        # Normalize 0.0 - 1.0 (Record normalized)
                        norm_x = pred_x / screen_width
                        norm_y = pred_y / screen_height
                        current_gaze = (norm_x, norm_y)

                        # Draw Gaze on Stimulus Frame (Blue cursor)
                        gx, gy = int(pred_x), int(pred_y)
                        cv2.circle(stimulus_frame, (gx, gy), 15, (255, 0, 0), 2) # Blue ring for gaze

                # --- Test Logic ---
                target_pos = None
                
                if state == "RUNNING":
                    elapsed = current_time - start_time
                    progress = elapsed / duration
                    
                    if progress >= 1.0:
                        state = "FINISHED"
                        print("Test Complete. saved.")
                        break
                    
                    # Moving Target Logic: Linear separate from 10% to 90% width
                    # Y fixed at 50% height
                    t_x = 0.1 + (0.8 * progress)
                    t_y = 0.5
                    target_pos = (t_x, t_y)
                    
                    # Draw Target (Green Filled Circle)
                    tx_pixel = int(t_x * screen_width)
                    ty_pixel = int(t_y * screen_height)
                    cv2.circle(stimulus_frame, (tx_pixel, ty_pixel), 20, (0, 255, 0), -1)
                    
                    # Record Data
                    if current_gaze:
                        writer.writerow([current_time, current_gaze[0], current_gaze[1], t_x, t_y, mode])
                    else:
                        # Record target even if gaze is missing (keeps timing consistent)
                        writer.writerow([current_time, "", "", t_x, t_y, mode])

                elif state == "IDLE":
                    # Show start instruction
                    cv2.putText(stimulus_frame, "Press SPACE to Start Test", (int(screen_width/2)-200, int(screen_height/2)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    
                # Setup Window to be fullscreen
                window_name = "Gaze Survey"
                cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow(window_name, stimulus_frame)

                key = cv2.waitKey(1)
                if key == 27: # ESC
                    break
                elif key == 32: # SPACE
                    if state == "IDLE":
                        state = "RUNNING"
                        start_time = time.time()
                        print("Test Started...")

if __name__ == "__main__":
    run_survey()
