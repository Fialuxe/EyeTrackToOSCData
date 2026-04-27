# Usage:
# python3 osc_server.py --camera 1 --filter kalman
#
# Data Format Specification:
# --------------------------------------------------------------------------------
# OSC Message Format (replaces osc for better reliability):
#   /gaze x y pupil - Eye gaze data (3 floats)
#     x: Horizontal gaze position, normalized [0.0, 1.0]
#     y: Vertical gaze position, normalized [0.0, 1.0]
#     pupil: Pupil size
#
#   /facemesh index x y z - Face mesh landmark data (4 values per landmark)
#     index: Landmark index [0-67]
#     x, y: Normalized position [0.0, 1.0]
#     z: Depth in meters
# --------------------------------------------------------------------------------
# This script runs the Eyetrax gaze estimation demo and streams gaze and MediaPipe FaceMesh data via OSC.

import os
import time
try: 
    import cv2
except ImportError as e:
    raise SystemExit("cv2 not installed. Run: pip install opencv-python") from e
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("mediapipe not installed. Run: pip install mediapipe") from e

# Eyetrax imports
try:
    from eyetrax.calibration import (
        run_5_point_calibration,
        run_9_point_calibration,
        run_lissajous_calibration,
    )
    from eyetrax.cli import parse_common_args
    from eyetrax.filters import KalmanSmoother, KDESmoother, NoSmoother, make_kalman
    from eyetrax.gaze import GazeEstimator
    from eyetrax.utils.draw import draw_cursor, make_thumbnail
    from eyetrax.utils.screen import get_screen_size
    from eyetrax.utils.video import camera, fullscreen, iter_frames
except ImportError as e:
    raise SystemExit("eyetrax not installed. Run: pip install eyetrax") from e

# OSC imports
try:
    from pythonosc import udp_client
except ImportError as e:
    raise SystemExit("pythonosc not installed. Run: pip install python-osc") from e

# --- OSC Configuration ---
OSC_IP = "127.0.0.1"  # localhost - Unity should be running on same machine
OSC_PORT = 8000       # Port Unity is listening on

# MediaPipe FaceMesh 68-point landmark mapping
FACEMESH_68_INDICES = [
    # Jawline (0-16): 17 points
    234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454,
    
    # Right eyebrow (17-21): 5 points
    70, 63, 105, 66, 107,
    
    # Left eyebrow (22-26): 5 points
    336, 296, 334, 293, 300,
    
    # Nose bridge (27-30): 4 points
    168, 6, 197, 195,
    
    # Nose bottom (31-35): 5 points
    5, 4, 1, 19, 94,
    
    # Right eye (36-41): 6 points
    33, 160, 158, 133, 153, 144,
    
    # Left eye (42-47): 6 points
    362, 385, 387, 263, 373, 380,
    
    # Outer lip (48-59): 12 points
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375,
    
    # Inner lip (60-67): 8 points
    78, 191, 80, 81, 82, 13, 312, 311
]

def create_osc_client():
    """
    Creates and returns an OSC UDP client for sending data to Unity.
    """
    print(f"Creating OSC client: {OSC_IP}:{OSC_PORT}")
    return udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)


def run_demo_with_osc():
    """
    Runs the Eyetrax demo and streams gaze and face mesh data via OSC.
    """
    args = parse_common_args()

    filter_method = args.filter
    camera_index = args.camera
    calibration_method = args.calibration
    background_path = args.background
    confidence_level = args.confidence

    gaze_estimator = GazeEstimator(model_name=args.model)

    if args.model_file and os.path.isfile(args.model_file):
        gaze_estimator.load_model(args.model_file)
        print(f"[demo] Loaded gaze model from {args.model_file}")
    else:
        if calibration_method == "9p":
            run_9_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "5p":
            run_5_point_calibration(gaze_estimator, camera_index=camera_index)
        else:
            run_lissajous_calibration(gaze_estimator, camera_index=camera_index)

    screen_width, screen_height = get_screen_size()
    if screen_width == 0 or screen_height == 0:
        raise ValueError("Could not get valid screen dimensions.")
    
    print(f"Screen resolution: {screen_width}x{screen_height}")

    if filter_method == "kalman":
        kalman = make_kalman()
        smoother = KalmanSmoother(kalman)
        try:
            smoother.tune(gaze_estimator, camera_index=camera_index)
        except Exception as e:
            print(f"Smoother tuning failed: {e}. Falling back to NoSmoother.", flush=True)
            smoother = NoSmoother()
    elif filter_method == "kde":
        smoother = KDESmoother(screen_width, screen_height, confidence=confidence_level)
    else:
        smoother = NoSmoother()

    if background_path and os.path.isfile(background_path):
        background = cv2.imread(background_path)
        background = cv2.resize(background, (screen_width, screen_height))
    else:
        background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        background[:] = (50, 50, 50)

    print(f"Opening camera {camera_index}...")

    # Create OSC client
    try:
        osc_client = create_osc_client()
        print("OSC client created successfully!")
    except Exception as e:
        print(f"Error creating OSC client: {e}")
        return

    print("Initializing FaceMesh...", flush=True)
    mp_face_mesh = mp.solutions.face_mesh
    with camera(camera_index) as cap, mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        last_print_time = time.time()
        frame_count = 0
        gaze_status = "N/A"
        face_status = "N/A"
        
        # Frame rate control
        target_fps = 30.0
        frame_interval = 1.0 / target_fps
        last_send_time = 0.0

        print("Entering frame loop...")
        has_received_frames = False

        for frame in iter_frames(cap):
            if not has_received_frames:
                print("First frame received!")
                has_received_frames = True
            frame_count += 1
            current_time = time.time()
            
            # Rate limiting
            if current_time - last_send_time < frame_interval:
                continue
            last_send_time = current_time
            
            features, blink_detected = gaze_estimator.extract_features(frame)

            # === GAZE PROCESSING ===
            try:
                if blink_detected:
                    gaze_x = np.clip(x_pred / screen_width, 0.0, 1.0)
                    gaze_y = np.clip(y_pred / screen_height, 0.0, 1.0)
                    osc_client.send_message("/gaze", [float(gaze_x), float(gaze_y), 1.0])
                    gaze_status = "BLINK"
                elif features is not None:
                    try:
                        gaze_point = gaze_estimator.predict(np.array([features]))[0]
                        x, y = map(int, gaze_point)
                        
                        if np.isfinite(x) and np.isfinite(y) and abs(x) < 100000 and abs(y) < 100000:
                            x_pred, y_pred = smoother.step(x, y)
                            
                            if x_pred is not None and y_pred is not None:
                                gaze_x = np.clip(x_pred / screen_width, 0.0, 1.0)
                                gaze_y = np.clip(y_pred / screen_height, 0.0, 1.0)
                                
                                osc_client.send_message("/gaze", [float(gaze_x), float(gaze_y), 0.0])
                                gaze_status = f"({gaze_x:.2f}, {gaze_y:.2f})"
                            else:
                                gaze_status = "INVALID"
                        else:
                            gaze_status = "OUT_OF_RANGE"
                    except Exception as e:
                        gaze_status = "NOT_CALIBRATED"
                        # print(f"Prediction error: {e}") # specific error suppression
                else:
                    gaze_status = "NO_FEATURES"
            except Exception as e:
                gaze_status = f"ERROR: {e}"

            # === FACEMESH PROCESSING ===
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    sent_count = 0
                    
                    for i, idx in enumerate(FACEMESH_68_INDICES):
                        lm = face_landmarks.landmark[idx]
                        
                        if np.isfinite(lm.x) and np.isfinite(lm.y) and np.isfinite(lm.z):
                            x = np.clip(lm.x, 0.0, 1.0)
                            y = np.clip(lm.y, 0.0, 1.0)
                            z = np.clip(lm.z, -1.0, 1.0)
                            
                            osc_client.send_message("/facemesh", [float(i), float(x), float(y), float(z)])
                            sent_count += 1
                    
                    face_status = f"{sent_count}/68"
                else:
                    face_status = "NO_FACE"
            except Exception as e:
                face_status = f"ERROR: {e}"

            # Status print
            if current_time - last_print_time >= 1.0:
                fps = frame_count / (current_time - last_print_time)
                print(f"\r[FPS: {fps:.1f}] Gaze: {gaze_status} | Face: {face_status}          ", end="", flush=True)
                last_print_time = current_time
                frame_count = 0

            # Simple visualization
            canvas = np.ones((400, 320, 3), dtype=np.uint8) * 30
            cv2.rectangle(canvas, (40, 40), (280, 200), (100, 255, 100), 2)
            
            if gaze_status.startswith("("):
                try:
                    coords = gaze_status.strip("()").split(", ")
                    gx = int(40 + float(coords[0]) * 240)
                    gy = int(40 + float(coords[1]) * 160)
                    cv2.circle(canvas, (gx, gy), 8, (0, 255, 255), -1)
                except:
                    pass
            
            cv2.putText(canvas, f"Gaze: {gaze_status}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(canvas, f"Face: {face_status}", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(canvas, "Press ESC to exit", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("OSC Gaze/Face Streamer", canvas)
            if cv2.waitKey(1) == 27:
                print("\nESC pressed. Stopping...")
                break
        
        if not has_received_frames:
            print("\nWARNING: No frames were received from the camera. Please check your camera connection or index.")

        print("\nDemo finished.")


if __name__ == "__main__":
    run_demo_with_osc()

