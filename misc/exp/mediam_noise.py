import os
import time
import cv2
import argparse  # ライブラリのパーサーの代わりに標準ライブラリを使用
import numpy as np
from pythonosc import udp_client

from eyetrax import GazeEstimator, run_9_point_calibration
from eyetrax.calibration import run_5_point_calibration, run_lissajous_calibration
import eyetrax.calibration as calib_module

from eyetrax.filters import KalmanSmoother, KDESmoother, NoSmoother, make_kalman
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, iter_frames

# --- Configuration ---
OSC_IP = "127.0.0.1"
OSC_PORT = 8000
TARGET_FPS = 30.0

# GUIを表示するかどうかのスイッチ
SHOW_GUI = True

def get_args():
    """
    EyeTrax内蔵の古いパーサーを破棄し、ドキュメントの最新仕様に合わせた
    カスタムパーサーを構築します。
    """
    parser = argparse.ArgumentParser(description="Optimized EyeTrax OSC Server")
    parser.add_argument("--filter", choices=["kalman", "kalman_ema", "kde", "none"], default="none")
    parser.add_argument("--ema-alpha", type=float, default=0.25)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--calibration", choices=["9p", "5p", "lissajous", "dense"], default="9p")
    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="ridge")
    parser.add_argument("--model-file", type=str, default=None)
    return parser.parse_args()


def execute_calibration(estimator, args):
    method = args.calibration
    if method == "5p":
        run_5_point_calibration(estimator, camera_index=args.camera)
        return
    if method == "lissajous":
        run_lissajous_calibration(estimator, camera_index=args.camera)
        return
    if method == "dense":
        dense_func = getattr(calib_module, 'run_dense_calibration', None)
        if dense_func:
            dense_func(estimator, camera_index=args.camera, rows=args.grid_rows, cols=args.grid_cols)
            return
        print("[Warning] 'dense' calibration API not found. Falling back to 9-point.")
    run_9_point_calibration(estimator, camera_index=args.camera)

def setup_gaze_estimator(args):
    estimator = GazeEstimator(model_name=args.model)
    if args.model_file and os.path.isfile(args.model_file):
        estimator.load_model(args.model_file)
        return estimator
    execute_calibration(estimator, args)
    if args.model_file:
        estimator.save_model(args.model_file)
    return estimator

def create_smoother(args, screen_width, screen_height, estimator):
    if args.filter == "kde":
        return KDESmoother(screen_width, screen_height, confidence=args.confidence)
        
    if args.filter in ["kalman", "kalman_ema"]:
        use_ema = (args.filter == "kalman_ema")
        try:
            from eyetrax.filters import KalmanEMASmoother
            smoother = KalmanEMASmoother(make_kalman(), alpha=args.ema_alpha)
        except ImportError:
            smoother = KalmanSmoother(make_kalman())
            if use_ema:
                setattr(smoother, 'custom_ema_alpha', args.ema_alpha)
        try:
            smoother.tune(estimator, camera_index=args.camera)
        except Exception:
            pass
        return smoother
        
    return NoSmoother()

def apply_custom_ema(smoother, current_x, current_y):
    if not hasattr(smoother, 'custom_ema_alpha'):
        return current_x, current_y
    if not hasattr(smoother, 'last_ema_x'):
        smoother.last_ema_x, smoother.last_ema_y = current_x, current_y
        return current_x, current_y
    alpha = smoother.custom_ema_alpha
    smoothed_x = alpha * current_x + (1.0 - alpha) * smoother.last_ema_x
    smoothed_y = alpha * current_y + (1.0 - alpha) * smoother.last_ema_y
    smoother.last_ema_x, smoother.last_ema_y = smoothed_x, smoothed_y
    return smoothed_x, smoothed_y

def draw_debug_gui(gaze_status):
    canvas = np.ones((400, 320, 3), dtype=np.uint8) * 30
    cv2.rectangle(canvas, (40, 40), (280, 200), (100, 255, 100), 2)
    
    if gaze_status.startswith("("):
        try:
            coords = gaze_status.strip("()").split(", ")
            gx = int(40 + float(coords[0]) * 240)
            gy = int(40 + float(coords[1]) * 160)
            cv2.circle(canvas, (gx, gy), 8, (0, 255, 255), -1)
        except ValueError:
            pass
            
    cv2.putText(canvas, f"Gaze: {gaze_status}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "Press ESC to exit", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow("OSC Gaze Streamer", canvas)
    
    return cv2.waitKey(1) == 27

def run_osc_server():
    args = get_args()  # 新しいパーサーを使用
    
    screen_width, screen_height = get_screen_size()
    if screen_width == 0 or screen_height == 0:
        raise ValueError("Invalid screen dimensions.")

    estimator = setup_gaze_estimator(args)
    smoother = create_smoother(args, screen_width, screen_height, estimator)
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    
    frame_interval = 1.0 / TARGET_FPS
    last_send_time = 0.0

    print(f"OSC server running at {OSC_IP}:{OSC_PORT}...")

    with camera(args.camera) as cap:
        for frame in iter_frames(cap):
            current_time = time.time()
            if current_time - last_send_time < frame_interval:
                continue
            last_send_time = current_time
            
            features, is_blinking = estimator.extract_features(frame)
            gaze_status = "NO_FEATURES"

            if is_blinking or features is None:
                try: smoother.step(None, None) 
                except Exception: pass
                osc_client.send_message("/gaze/blink", 1.0)
                gaze_status = "BLINK" if is_blinking else "NO_FEATURES"
            else:
                raw_gaze = estimator.predict(np.array([features]))[0]
                raw_x, raw_y = map(int, raw_gaze)

                if np.isfinite(raw_x) and np.isfinite(raw_y) and abs(raw_x) < 100000:
                    pred_x, pred_y = smoother.step(raw_x, raw_y)
                    
                    if pred_x is not None and pred_y is not None:
                        final_x, final_y = apply_custom_ema(smoother, pred_x, pred_y)
                        
                        norm_x = np.clip(final_x / screen_width, 0.0, 1.0)
                        norm_y = np.clip(final_y / screen_height, 0.0, 1.0)
                        
                        osc_client.send_message("/gaze", [float(norm_x), float(norm_y), 0.0])
                        gaze_status = f"({norm_x:.2f}, {norm_y:.2f})"
                    else:
                        gaze_status = "INVALID"

            if SHOW_GUI:
                if draw_debug_gui(gaze_status):
                    print("ESC pressed. Stopping...")
                    break

if __name__ == "__main__":
    run_osc_server()