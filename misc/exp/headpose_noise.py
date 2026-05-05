import os
import cv2
import argparse
import numpy as np
from pythonosc import udp_client

from eyetrax import GazeEstimator, run_9_point_calibration
from eyetrax.calibration import run_5_point_calibration, run_lissajous_calibration
import eyetrax.calibration as calib_module
from eyetrax.filters import KalmanSmoother, NoSmoother, make_kalman
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, iter_frames

OSC_IP = "127.0.0.1"
OSC_PORT = 8000
SHOW_GUI = True
REF_POSE_FRAMES = 60
MAX_HEAD_ANGLE_RAD = np.radians(55)


def get_args():
    parser = argparse.ArgumentParser(description="EyeTrax OSC Server with Head Pose Compensation")
    parser.add_argument("--filter", choices=["kalman", "kalman_ema", "none"], default="none")
    parser.add_argument("--ema-alpha", type=float, default=0.25)
    parser.add_argument("--comp-x", type=float, default=0.0, help="Yaw compensation gain (pixels/rad); auto-calibrated")
    parser.add_argument("--comp-y", type=float, default=0.0, help="Pitch compensation gain (pixels/rad); auto-calibrated")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--calibration", choices=["9p", "5p", "lissajous", "dense"], default="9p")
    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--model", type=str, default="ridge")
    parser.add_argument("--model-file", type=str, default=None)
    return parser.parse_args()


def setup_gaze_estimator(args):
    estimator = GazeEstimator(model_name=args.model)
    if args.model_file and os.path.isfile(args.model_file):
        estimator.load_model(args.model_file)
    else:
        execute_calibration(estimator, args)
        if args.model_file:
            estimator.save_model(args.model_file)
    return estimator


def execute_calibration(estimator, args):
    method = args.calibration
    if method == "dense":
        dense_func = getattr(calib_module, "run_dense_calibration", None)
        if dense_func:
            dense_func(estimator, camera_index=args.camera, rows=args.grid_rows, cols=args.grid_cols)
            return
    if method == "5p":
        run_5_point_calibration(estimator, camera_index=args.camera)
        return
    if method == "lissajous":
        run_lissajous_calibration(estimator, camera_index=args.camera)
        return
    run_9_point_calibration(estimator, camera_index=args.camera)


def apply_ema(last_val, current_val, alpha):
    if last_val is None:
        return current_val
    return alpha * current_val + (1.0 - alpha) * last_val


def capture_reference_pose(estimator, cap):
    """
    Collects REF_POSE_FRAMES of valid face detections and returns the mean
    (yaw, pitch) in radians.  eyetrax appends [yaw, pitch, roll] at features[-3:].
    Call this right after calibration while the user looks straight ahead.
    """
    print(f"\n[Ref Pose] Look straight at the screen. Capturing {REF_POSE_FRAMES} frames...")
    yaws, pitches = [], []
    canvas = np.zeros((120, 400, 3), dtype=np.uint8)

    for frame in iter_frames(cap):
        features, _ = estimator.extract_features(frame)
        if features is not None:
            yaw_r, pitch_r = float(features[-3]), float(features[-2])
            if abs(yaw_r) < MAX_HEAD_ANGLE_RAD and abs(pitch_r) < MAX_HEAD_ANGLE_RAD:
                yaws.append(yaw_r)
                pitches.append(pitch_r)

        canvas[:] = 30
        cv2.putText(canvas, "Look straight ahead!", (10, 35), 1, 1.5, (255, 255, 255), 2)
        bar_w = int(400 * len(yaws) / REF_POSE_FRAMES)
        cv2.rectangle(canvas, (10, 55), (10 + bar_w, 85), (0, 200, 100), -1)
        cv2.putText(canvas, f"{len(yaws)}/{REF_POSE_FRAMES}", (10, 110), 1, 1, (200, 200, 200), 1)
        cv2.imshow("Capturing Reference Pose", canvas)
        cv2.waitKey(1)

        if len(yaws) >= REF_POSE_FRAMES:
            break

    cv2.destroyWindow("Capturing Reference Pose")

    if yaws:
        ref_yaw = float(np.mean(yaws))
        ref_pitch = float(np.mean(pitches))
        print(f"[Ref Pose] Reference: yaw={np.degrees(ref_yaw):.1f}°  pitch={np.degrees(ref_pitch):.1f}°")
        return ref_yaw, ref_pitch

    print("[Ref Pose] No face detected. Using (0, 0).")
    return 0.0, 0.0


def run_osc_server():
    args = get_args()
    screen_width, screen_height = get_screen_size()
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    estimator = setup_gaze_estimator(args)

    if args.filter in ["kalman", "kalman_ema"]:
        smoother = KalmanSmoother(make_kalman())
        try:
            smoother.tune(estimator, camera_index=args.camera)
        except Exception:
            pass
    else:
        smoother = NoSmoother()

    last_gaze = [None, None]
    comp_x = args.comp_x
    comp_y = args.comp_y

    auto_calib_active = False
    calib_delta_yaw: list[float] = []
    calib_delta_pitch: list[float] = []
    calib_gaze_x: list[float] = []
    calib_gaze_y: list[float] = []

    # In-loop reference re-capture state
    ref_capture_buf_yaw: list[float] = []
    ref_capture_buf_pitch: list[float] = []
    ref_capture_remaining = 0

    print(f"OSC server started → {OSC_IP}:{OSC_PORT}")
    print("Keys: c=auto-calib  r=re-capture reference  ESC=quit")

    with camera(args.camera) as cap:
        ref_yaw, ref_pitch = capture_reference_pose(estimator, cap)

        for frame in iter_frames(cap):
            features, is_blinking = estimator.extract_features(frame)

            # --- Head pose from features[-3:] = [yaw, pitch, roll] in radians ---
            # eyetrax/gaze.py lines 79-82 appends these after the landmark coords.
            head_yaw = ref_yaw
            head_pitch = ref_pitch
            face_valid = features is not None

            if features is not None:
                yaw_r, pitch_r = float(features[-3]), float(features[-2])
                if abs(yaw_r) > MAX_HEAD_ANGLE_RAD or abs(pitch_r) > MAX_HEAD_ANGLE_RAD:
                    is_blinking = True
                    face_valid = False
                else:
                    head_yaw = yaw_r
                    head_pitch = pitch_r
                    osc_client.send_message("/head/rotation",
                        [head_yaw, head_pitch, float(features[-1])])

            # --- In-loop reference pose re-capture ('r' key) ---
            if ref_capture_remaining > 0 and face_valid:
                ref_capture_buf_yaw.append(head_yaw)
                ref_capture_buf_pitch.append(head_pitch)
                ref_capture_remaining -= 1
                if ref_capture_remaining == 0:
                    ref_yaw = float(np.mean(ref_capture_buf_yaw))
                    ref_pitch = float(np.mean(ref_capture_buf_pitch))
                    print(f"[Ref Pose] New reference: yaw={np.degrees(ref_yaw):.1f}°  pitch={np.degrees(ref_pitch):.1f}°")

            # --- Gaze prediction & compensation ---
            gaze_status = "BLINK/LOST"

            if not is_blinking and face_valid:
                raw_gaze = estimator.predict(np.array([features]))[0]

                # Delta from reference head pose — zero when head is in calibration position.
                # Using absolute yaw/pitch (as the original code did) would apply a large
                # correction even when the user hasn't moved from their natural pose.
                delta_yaw = head_yaw - ref_yaw
                delta_pitch = head_pitch - ref_pitch

                if auto_calib_active:
                    calib_delta_yaw.append(delta_yaw)
                    calib_delta_pitch.append(delta_pitch)
                    calib_gaze_x.append(raw_gaze[0])
                    calib_gaze_y.append(raw_gaze[1])

                # Cancel the linear coupling between head-pose delta and gaze drift.
                # comp_x / comp_y are fitted by auto-calib; both start at 0 (no effect).
                compensated_x = raw_gaze[0] + delta_yaw * comp_x
                compensated_y = raw_gaze[1] - delta_pitch * comp_y

                px, py = smoother.step(int(compensated_x), int(compensated_y))
                if px is not None:
                    norm_x_raw = px / screen_width
                    norm_y_raw = py / screen_height
                    if args.filter == "kalman_ema":
                        last_gaze[0] = apply_ema(last_gaze[0], norm_x_raw, args.ema_alpha)
                        last_gaze[1] = apply_ema(last_gaze[1], norm_y_raw, args.ema_alpha)
                    else:
                        last_gaze[0], last_gaze[1] = norm_x_raw, norm_y_raw

                    norm_x = float(np.clip(last_gaze[0], 0.0, 1.0))
                    norm_y = float(np.clip(last_gaze[1], 0.0, 1.0))
                    osc_client.send_message("/gaze", [norm_x, norm_y, 0.0])
                    gaze_status = f"({norm_x:.2f}, {norm_y:.2f})"
            else:
                try:
                    smoother.step(None, None)
                except Exception:
                    pass
                osc_client.send_message("/gaze/blink", 1.0)

            # --- GUI ---
            if SHOW_GUI:
                canvas = np.ones((360, 340, 3), dtype=np.uint8) * 30
                dy_deg = np.degrees(head_yaw - ref_yaw)
                dp_deg = np.degrees(head_pitch - ref_pitch)
                pose_color = (0, 100, 255) if (not face_valid or is_blinking) else (0, 255, 0)

                cv2.putText(canvas, f"Gaze: {gaze_status}", (10, 150), 1, 1, (255, 255, 255), 1)
                cv2.putText(canvas, f"dYaw:{dy_deg:+.1f}  dPitch:{dp_deg:+.1f}", (10, 175), 1, 1, pose_color, 1)
                cv2.putText(canvas,
                    f"Ref  yaw:{np.degrees(ref_yaw):.1f}  pitch:{np.degrees(ref_pitch):.1f}",
                    (10, 200), 1, 1, (120, 120, 120), 1)
                cv2.putText(canvas, f"Comp-X:{comp_x:.2f}  Comp-Y:{comp_y:.2f}", (10, 230), 1, 1, (200, 200, 100), 1)

                if ref_capture_remaining > 0:
                    cv2.putText(canvas, f"Re-capturing ref... {REF_POSE_FRAMES - ref_capture_remaining}/{REF_POSE_FRAMES}",
                        (10, 285), 1, 1, (0, 200, 255), 1)
                    cv2.putText(canvas, "Look straight ahead!", (10, 305), 1, 1, (0, 200, 255), 1)
                elif auto_calib_active:
                    cv2.putText(canvas, f"AUTO CALIB: {len(calib_delta_yaw)} frames", (10, 285), 1, 1, (0, 255, 255), 1)
                    cv2.putText(canvas, "Look at CENTER, move head", (10, 305), 1, 1, (0, 255, 255), 1)
                else:
                    cv2.putText(canvas, "c=auto-calib  r=re-ref  ESC=quit", (10, 305), 1, 1, (100, 100, 100), 1)

                cv2.imshow("EyeTrax Debug", canvas)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break

                elif key == ord("r"):
                    print("[Ref Pose] Re-capturing reference. Look straight ahead.")
                    ref_capture_buf_yaw.clear()
                    ref_capture_buf_pitch.clear()
                    ref_capture_remaining = REF_POSE_FRAMES

                elif key == ord("c"):
                    if not auto_calib_active:
                        print("\n[Auto-Calib] Started. Look at the screen center and move your head around.")
                        calib_delta_yaw.clear()
                        calib_delta_pitch.clear()
                        calib_gaze_x.clear()
                        calib_gaze_y.clear()
                        auto_calib_active = True
                    else:
                        auto_calib_active = False
                        n = len(calib_delta_yaw)
                        if n > 30:
                            # Model: raw_gaze = true_gaze + m * delta_pose + b
                            # Compensation cancels the m term:
                            #   compensated_x = raw + delta_yaw * comp_x  → comp_x = -m_x
                            #   compensated_y = raw - delta_pitch * comp_y → comp_y = +m_y
                            m_x, _ = np.polyfit(calib_delta_yaw, calib_gaze_x, 1)
                            m_y, _ = np.polyfit(calib_delta_pitch, calib_gaze_y, 1)
                            comp_x = -m_x
                            comp_y = m_y
                            print(f"[Auto-Calib] Done! N={n}")
                            print(f"[Auto-Calib] comp_x={comp_x:.2f}  comp_y={comp_y:.2f}")
                        else:
                            print(f"[Auto-Calib] Not enough data ({n} frames, need > 30). Move your head more.")


if __name__ == "__main__":
    run_osc_server()
