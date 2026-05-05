"""
gaze_osc/main.py — entry point

Usage:
    python main.py                                          # hybrid filter, CPU
    python main.py --filter kalman --device gpu:0
    python main.py --calibrate --screen-w 1920 --screen-h 1080
"""

import time
import argparse
import numpy as np
import cv2

from config           import Config
from capture          import FrameCapture
from gaze_estimator   import GazeEstimator
from face_pipeline    import FacePipeline
from head_pose        import HeadPoseEstimator
from blink_detector   import BlinkDetector
from screen_projector import ScreenProjector, angles_to_vector
from calibration      import Calibrator
from osc_sender       import OSCSender
from filters          import build_filter


def parse_args():
    p = argparse.ArgumentParser(description="Webcam gaze-to-OSC sender")
    p.add_argument("--filter",    default="hybrid",
                   choices=["kalman", "kde", "hybrid"])
    p.add_argument("--device",    default="cpu",
                   help="l2cs device: cpu / gpu:0 / cuda")
    p.add_argument("--calibrate", action="store_true",
                   help="Run 9-point calibration before streaming")
    p.add_argument("--screen-w",  type=int, default=1920)
    p.add_argument("--screen-h",  type=int, default=1080)
    p.add_argument("--osc-ip",    default="127.0.0.1")
    p.add_argument("--osc-port",  type=int, default=8000)
    return p.parse_args()


def _draw_calib_target(win_w, win_h, uv):
    """Return a blank BGR image with a calibration target circle at (u, v)."""
    img = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    cx = int(uv[0] * win_w)
    cy = int(uv[1] * win_h)
    cv2.circle(img, (cx, cy), 20, (0, 255, 0), -1)
    cv2.circle(img, (cx, cy),  4, (0,   0, 0), -1)
    cv2.putText(img, "Look at the dot — hold still",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    return img


def main():
    args = parse_args()
    cfg  = Config(
        filter_type   = args.filter,
        l2cs_device   = args.device,
        screen_width  = args.screen_w,
        screen_height = args.screen_h,
        osc_ip        = args.osc_ip,
        osc_port      = args.osc_port,
    )

    # ── Init ─────────────────────────────────────────────────
    capture   = FrameCapture(cfg)
    gaze_est  = GazeEstimator(cfg)
    face_pipe = FacePipeline(cfg)
    osc       = OSCSender(cfg)
    filt      = build_filter(cfg)
    projector = ScreenProjector(cfg)

    # Wait for first frame to get actual resolution
    frame = None
    while frame is None:
        frame = capture.read()
        time.sleep(0.005)
    h, w = frame.shape[:2]
    print(f"[INFO] Camera resolution: {w}x{h}")

    head_pose_est = HeadPoseEstimator(cfg, w, h)
    blink_det     = BlinkDetector(cfg)
    calibrator    = Calibrator(cfg) if args.calibrate else None

    if calibrator:
        print("[INFO] Calibration mode — follow the green dot on screen.")
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    send_interval = 1.0 / cfg.send_fps
    last_send     = time.time()

    print(f"[INFO] Streaming | filter={cfg.filter_type} | device={cfg.l2cs_device}"
          f" | OSC → {cfg.osc_ip}:{cfg.osc_port}")

    try:
        while True:
            frame = capture.read()
            if frame is None:
                continue

            # ── L2CS-Net gaze estimation ──────────────────────
            # Pass full BGR frame — Pipeline.step() handles face detection internally.
            # Output is in camera coords (Gaze360 weights); no head-pose correction needed.
            pitch_rad, yaw_rad, _bbox = gaze_est.predict(frame)

            # ── MediaPipe landmarks ───────────────────────────
            landmarks_px, _ = face_pipe.process(frame)

            # ── Blink detection ───────────────────────────────
            is_blink = blink_det.update(landmarks_px)

            # ── solvePnP → tvec (ray origin for screen projection) ──
            tvec = None
            if landmarks_px is not None:
                tvec_raw, _rot = head_pose_est.estimate(landmarks_px)
                tvec = tvec_raw.flatten()  # (3,1) → (3,)

            # ── Gaze vector → screen UV ───────────────────────
            if pitch_rad is not None:
                gaze_vec = angles_to_vector(pitch_rad, yaw_rad)
                u_raw, v_raw = projector.project(gaze_vec, eye_pos_mm=tvec)
            else:
                u_raw, v_raw = 0.5, 0.5

            # ── Calibration ───────────────────────────────────
            if calibrator is not None:
                if not calibrator.is_done():
                    if pitch_rad is not None:
                        calibrator.update(u_raw, v_raw)

                    target_uv = calibrator.current_target_uv
                    if target_uv is not None:
                        calib_img = _draw_calib_target(args.screen_w, args.screen_h,
                                                       target_uv)
                        cv2.imshow("Calibration", calib_img)
                        if cv2.waitKey(1) & 0xFF == 27:  # ESC to abort
                            break

                elif projector.homography is None:
                    projector.homography = calibrator.get_homography()
                    print("[INFO] Calibration complete — homography applied.")
                    cv2.destroyWindow("Calibration")

            # ── Filter ───────────────────────────────────────
            filtered = filt.update([u_raw, v_raw])
            u_f = float(np.clip(filtered[0], 0.0, 1.0))
            v_f = float(np.clip(filtered[1], 0.0, 1.0))

            # ── OSC send @ 30 fps ────────────────────────────
            now = time.time()
            if now - last_send >= send_interval:
                if is_blink:
                    osc.send_blink()
                else:
                    osc.send_gaze(u_f, v_f)
                last_send = now

    except KeyboardInterrupt:
        print("[INFO] Interrupted — shutting down.")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
