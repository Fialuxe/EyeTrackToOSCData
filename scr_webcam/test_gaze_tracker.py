#!/usr/bin/env python3
"""Calibrated gaze tracker with head-correction.

Combines 25-point GPR calibration (high accuracy) with Lissajous-based
head-movement correction, Kalman+EMA smoothing, and OSC output.

Usage:
  python test_gaze_tracker.py --calibrate              # 25-pt GPR + Lissajous
  python test_gaze_tracker.py --calibrate --skip-lissajous
  python test_gaze_tracker.py --preview --device cuda   # GPU recommended
  python test_gaze_tracker.py --osc-port 8000

Keys during runtime:
  Q = quit
  R = re-calibrate (25-pt GPR + Lissajous, then resume)
  G = re-calibrate GPR only (skip Lissajous)

Press Q to quit.
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from l2cs import Pipeline
from pythonosc.udp_client import SimpleUDPClient
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.linear_model import Ridge


# ── L2CS pipeline (singleton) ────────────────────────────────────────────────

_pipeline = None


def _get_pipeline(weights="models/L2CSNet_gaze360.pkl", device="cpu"):
    global _pipeline
    if _pipeline is None:
        print(f"Loading L2CS-Net + RetinaFace ({device}) …")
        _pipeline = Pipeline(
            weights=Path(weights),
            arch="ResNet50",
            device=torch.device(device),
            include_detector=True,
            confidence_threshold=0.5,
        )
    return _pipeline


def detect_gaze(frame: np.ndarray):
    """Return (yaw, pitch, face_cx_norm, face_cy_norm, bbox_abs) or None."""
    pipe = _get_pipeline()
    try:
        r = pipe.step(frame)
    except Exception:
        return None
    if r.pitch.size == 0:
        return None

    yaw = float(r.yaw[0])
    pitch = float(r.pitch[0])

    h, w = frame.shape[:2]
    box = r.bboxes[0]
    cx = (float(box[0]) + float(box[2])) / 2.0 / w
    cy = (float(box[1]) + float(box[3])) / 2.0 / h

    bbox_abs = (
        max(0, int(box[0])), max(0, int(box[1])),
        max(1, int(box[2])), max(1, int(box[3])),
    )
    return yaw, pitch, cx, cy, bbox_abs


# ── Filters ───────────────────────────────────────────────────────────────────

class KalmanEMA:
    """2-D Kalman (position + velocity) → EMA.

    Theory:
      R = GPR prediction variance (measurement noise)
      Q = R * q_ratio (process noise; ratio encodes "gaze moves slowly
          relative to measurement noise"; 0.01 = 1% of R per frame)
    """

    def __init__(self, R: float = 1e-2, q_ratio: float = 0.01, alpha: float = 0.25):
        Q = R * q_ratio
        self._F = np.array([[1, 1, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]], dtype=np.float64)
        self._H = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0]], dtype=np.float64)
        self._Q = np.eye(4) * Q
        self._R = np.eye(2) * R
        self._alpha = alpha
        self.reset()

    def update(self, x: float, y: float):
        z = np.array([x, y])

        if not self._init:
            self._state = np.array([x, 0.0, y, 0.0])
            self._cov = np.eye(4)
            self._init = True
            self._ex, self._ey = x, y
            return x, y

        # predict
        sp = self._F @ self._state
        cp = self._F @ self._cov @ self._F.T + self._Q
        # update
        S = self._H @ cp @ self._H.T + self._R
        K = cp @ self._H.T @ np.linalg.inv(S)
        self._state = sp + K @ (z - self._H @ sp)
        self._cov = (np.eye(4) - K @ self._H) @ cp

        kx, ky = float(self._state[0]), float(self._state[2])
        self._ex = self._alpha * kx + (1 - self._alpha) * self._ex
        self._ey = self._alpha * ky + (1 - self._alpha) * self._ey
        return self._ex, self._ey

    def reset(self):
        self._state = np.zeros(4)
        self._cov = np.eye(4)
        self._init = False
        self._ex = self._ey = None


# ── GPR helpers ───────────────────────────────────────────────────────────────

def _make_kernel():
    return (
        C(1.0, (1e-3, 1e2))
        * RBF(length_scale=0.3, length_scale_bounds=(0.02, 2.0))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 0.5))
    )


def _fit_gpr(X, y):
    return GaussianProcessRegressor(
        kernel=_make_kernel(), n_restarts_optimizer=5, normalize_y=True
    ).fit(X, y)


def apply_gpr(calib, yaw, pitch):
    X = np.array([[yaw, pitch]])
    sx = float(np.clip(calib["gpr_x"].predict(X)[0], 0.0, 1.0))
    sy = float(np.clip(calib["gpr_y"].predict(X)[0], 0.0, 1.0))
    return sx, sy


# ── Head-correction (Ridge on residuals) ──────────────────────────────────────

def _build_head_features(dfx, dfy):
    return np.column_stack([dfx, dfy, dfx**2, dfy**2, dfx * dfy])


def _fit_head_correction(gpr_x, gpr_y, yaw, pitch, fx, fy, sx, sy, ref_fx, ref_fy):
    X_gaze = np.column_stack([yaw, pitch])
    pred_x = gpr_x.predict(X_gaze)
    pred_y = gpr_y.predict(X_gaze)

    residual_x = pred_x - sx
    residual_y = pred_y - sy

    dfx = fx - ref_fx
    dfy = fy - ref_fy
    H = _build_head_features(dfx, dfy)

    ridge_x = Ridge(alpha=1.0).fit(H, residual_x)
    ridge_y = Ridge(alpha=1.0).fit(H, residual_y)
    return ridge_x, ridge_y


def apply_head_correction(ridge_x, ridge_y, sx, sy, fx, fy, ref_fx, ref_fy,
                          max_correction=0.10):
    dfx = fx - ref_fx
    dfy = fy - ref_fy
    h = _build_head_features(np.array([dfx]), np.array([dfy]))
    cx = float(np.clip(ridge_x.predict(h)[0], -max_correction, max_correction))
    cy = float(np.clip(ridge_y.predict(h)[0], -max_correction, max_correction))
    return sx - cx, sy - cy


# ── Screen resolution helper ─────────────────────────────────────────────────

def _detect_screen_size(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.waitKey(1)
    try:
        rect = cv2.getWindowImageRect(window_name)
        if rect[2] > 0 and rect[3] > 0:
            return rect[2], rect[3]
    except Exception:
        pass
    return 1920, 1080


# ── 25-point GPR calibration ─────────────────────────────────────────────────

_COLS, _ROWS = 5, 5
_GRID = [(c / (_COLS - 1), r / (_ROWS - 1)) for r in range(_ROWS) for c in range(_COLS)]
_MARGIN = 0.08
_SAMPLES = 30


def run_calibration(cap, calib_file, skip_lissajous=False):
    scr_w, scr_h = _detect_screen_size("Calibration")

    pts_scr = [
        (int(_MARGIN * scr_w + gx * (1 - 2 * _MARGIN) * scr_w),
         int(_MARGIN * scr_h + gy * (1 - 2 * _MARGIN) * scr_h))
        for gx, gy in _GRID
    ]

    X_train, y_sx, y_sy = [], [], []
    face_xs, face_ys = [], []

    for i, (px, py) in enumerate(pts_scr):
        buf_y, buf_p, buf_fx, buf_fy = [], [], [], []
        collecting = False

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            canvas = np.zeros((scr_h, scr_w, 3), dtype=np.uint8)
            for j, (qx, qy) in enumerate(pts_scr):
                cv2.circle(canvas, (qx, qy), 6, (50, 50, 50), -1)

            dot_color = (0, 80, 255) if collecting else (0, 200, 255)
            cv2.circle(canvas, (px, py), 16, dot_color, -1)
            cv2.circle(canvas, (px, py), 18, (255, 255, 255), 1)

            status = f"Dot {i+1}/{len(pts_scr)}  SPACE=record  ({len(buf_y)}/{_SAMPLES})"
            cv2.putText(canvas, status, (20, scr_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

            gaze = detect_gaze(frame)
            if gaze is not None:
                yaw, pitch, fcx, fcy, _ = gaze
                cv2.putText(canvas,
                            f"Face OK  yaw:{math.degrees(yaw):+.1f} pitch:{math.degrees(pitch):+.1f}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                if collecting:
                    buf_y.append(yaw)
                    buf_p.append(pitch)
                    buf_fx.append(fcx)
                    buf_fy.append(fcy)
                    if len(buf_y) >= _SAMPLES:
                        X_train.append([float(np.mean(buf_y)), float(np.mean(buf_p))])
                        y_sx.append(px / scr_w)
                        y_sy.append(py / scr_h)
                        face_xs.append(float(np.mean(buf_fx)))
                        face_ys.append(float(np.mean(buf_fy)))
                        break
            else:
                cv2.putText(canvas, "Face: NOT DETECTED", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            cv2.imshow("Calibration", canvas)
            key = cv2.waitKey(16) & 0xFF
            if key == 27:
                cv2.destroyWindow("Calibration")
                return None
            if key == ord(" ") and not collecting:
                collecting = True
                buf_y.clear(); buf_p.clear(); buf_fx.clear(); buf_fy.clear()

    # ── fit GPR ──
    print("Fitting GPR … ", end="", flush=True)
    X = np.array(X_train)
    gpr_x = _fit_gpr(X, y_sx)
    gpr_y = _fit_gpr(X, y_sy)
    print("done.")

    ref_fx = float(np.mean(face_xs))
    ref_fy = float(np.mean(face_ys))

    _, std_x = gpr_x.predict(X, return_std=True)
    _, std_y = gpr_y.predict(X, return_std=True)
    gpr_variance = float(np.median(np.concatenate([std_x, std_y])) ** 2)
    print(f"GPR median variance: {gpr_variance:.6f}")

    # ── Lissajous ──
    ridge_x, ridge_y = None, None
    if not skip_lissajous:
        print("Lissajous calibration — follow the dot, move your head naturally …")
        liss = _run_lissajous(cap, scr_w, scr_h)
        if liss is not None and len(liss["yaw"]) >= 50:
            ridge_x, ridge_y = _fit_head_correction(
                gpr_x, gpr_y,
                np.array(liss["yaw"]), np.array(liss["pitch"]),
                np.array(liss["fx"]), np.array(liss["fy"]),
                np.array(liss["sx"]), np.array(liss["sy"]),
                ref_fx, ref_fy,
            )
            print(f"Head correction fitted on {len(liss['yaw'])} samples.")
        else:
            print("Insufficient Lissajous data — head correction skipped.")

    cv2.destroyWindow("Calibration")

    calib = {
        "gpr_x": gpr_x, "gpr_y": gpr_y,
        "ref_fx": ref_fx, "ref_fy": ref_fy,
        "gpr_variance": gpr_variance,
        "ridge_x": ridge_x, "ridge_y": ridge_y,
    }
    joblib.dump(calib, calib_file)
    print(f"Saved → {calib_file}")
    return calib


# ── Lissajous collection ─────────────────────────────────────────────────────

def _run_lissajous(cap, scr_w, scr_h, duration=20.0, settle=3.0):
    """Lissajous dot for head-correction data collection.

    Frequencies a=1.5, b=1.0 (slow enough for comfortable tracking).
    ~3 full loops in 20 s at these frequencies.
    First `settle` seconds discarded (user hasn't locked on).
    """
    freq_a, freq_b = 1.5, 1.0
    phase = math.pi / 2.0
    amplitude = 0.4

    data = {"yaw": [], "pitch": [], "fx": [], "fy": [], "sx": [], "sy": []}
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        t = time.perf_counter() - t0
        if t > duration:
            break

        lx = 0.5 + amplitude * math.sin(freq_a * t + phase)
        ly = 0.5 + amplitude * math.sin(freq_b * t)

        canvas = np.zeros((scr_h, scr_w, 3), dtype=np.uint8)
        dot_px, dot_py = int(lx * scr_w), int(ly * scr_h)
        cv2.circle(canvas, (dot_px, dot_py), 18, (0, 200, 255), -1)
        cv2.circle(canvas, (dot_px, dot_py), 20, (255, 255, 255), 1)

        remaining = max(0, duration - t)
        if t < settle:
            msg = f"Follow the dot — settling ({settle - t:.0f}s)"
            cv2.putText(canvas, msg, (20, scr_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        else:
            n = len(data["yaw"])
            msg = f"Recording ({remaining:.0f}s)  samples: {n}"
            cv2.putText(canvas, msg, (20, scr_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

        gaze = detect_gaze(frame)
        if gaze is not None and t >= settle:
            yaw, pitch, fcx, fcy, _ = gaze
            data["yaw"].append(yaw)
            data["pitch"].append(pitch)
            data["fx"].append(fcx)
            data["fy"].append(fcy)
            data["sx"].append(lx)
            data["sy"].append(ly)

        cv2.imshow("Calibration", canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            return None

    return data


# ── Preview renderer ─────────────────────────────────────────────────────────

def render_preview(frame, gaze, sx, sy, fps, flip_yaw, flip_pitch):
    h, w = frame.shape[:2]

    if gaze is None:
        cv2.putText(frame, "NO FACE", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        return

    yaw_raw, pitch_raw, _, _, bbox = gaze
    x1, y1, x2, y2 = bbox

    # BBox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Gaze arrow — flip signs let you fix direction at runtime
    yaw_a = -yaw_raw if flip_yaw else yaw_raw
    pitch_a = -pitch_raw if flip_pitch else pitch_raw

    bcx, bcy = (x1 + x2) // 2, (y1 + y2) // 2
    arrow_len = (x2 - x1) * 1.5
    dx = -math.sin(yaw_a) * math.cos(pitch_a) * arrow_len
    dy = -math.sin(pitch_a) * arrow_len
    cv2.arrowedLine(frame, (bcx, bcy),
                    (int(bcx + dx), int(bcy + dy)), (0, 0, 255), 3, tipLength=0.25)

    # Angle text
    cv2.putText(frame,
                f"yaw:{math.degrees(yaw_raw):+.1f} pitch:{math.degrees(pitch_raw):+.1f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Calibrated gaze dot
    gx, gy = int(sx * w), int(sy * h)
    cv2.circle(frame, (gx, gy), 20, (0, 80, 255), -1)
    cv2.circle(frame, (gx, gy), 22, (255, 255, 255), 2)
    cv2.putText(frame, f"({sx:.2f},{sy:.2f})",
                (gx + 26, gy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # FPS + keybinds
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "Q=quit  R=recalib  G=GPR-only", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(cap, calib, calib_file, osc=None, preview=False,
        ema_alpha=0.25, max_fps=30.0, flip_yaw=False, flip_pitch=False):

    has_head_corr = calib.get("ridge_x") is not None
    ref_fx = calib.get("ref_fx", 0.5)
    ref_fy = calib.get("ref_fy", 0.5)

    gpr_var = calib.get("gpr_variance", 1e-2)
    filt = KalmanEMA(R=gpr_var, q_ratio=0.01, alpha=ema_alpha)

    frame_interval = 1.0 / max_fps if max_fps > 0 else 0.0
    prev_time = time.perf_counter()
    fps = 0.0
    sx, sy = 0.5, 0.5

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        now = time.perf_counter()
        dt = now - prev_time
        fps = 1.0 / dt if dt > 0 else 0.0
        prev_time = now

        gaze = detect_gaze(frame)

        if gaze is not None:
            yaw, pitch, fcx, fcy, bbox = gaze
            sx, sy = apply_gpr(calib, yaw, pitch)

            if has_head_corr:
                sx, sy = apply_head_correction(
                    calib["ridge_x"], calib["ridge_y"],
                    sx, sy, fcx, fcy, ref_fx, ref_fy,
                )

            sx = float(np.clip(sx, 0.0, 1.0))
            sy = float(np.clip(sy, 0.0, 1.0))
            sx, sy = filt.update(sx, sy)
            sx = float(np.clip(sx, 0.0, 1.0))
            sy = float(np.clip(sy, 0.0, 1.0))

            if osc:
                osc.send_message("/gaze", [sx, sy, 0.0])
        else:
            filt.reset()
            if osc:
                osc.send_message("/gaze/blink", 1.0)

        if preview:
            render_preview(frame, gaze, sx, sy, fps, flip_yaw, flip_pitch)
            cv2.imshow("Gaze Tracker  [Q=quit R=recalib]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # ── Re-calibration hotkeys ──
        if key in (ord("r"), ord("g")):
            skip_liss = (key == ord("g"))
            print("\n── Re-calibrating ──")
            new_calib = run_calibration(cap, calib_file, skip_lissajous=skip_liss)
            if new_calib is not None:
                calib = new_calib
                has_head_corr = calib.get("ridge_x") is not None
                ref_fx = calib.get("ref_fx", 0.5)
                ref_fy = calib.get("ref_fy", 0.5)
                gpr_var = calib.get("gpr_variance", 1e-2)
                filt = KalmanEMA(R=gpr_var, q_ratio=0.01, alpha=ema_alpha)
                print("Re-calibration complete — resuming.\n")
            else:
                print("Re-calibration cancelled — using previous.\n")

        # FPS limiter
        elapsed = time.perf_counter() - now
        wait = frame_interval - elapsed
        if wait > 0:
            time.sleep(wait)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate",       action="store_true")
    ap.add_argument("--skip-lissajous",  action="store_true")
    ap.add_argument("--calib-file",      default="calibration_gpr.pkl")
    ap.add_argument("--weights",         default="models/L2CSNet_gaze360.pkl")
    ap.add_argument("--device",          default="cpu",
                    help="'cpu' or 'cuda' (GPU strongly recommended)")
    ap.add_argument("--camera",          type=int, default=0)
    ap.add_argument("--osc-ip",          default="127.0.0.1")
    ap.add_argument("--osc-port",        type=int, default=0, help="0 = disabled")
    ap.add_argument("--ema-alpha",       type=float, default=0.25)
    ap.add_argument("--max-fps",         type=float, default=30.0)
    ap.add_argument("--preview",         action="store_true",
                    help="Show camera with BBox, arrow, gaze dot")
    ap.add_argument("--flip-yaw",        action="store_true",
                    help="Flip arrow yaw direction (debug)")
    ap.add_argument("--flip-pitch",      action="store_true",
                    help="Flip arrow pitch direction (debug)")
    args = ap.parse_args()

    _get_pipeline(args.weights, args.device)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera.")

    if args.calibrate:
        run_calibration(cap, args.calib_file, skip_lissajous=args.skip_lissajous)
        cap.release()
        return

    print(f"Loading calibration from {args.calib_file} …")
    calib = joblib.load(args.calib_file)

    osc = SimpleUDPClient(args.osc_ip, args.osc_port) if args.osc_port else None

    run(cap, calib, args.calib_file,
        osc=osc, preview=args.preview,
        ema_alpha=args.ema_alpha, max_fps=args.max_fps,
        flip_yaw=args.flip_yaw, flip_pitch=args.flip_pitch)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()