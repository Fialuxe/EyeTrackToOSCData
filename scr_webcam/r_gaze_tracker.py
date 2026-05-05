#!/usr/bin/env python3
"""Calibrated gaze tracker with head-position input normalization.

Architecture:
  1. L2CS-Net → raw (yaw, pitch, face_x, face_y)
  2. Head normalization: subtract learned offset so that GPR always sees
     "as if you were sitting in calibration position"
     yaw'   = yaw   - f_yaw(Δfx, Δfy)
     pitch' = pitch - f_pitch(Δfx, Δfy)
  3. GPR (2-D): (yaw', pitch') → screen (sx, sy)
  4. Kalman+EMA smoothing
  5. OSC output + preview

Usage:
  python gaze_tracker.py --calibrate                # 25-pt GPR + Lissajous normalization
  python gaze_tracker.py --calibrate --skip-lissajous
  python gaze_tracker.py --preview --device cuda
  python gaze_tracker.py --osc-port 8000

Keys during runtime:
  Q = quit
  R = re-calibrate (GPR + Lissajous)
  G = re-calibrate GPR only
  L = re-do Lissajous only (keep existing GPR)
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

    R = GPR prediction variance (measurement noise).
    Q = R * q_ratio (process noise).
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

        sp = self._F @ self._state
        cp = self._F @ self._cov @ self._F.T + self._Q
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


# ── GPR ───────────────────────────────────────────────────────────────────────

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


# ── Head normalization (BEFORE GPR) ──────────────────────────────────────────
#
# The key insight: when the user moves their head, L2CS returns different
# (yaw, pitch) even if they're looking at the same screen point.  We learn
# the mapping  (Δfx, Δfy) → (Δyaw, Δpitch)  from Lissajous data, then
# subtract it so GPR always receives "calibration-equivalent" angles.
#
# Features: [Δfx, Δfy, Δfx², Δfy², Δfx·Δfy]
# Two Ridge models: one for Δyaw, one for Δpitch.

def _build_head_features(dfx, dfy):
    return np.column_stack([dfx, dfy, dfx**2, dfy**2, dfx * dfy])


def _fit_head_normalizer(calib, liss_data, ref_fx, ref_fy):
    """Learn how face displacement affects yaw/pitch.

    For each Lissajous sample we know:
      - the screen point the user was looking at (sx, sy)
      - the raw (yaw, pitch) L2CS returned
      - the face position (fx, fy)

    We use the GPR to find what (yaw, pitch) *should have been* for that
    screen point (i.e. the angles if the user's head were at calibration
    position).  The difference is the head-induced offset.

    Δyaw   = raw_yaw   - expected_yaw    (caused by head movement)
    Δpitch = raw_pitch  - expected_pitch

    We regress: (Δfx, Δfy) → (Δyaw, Δpitch)
    """
    gpr_x = calib["gpr_x"]
    gpr_y = calib["gpr_y"]

    raw_yaw   = np.array(liss_data["yaw"])
    raw_pitch = np.array(liss_data["pitch"])
    fx = np.array(liss_data["fx"])
    fy = np.array(liss_data["fy"])
    sx = np.array(liss_data["sx"])
    sy = np.array(liss_data["sy"])

    # For each sample, find the (yaw, pitch) that GPR would map to the
    # same screen point.  Since GPR is 2D→1D (separate for x and y),
    # we can't directly invert it.  But we can estimate the expected
    # angles from the calibration training data.
    #
    # Approach: use the calibration training points to build a reverse
    # mapping  (sx, sy) → (yaw, pitch).  This is the "what angles did
    # the user have when looking at this screen region during calibration?"
    #
    # We use Ridge regression for this reverse mapping (it's just an
    # approximation, but it's stable).

    calib_X = calib["train_X"]   # (N, 2) = (yaw, pitch) at calibration
    calib_sx = calib["train_sx"]  # screen x
    calib_sy = calib["train_sy"]  # screen y

    # Reverse map: screen → expected angles
    screen_features = np.column_stack([calib_sx, calib_sy,
                                       np.array(calib_sx)**2,
                                       np.array(calib_sy)**2,
                                       np.array(calib_sx) * np.array(calib_sy)])
    rev_yaw   = Ridge(alpha=0.1).fit(screen_features, calib_X[:, 0])
    rev_pitch = Ridge(alpha=0.1).fit(screen_features, calib_X[:, 1])

    # Expected angles for each Lissajous sample
    liss_screen_feat = np.column_stack([sx, sy, sx**2, sy**2, sx * sy])
    expected_yaw   = rev_yaw.predict(liss_screen_feat)
    expected_pitch = rev_pitch.predict(liss_screen_feat)

    # Head-induced offset
    delta_yaw   = raw_yaw   - expected_yaw
    delta_pitch = raw_pitch - expected_pitch

    # Regress on face displacement
    dfx = fx - ref_fx
    dfy = fy - ref_fy
    H = _build_head_features(dfx, dfy)

    norm_yaw   = Ridge(alpha=0.5).fit(H, delta_yaw)
    norm_pitch = Ridge(alpha=0.5).fit(H, delta_pitch)

    return norm_yaw, norm_pitch


def normalize_gaze(calib, yaw, pitch, fx, fy):
    """Subtract head-movement offset from raw gaze angles.

    Returns (yaw', pitch') as if the user were at calibration position.
    """
    if calib.get("norm_yaw") is None:
        return yaw, pitch

    ref_fx = calib["ref_fx"]
    ref_fy = calib["ref_fy"]
    dfx = fx - ref_fx
    dfy = fy - ref_fy

    h = _build_head_features(np.array([dfx]), np.array([dfy]))
    corr_yaw   = float(calib["norm_yaw"].predict(h)[0])
    corr_pitch = float(calib["norm_pitch"].predict(h)[0])

    return yaw - corr_yaw, pitch - corr_pitch


# ── Outlier filter ────────────────────────────────────────────────────────────

def _filter_outliers(values, max_zscore=2.0):
    arr = np.array(values)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad < 1e-9:
        return np.ones(len(arr), dtype=bool)
    z = np.abs(arr - med) / (mad * 1.4826)
    return z < max_zscore


# ── Screen resolution ────────────────────────────────────────────────────────

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
_SAMPLES = 50


def run_calibration(cap, calib_file, skip_lissajous=False):
    scr_w, scr_h = _detect_screen_size("Calibration")

    pts_scr = [
        (int(_MARGIN * scr_w + gx * (1 - 2 * _MARGIN) * scr_w),
         int(_MARGIN * scr_h + gy * (1 - 2 * _MARGIN) * scr_h))
        for gx, gy in _GRID
    ]

    canvas_base = np.zeros((scr_h, scr_w, 3), dtype=np.uint8)
    for qx, qy in pts_scr:
        cv2.circle(canvas_base, (qx, qy), 6, (50, 50, 50), -1)

    X_train, y_sx, y_sy = [], [], []
    face_xs, face_ys = [], []

    for i, (px, py) in enumerate(pts_scr):
        buf_y, buf_p, buf_fx, buf_fy = [], [], [], []
        collecting = False

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            canvas = canvas_base.copy()
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
                        # outlier filter
                        mask = _filter_outliers(buf_y) & _filter_outliers(buf_p)
                        if mask.sum() < 10:
                            mask = np.ones(len(buf_y), dtype=bool)
                        ay = np.array(buf_y)[mask]
                        ap = np.array(buf_p)[mask]
                        afx = np.array(buf_fx)[mask]
                        afy = np.array(buf_fy)[mask]

                        X_train.append([float(ay.mean()), float(ap.mean())])
                        y_sx.append(px / scr_w)
                        y_sy.append(py / scr_h)
                        face_xs.append(float(afx.mean()))
                        face_ys.append(float(afy.mean()))
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
    print("Fitting GPR (25-pt, 2-D) … ", end="", flush=True)
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

    calib = {
        "gpr_x": gpr_x, "gpr_y": gpr_y,
        "ref_fx": ref_fx, "ref_fy": ref_fy,
        "gpr_variance": gpr_variance,
        "train_X": X,
        "train_sx": y_sx,
        "train_sy": y_sy,
        "norm_yaw": None, "norm_pitch": None,
    }

    # ── Lissajous head normalization ──
    if not skip_lissajous:
        print("Lissajous — follow the dot AND move your head around …")
        liss = _run_lissajous(cap, scr_w, scr_h)
        if liss is not None and len(liss["yaw"]) >= 60:
            norm_yaw, norm_pitch = _fit_head_normalizer(
                calib, liss, ref_fx, ref_fy)
            calib["norm_yaw"]   = norm_yaw
            calib["norm_pitch"] = norm_pitch
            print(f"Head normalizer fitted on {len(liss['yaw'])} samples.")
        else:
            print("Insufficient Lissajous data — no head normalization.")

    cv2.destroyWindow("Calibration")

    joblib.dump(calib, calib_file)
    print(f"Saved → {calib_file}")
    return calib


def run_lissajous_only(cap, calib, calib_file):
    """Re-do just the Lissajous step, keeping existing GPR."""
    scr_w, scr_h = _detect_screen_size("Calibration")
    print("Lissajous — follow the dot AND move your head around …")
    liss = _run_lissajous(cap, scr_w, scr_h)
    if liss is not None and len(liss["yaw"]) >= 60:
        ref_fx = calib["ref_fx"]
        ref_fy = calib["ref_fy"]
        norm_yaw, norm_pitch = _fit_head_normalizer(calib, liss, ref_fx, ref_fy)
        calib["norm_yaw"]   = norm_yaw
        calib["norm_pitch"] = norm_pitch
        joblib.dump(calib, calib_file)
        print(f"Head normalizer updated ({len(liss['yaw'])} samples). Saved.")
    else:
        print("Insufficient data — normalizer unchanged.")
    cv2.destroyWindow("Calibration")
    return calib


# ── Lissajous collection ─────────────────────────────────────────────────────

def _run_lissajous(cap, scr_w, scr_h, duration=25.0, settle=3.0):
    """Lissajous dot. User must follow dot AND move head."""
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
            color = (100, 100, 100)
        else:
            n = len(data["yaw"])
            msg = f"Follow dot + MOVE HEAD slowly  ({remaining:.0f}s)  n={n}"
            color = (0, 255, 255)

        cv2.putText(canvas, msg, (20, scr_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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

def render_preview(frame, gaze, sx, sy, fps, flip_yaw, flip_pitch, has_norm):
    h, w = frame.shape[:2]

    if gaze is None:
        cv2.putText(frame, "NO FACE", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        return

    yaw_raw, pitch_raw, _, _, bbox = gaze
    x1, y1, x2, y2 = bbox

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    yaw_a = -yaw_raw if flip_yaw else yaw_raw
    pitch_a = -pitch_raw if flip_pitch else pitch_raw
    bcx, bcy = (x1 + x2) // 2, (y1 + y2) // 2
    arrow_len = (x2 - x1) * 1.5
    dx = -math.sin(yaw_a) * math.cos(pitch_a) * arrow_len
    dy = -math.sin(pitch_a) * arrow_len
    cv2.arrowedLine(frame, (bcx, bcy),
                    (int(bcx + dx), int(bcy + dy)), (0, 0, 255), 3, tipLength=0.25)

    cv2.putText(frame,
                f"yaw:{math.degrees(yaw_raw):+.1f} pitch:{math.degrees(pitch_raw):+.1f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    gx, gy = int(sx * w), int(sy * h)
    cv2.circle(frame, (gx, gy), 20, (0, 80, 255), -1)
    cv2.circle(frame, (gx, gy), 22, (255, 255, 255), 2)
    cv2.putText(frame, f"({sx:.2f},{sy:.2f})",
                (gx + 26, gy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    norm_label = "NORM" if has_norm else "raw"
    cv2.putText(frame, f"FPS: {fps:.1f}  [{norm_label}]", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "Q=quit R=recalib G=GPR-only L=lissajous-only",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(cap, calib, calib_file, osc=None, preview=False,
        ema_alpha=0.25, max_fps=30.0, flip_yaw=False, flip_pitch=False):

    has_norm = calib.get("norm_yaw") is not None
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

            # ── HEAD NORMALIZATION (before GPR) ──
            yaw_n, pitch_n = normalize_gaze(calib, yaw, pitch, fcx, fcy)

            # ── GPR ──
            sx, sy = apply_gpr(calib, yaw_n, pitch_n)

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
            render_preview(frame, gaze, sx, sy, fps, flip_yaw, flip_pitch, has_norm)
            cv2.imshow("Gaze Tracker  [Q R G L]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # ── Re-calibration hotkeys ──
        if key in (ord("r"), ord("g")):
            skip_liss = (key == ord("g"))
            print(f"\n── Re-calibrating ──")
            new_calib = run_calibration(cap, calib_file, skip_lissajous=skip_liss)
            if new_calib is not None:
                calib = new_calib
                has_norm = calib.get("norm_yaw") is not None
                gpr_var = calib.get("gpr_variance", 1e-2)
                filt = KalmanEMA(R=gpr_var, q_ratio=0.01, alpha=ema_alpha)
                print("Done — resuming.\n")
            else:
                print("Cancelled.\n")

        elif key == ord("l"):
            print("\n── Re-doing Lissajous only ──")
            calib = run_lissajous_only(cap, calib, calib_file)
            has_norm = calib.get("norm_yaw") is not None
            print("Done — resuming.\n")

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
    ap.add_argument("--preview",         action="store_true")
    ap.add_argument("--flip-yaw",        action="store_true")
    ap.add_argument("--flip-pitch",      action="store_true")
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
    has_norm = calib.get("norm_yaw") is not None
    print(f"Head normalization: {'ON' if has_norm else 'OFF'}")

    osc = SimpleUDPClient(args.osc_ip, args.osc_port) if args.osc_port else None

    run(cap, calib, args.calib_file,
        osc=osc, preview=args.preview,
        ema_alpha=args.ema_alpha, max_fps=args.max_fps,
        flip_yaw=args.flip_yaw, flip_pitch=args.flip_pitch)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()