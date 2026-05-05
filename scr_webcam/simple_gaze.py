#!/usr/bin/env python3
"""Simple calibrated gaze tracker — GPR edition.

Usage:
  python simple_gaze.py --calibrate        # 25-point calibration
  python simple_gaze.py                    # run with saved calibration
  python simple_gaze.py --osc-port 8000    # also send OSC /gaze [x, y, 0]

Press Q to quit.
"""

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from l2cs import Pipeline
from pythonosc.udp_client import SimpleUDPClient
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel

# ── pipeline ──────────────────────────────────────────────────────────────────

pipeline = Pipeline(
    weights=Path("models/L2CSNet_gaze360.pkl"),
    arch="ResNet50",
    device=torch.device("cpu"),
    include_detector=True,
    confidence_threshold=0.5,
)


def detect_gaze(frame: np.ndarray):
    """Return (yaw, pitch) in radians, or None if no face found."""
    try:
        r = pipeline.step(frame)
    except Exception:
        return None
    if r.pitch.size == 0:
        return None
    return float(r.yaw[0]), float(r.pitch[0])


# ── EMA filter ────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, alpha: float = 0.2):
        self._alpha = alpha
        self._v = None

    def update(self, x: float) -> float:
        self._v = x if self._v is None else self._alpha * x + (1 - self._alpha) * self._v
        return self._v

    def reset(self):
        self._v = None


# ── calibration ───────────────────────────────────────────────────────────────

_COLS, _ROWS = 5, 5
_GRID    = [(c / (_COLS - 1), r / (_ROWS - 1)) for r in range(_ROWS) for c in range(_COLS)]
_MARGIN  = 0.08
_SAMPLES = 30


def _grid_pixels(W: int, H: int):
    return [
        (int(_MARGIN * W + gx * (1 - 2 * _MARGIN) * W),
         int(_MARGIN * H + gy * (1 - 2 * _MARGIN) * H))
        for gx, gy in _GRID
    ]


def _make_kernel():
    return (
        C(1.0, (1e-3, 1e2))
        * RBF(length_scale=0.3, length_scale_bounds=(0.02, 2.0))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 0.5))
    )


def run_calibration(cap: cv2.VideoCapture, calib_file: str):
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    pts = _grid_pixels(W, H)

    X_train, y_sx, y_sy = [], [], []

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for i, (px, py) in enumerate(pts):
        buf_y, buf_p = [], []
        collecting = False

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            for j, (qx, qy) in enumerate(pts):
                cv2.circle(canvas, (qx, qy), 6, (50, 50, 50), -1)

            dot_color = (0, 80, 255) if collecting else (0, 200, 255)
            cv2.circle(canvas, (px, py), 16, dot_color, -1)
            cv2.circle(canvas, (px, py), 18, (255, 255, 255), 1)

            cv2.putText(canvas, f"Dot {i+1}/{len(pts)}  —  SPACE to record  ({len(buf_y)}/{_SAMPLES})",
                        (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

            gaze = detect_gaze(frame)
            if gaze and collecting:
                buf_y.append(gaze[0])
                buf_p.append(gaze[1])
                if len(buf_y) >= _SAMPLES:
                    X_train.append([float(np.mean(buf_y)), float(np.mean(buf_p))])
                    y_sx.append(px / W)
                    y_sy.append(py / H)
                    break

            cv2.imshow("Calibration", canvas)
            key = cv2.waitKey(16) & 0xFF
            if key == 27:
                cv2.destroyWindow("Calibration")
                return
            if key == ord(" ") and not collecting:
                collecting = True
                buf_y.clear(); buf_p.clear()

    cv2.destroyWindow("Calibration")

    print("Fitting GPR … ", end="", flush=True)
    X = np.array(X_train)
    gpr_x = GaussianProcessRegressor(kernel=_make_kernel(), n_restarts_optimizer=5,
                                     normalize_y=True).fit(X, y_sx)
    gpr_y = GaussianProcessRegressor(kernel=_make_kernel(), n_restarts_optimizer=5,
                                     normalize_y=True).fit(X, y_sy)
    print("done.")

    joblib.dump({"gpr_x": gpr_x, "gpr_y": gpr_y}, calib_file)
    print(f"Saved → {calib_file}")


def load_calibration(calib_file: str) -> dict:
    return joblib.load(calib_file)


def apply_calibration(calib: dict, yaw: float, pitch: float):
    X = np.array([[yaw, pitch]])
    sx = float(np.clip(calib["gpr_x"].predict(X)[0], 0.0, 1.0))
    sy = float(np.clip(calib["gpr_y"].predict(X)[0], 0.0, 1.0))
    return sx, sy


# ── main loop ─────────────────────────────────────────────────────────────────

def run(cap: cv2.VideoCapture, calib: dict, osc=None, alpha: float = 0.2):
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    ex, ey = EMA(alpha), EMA(alpha)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gaze = detect_gaze(frame)

        if gaze:
            sx, sy = apply_calibration(calib, gaze[0], gaze[1])
            sx, sy = ex.update(sx), ey.update(sy)

            gx, gy = int(sx * W), int(sy * H)
            cv2.circle(frame, (gx, gy), 20, (0, 80, 255), -1)
            cv2.circle(frame, (gx, gy), 22, (255, 255, 255), 2)
            cv2.putText(frame, f"({sx:.2f}, {sy:.2f})",
                        (gx + 26, gy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if osc:
                osc.send_message("/gaze", [sx, sy, 0.0])
        else:
            ex.reset(); ey.reset()
            cv2.putText(frame, "NO FACE", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.imshow("Gaze  [Q=quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibrate",  action="store_true")
    parser.add_argument("--calib-file", default="calibration_gpr.pkl")
    parser.add_argument("--camera",     type=int,   default=0)
    parser.add_argument("--osc-ip",     default="127.0.0.1")
    parser.add_argument("--osc-port",   type=int,   default=0,   help="0 = disabled")
    parser.add_argument("--ema-alpha",  type=float, default=0.2,
                        help="EMA smoothing — lower = smoother/laggier (default 0.2)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera.")

    print("Loading L2CS-Net + RetinaFace …")

    if args.calibrate:
        run_calibration(cap, args.calib_file)
        cap.release()
        return

    print(f"Loading calibration from {args.calib_file} …")
    calib = load_calibration(args.calib_file)
    osc   = SimpleUDPClient(args.osc_ip, args.osc_port) if args.osc_port else None
    run(cap, calib, osc, args.ema_alpha)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
