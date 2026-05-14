#!/usr/bin/env python3
"""
Calibrated gaze tracker with:

- L2CS-Net gaze estimation
- 25-point GPR calibration
- Lissajous head normalization
- Kalman + EMA smoothing
- OSC output
- OSC conflict auto-kill
- High priority mode
"""

import argparse
import math
import os
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import psutil
import torch

from l2cs import Pipeline
from pythonosc.udp_client import SimpleUDPClient

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    RBF,
    WhiteKernel,
)

from sklearn.linear_model import Ridge


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def kill_conflicting_python_processes(port: int):

    current_pid = os.getpid()

    print(f"[OSC] Checking conflicts on UDP port {port}")

    for proc in psutil.process_iter(
        ["pid", "name", "cmdline"]
    ):

        try:

            pid = proc.info["pid"]

            if pid == current_pid:
                continue

            name = proc.info["name"] or ""
            cmdline = " ".join(proc.info["cmdline"] or [])

            if (
                "python" not in name.lower()
                and "python" not in cmdline.lower()
            ):
                continue

            try:
                conns = proc.net_connections(kind="inet")

            except Exception:
                continue

            for conn in conns:

                if not conn.laddr:
                    continue

                if conn.laddr.port == port:

                    print(f"[KILL] PID={pid}")
                    print(f"       {cmdline}")

                    try:

                        proc.terminate()
                        proc.wait(timeout=3)

                    except psutil.TimeoutExpired:

                        print(f"[FORCE KILL] PID={pid}")
                        proc.kill()

                    break

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        ):
            continue


def set_high_priority():

    try:

        p = psutil.Process(os.getpid())

        if os.name == "nt":

            p.nice(psutil.HIGH_PRIORITY_CLASS)

        else:

            os.nice(-10)

        print("[PRIORITY] High priority enabled")

    except Exception as e:

        print(f"[PRIORITY] Failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# L2CS
# ─────────────────────────────────────────────────────────────────────────────

_pipeline = None


def _get_pipeline(
    weights="models/L2CSNet_gaze360.pkl",
    device="cpu",
):

    global _pipeline

    if _pipeline is None:

        print(
            f"Loading L2CS-Net + RetinaFace ({device}) ..."
        )

        _pipeline = Pipeline(
            weights=Path(weights),
            arch="ResNet50",
            device=torch.device(device),
            include_detector=True,
            confidence_threshold=0.5,
        )

    return _pipeline


def detect_gaze(frame):

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

    cx = (
        (float(box[0]) + float(box[2]))
        / 2.0
        / w
    )

    cy = (
        (float(box[1]) + float(box[3]))
        / 2.0
        / h
    )

    bbox_abs = (
        max(0, int(box[0])),
        max(0, int(box[1])),
        max(1, int(box[2])),
        max(1, int(box[3])),
    )

    return yaw, pitch, cx, cy, bbox_abs


# ─────────────────────────────────────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────────────────────────────────────

class KalmanEMA:

    def __init__(
        self,
        R=1e-2,
        q_ratio=0.01,
        alpha=0.25,
    ):

        Q = R * q_ratio

        self._F = np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        self._H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ], dtype=np.float64)

        self._Q = np.eye(4) * Q
        self._R = np.eye(2) * R

        self._alpha = alpha

        self.reset()

    def update(self, x, y):

        z = np.array([x, y])

        if not self._init:

            self._state = np.array([
                x,
                0.0,
                y,
                0.0,
            ])

            self._cov = np.eye(4)

            self._init = True

            self._ex = x
            self._ey = y

            return x, y

        sp = self._F @ self._state

        cp = (
            self._F
            @ self._cov
            @ self._F.T
            + self._Q
        )

        S = (
            self._H
            @ cp
            @ self._H.T
            + self._R
        )

        K = (
            cp
            @ self._H.T
            @ np.linalg.inv(S)
        )

        self._state = (
            sp
            + K @ (z - self._H @ sp)
        )

        self._cov = (
            (np.eye(4) - K @ self._H)
            @ cp
        )

        kx = float(self._state[0])
        ky = float(self._state[2])

        self._ex = (
            self._alpha * kx
            + (1 - self._alpha) * self._ex
        )

        self._ey = (
            self._alpha * ky
            + (1 - self._alpha) * self._ey
        )

        return self._ex, self._ey

    def reset(self):

        self._state = np.zeros(4)
        self._cov = np.eye(4)

        self._init = False

        self._ex = None
        self._ey = None


# ─────────────────────────────────────────────────────────────────────────────
# GPR
# ─────────────────────────────────────────────────────────────────────────────

def _make_kernel():

    return (
        C(1.0, (1e-3, 1e2))
        * RBF(
            length_scale=0.3,
            length_scale_bounds=(0.02, 2.0),
        )
        + WhiteKernel(
            noise_level=1e-2,
            noise_level_bounds=(1e-5, 0.5),
        )
    )


def _fit_gpr(X, y):

    return GaussianProcessRegressor(
        kernel=_make_kernel(),
        n_restarts_optimizer=5,
        normalize_y=True,
    ).fit(X, y)


def apply_gpr(calib, yaw, pitch):

    X = np.array([[yaw, pitch]])

    sx = float(
        np.clip(
            calib["gpr_x"].predict(X)[0],
            0.0,
            1.0,
        )
    )

    sy = float(
        np.clip(
            calib["gpr_y"].predict(X)[0],
            0.0,
            1.0,
        )
    )

    return sx, sy


# ─────────────────────────────────────────────────────────────────────────────
# HEAD NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _build_head_features(dfx, dfy):

    return np.column_stack([
        dfx,
        dfy,
        dfx**2,
        dfy**2,
        dfx * dfy,
    ])


def normalize_gaze(
    calib,
    yaw,
    pitch,
    fx,
    fy,
):

    if calib.get("norm_yaw") is None:
        return yaw, pitch

    ref_fx = calib["ref_fx"]
    ref_fy = calib["ref_fy"]

    dfx = fx - ref_fx
    dfy = fy - ref_fy

    h = _build_head_features(
        np.array([dfx]),
        np.array([dfy]),
    )

    corr_yaw = float(
        calib["norm_yaw"].predict(h)[0]
    )

    corr_pitch = float(
        calib["norm_pitch"].predict(h)[0]
    )

    return (
        yaw - corr_yaw,
        pitch - corr_pitch,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

_COLS, _ROWS = 3, 3

_GRID = [
    (
        c / (_COLS - 1),
        r / (_ROWS - 1),
    )
    for r in range(_ROWS)
    for c in range(_COLS)
]

_MARGIN = 0.02
_SAMPLES = 30


def _filter_outliers(
    values,
    max_zscore=2.0,
):

    arr = np.array(values)

    med = np.median(arr)

    mad = np.median(
        np.abs(arr - med)
    )

    if mad < 1e-9:
        return np.ones(len(arr), dtype=bool)

    z = np.abs(arr - med) / (mad * 1.4826)

    return z < max_zscore


def _detect_screen_size(window_name):

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    cv2.imshow(
        window_name,
        np.zeros((100, 100, 3), dtype=np.uint8),
    )

    cv2.waitKey(1)

    try:

        rect = cv2.getWindowImageRect(window_name)

        if rect[2] > 0 and rect[3] > 0:

            return rect[2], rect[3]

    except Exception:
        pass

    return 1920, 1080


def _fit_head_normalizer(
    calib,
    liss_data,
    ref_fx,
    ref_fy,
):

    raw_yaw = np.array(liss_data["yaw"])
    raw_pitch = np.array(liss_data["pitch"])

    fx = np.array(liss_data["fx"])
    fy = np.array(liss_data["fy"])

    sx = np.array(liss_data["sx"])
    sy = np.array(liss_data["sy"])

    calib_X = calib["train_X"]

    calib_sx = calib["train_sx"]
    calib_sy = calib["train_sy"]

    screen_features = np.column_stack([
        calib_sx,
        calib_sy,
        np.array(calib_sx)**2,
        np.array(calib_sy)**2,
        np.array(calib_sx) * np.array(calib_sy),
    ])

    rev_yaw = Ridge(alpha=0.1).fit(
        screen_features,
        calib_X[:, 0],
    )

    rev_pitch = Ridge(alpha=0.1).fit(
        screen_features,
        calib_X[:, 1],
    )

    liss_screen_feat = np.column_stack([
        sx,
        sy,
        sx**2,
        sy**2,
        sx * sy,
    ])

    expected_yaw = rev_yaw.predict(
        liss_screen_feat
    )

    expected_pitch = rev_pitch.predict(
        liss_screen_feat
    )

    delta_yaw = raw_yaw - expected_yaw
    delta_pitch = raw_pitch - expected_pitch

    dfx = fx - ref_fx
    dfy = fy - ref_fy

    H = _build_head_features(dfx, dfy)

    norm_yaw = Ridge(alpha=0.5).fit(
        H,
        delta_yaw,
    )

    norm_pitch = Ridge(alpha=0.5).fit(
        H,
        delta_pitch,
    )

    return norm_yaw, norm_pitch


# ─────────────────────────────────────────────────────────────────────────────
# PREVIEW
# ─────────────────────────────────────────────────────────────────────────────

def render_preview(
    frame,
    gaze,
    sx,
    sy,
    fps,
):

    h, w = frame.shape[:2]

    if gaze is None:

        cv2.putText(
            frame,
            "NO FACE",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            2,
        )

        return

    yaw, pitch, _, _, bbox = gaze

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    gx = int(sx * w)
    gy = int(sy * h)

    cv2.circle(
        frame,
        (gx, gy),
        20,
        (0, 80, 255),
        -1,
    )

    cv2.circle(
        frame,
        (gx, gy),
        22,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run(
    cap,
    calib,
    osc=None,
    preview=False,
    ema_alpha=0.25,
    max_fps=30.0,
):

    gpr_var = calib.get("gpr_variance", 1e-2)

    filt = KalmanEMA(
        R=gpr_var,
        q_ratio=0.01,
        alpha=ema_alpha,
    )

    prev_time = time.perf_counter()

    sx = 0.5
    sy = 0.5

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

            yaw, pitch, fcx, fcy, _ = gaze

            yaw_n, pitch_n = normalize_gaze(
                calib,
                yaw,
                pitch,
                fcx,
                fcy,
            )

            sx, sy = apply_gpr(
                calib,
                yaw_n,
                pitch_n,
            )

            sx, sy = filt.update(sx, sy)

            sx = float(np.clip(sx, 0.0, 1.0))
            sy = float(np.clip(sy, 0.0, 1.0))

            if osc:

                osc.send_message(
                    "/gaze",
                    [sx, sy, 0.0],
                )

        else:

            filt.reset()

            if osc:

                osc.send_message(
                    "/gaze/blink",
                    1.0,
                )

        if preview:

            render_preview(
                frame,
                gaze,
                sx,
                sy,
                fps,
            )

            cv2.imshow(
                "Gaze Tracker",
                frame,
            )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--calib-file",
        default="calibration_gpr.pkl",
    )

    ap.add_argument(
        "--weights",
        default="models/L2CSNet_gaze360.pkl",
    )

    ap.add_argument(
        "--device",
        default="cpu",
    )

    ap.add_argument(
        "--camera",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--osc-ip",
        default="127.0.0.1",
    )

    ap.add_argument(
        "--osc-port",
        type=int,
        default=8000,
    )

    ap.add_argument(
        "--preview",
        action="store_true",
    )

    ap.add_argument(
        "--ema-alpha",
        type=float,
        default=0.25,
    )

    ap.add_argument(
        "--max-fps",
        type=float,
        default=30.0,
    )

    args = ap.parse_args()

    if args.osc_port:

        kill_conflicting_python_processes(
            args.osc_port
        )

    set_high_priority()

    _get_pipeline(
        args.weights,
        args.device,
    )

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():

        raise SystemExit(
            "Cannot open camera."
        )

    print(
        f"Loading calibration: "
        f"{args.calib_file}"
    )

    calib = joblib.load(
        args.calib_file
    )

    osc = None

    if args.osc_port:

        osc = SimpleUDPClient(
            args.osc_ip,
            args.osc_port,
        )

    run(
        cap,
        calib,
        osc=osc,
        preview=args.preview,
        ema_alpha=args.ema_alpha,
        max_fps=args.max_fps,
    )

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
