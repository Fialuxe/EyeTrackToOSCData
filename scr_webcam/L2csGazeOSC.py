#!/usr/bin/env python3
"""L2CS-Net + MediaPipe ハイブリッド視線推定 OSC 送信スクリプト

MediaPipeで安定した顔枠(Bounding Box)を抽出し、L2CS-Netで視線(Pitch/Yaw)を推論。
Ridge回帰を用いて頭の動きにロバストな正規化画面座標として OSC 送信する。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sklearn.linear_model import Ridge
import torch
from l2cs import Pipeline
from pythonosc.udp_client import SimpleUDPClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hybrid_gaze_osc")


# =========================================================================
# Domain Models
# =========================================================================
@dataclass(frozen=True)
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2)

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class GazeCoordinate:
    x: float
    y: float
    z: float = 0.0

    def clipped(self) -> "GazeCoordinate":
        return GazeCoordinate(
            x=float(np.clip(self.x, 0.0, 1.0)),
            y=float(np.clip(self.y, 0.0, 1.0)),
            z=float(np.clip(self.z, 0.0, 1.0)),
        )


@dataclass
class GazeEstimationResult:
    is_blink_or_lost: bool = False
    raw_pitch: float = 0.0
    raw_yaw: float = 0.0
    bbox: Optional[BoundingBox] = None


# =========================================================================
# Gaze Angle & Position → Screen Coordinate 変換
# =========================================================================
class GazeMapper(ABC):
    @abstractmethod
    def map(self, yaw: float, pitch: float, face_x: float, face_y: float) -> GazeCoordinate:
        pass


class LinearGazeMapper(GazeMapper):
    def __init__(self, yaw_range=(-0.5, 0.5), pitch_range=(-0.4, 0.4)):
        self._yaw_range = yaw_range
        self._pitch_range = pitch_range

    def map(self, yaw: float, pitch: float, face_x: float, face_y: float) -> GazeCoordinate:
        yaw_min, yaw_max = self._yaw_range
        pitch_min, pitch_max = self._pitch_range
        x = (-yaw - yaw_min) / (yaw_max - yaw_min)
        y = (-pitch - pitch_min) / (pitch_max - pitch_min)
        return GazeCoordinate(x=x, y=y, z=0.0).clipped()


class CalibratedGazeMapper(GazeMapper):
    def __init__(self, coeffs_x: np.ndarray, coeffs_y: np.ndarray) -> None:
        self._coeffs_x = coeffs_x
        self._coeffs_y = coeffs_y

    def _build_features(self, yaw: float, pitch: float, fx: float, fy: float) -> np.ndarray:
        return np.array([
            1.0, yaw, pitch, fx, fy,
            yaw ** 2, pitch ** 2, yaw * fx, pitch * fy, yaw * pitch
        ])

    def map(self, yaw: float, pitch: float, face_x: float, face_y: float) -> GazeCoordinate:
        features = self._build_features(yaw, pitch, face_x, face_y)
        x = float(features @ self._coeffs_x)
        y = float(features @ self._coeffs_y)
        return GazeCoordinate(x=x, y=y, z=0.0).clipped()

    def to_dict(self) -> dict:
        return {"coeffs_x": self._coeffs_x.tolist(), "coeffs_y": self._coeffs_y.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "CalibratedGazeMapper":
        return cls(coeffs_x=np.array(data["coeffs_x"]), coeffs_y=np.array(data["coeffs_y"]))

    @classmethod
    def fit(cls, yaw_samples, pitch_samples, face_x_samples, face_y_samples, screen_x, screen_y):
        n = len(yaw_samples)
        A = np.zeros((n, 10))
        dummy = cls(np.zeros(10), np.zeros(10))
        for i in range(n):
            A[i] = dummy._build_features(yaw_samples[i], pitch_samples[i], face_x_samples[i], face_y_samples[i])

        bx = np.array(screen_x)
        by = np.array(screen_y)

        model_x = Ridge(alpha=0.1, fit_intercept=False)
        model_y = Ridge(alpha=0.1, fit_intercept=False)
        model_x.fit(A, bx)
        model_y.fit(A, by)

        logger.info("キャリブレーション係数 X: %s", np.round(model_x.coef_, 3))
        logger.info("キャリブレーション係数 Y: %s", np.round(model_y.coef_, 3))

        return cls(coeffs_x=model_x.coef_, coeffs_y=model_y.coef_)


# =========================================================================
# 9 点キャリブレーション
# =========================================================================
class CalibrationRunner:
    GRID_COLS, GRID_ROWS = 3, 3
    MARGIN, SAMPLE_FRAMES = 0.1, 15
    MARKER_RADIUS = 20
    MARKER_COLOR = (0, 255, 255)
    MARKER_COLOR_ACTIVE = (0, 0, 255)
    BG_COLOR = (30, 30, 30)
    TEXT_COLOR = (255, 255, 255)
    WINDOW_NAME = "Calibration"

    def __init__(self, estimator: "L2CSPipelineEstimator", camera_id: int = 0):
        self._estimator = estimator
        self._camera_id = camera_id

    def run(self) -> Optional[CalibratedGazeMapper]:
        cap = cv2.VideoCapture(self._camera_id)
        if not cap.isOpened():
            logger.error("カメラを開けません。")
            return None

        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        points = self._generate_grid_points()

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        screen_w, screen_h = 1920, 1080
        try:
            cv2.imshow(self.WINDOW_NAME, np.zeros((100, 100, 3), dtype=np.uint8))
            cv2.waitKey(1)
            rect = cv2.getWindowImageRect(self.WINDOW_NAME)
            if rect[2] > 0 and rect[3] > 0:
                screen_w, screen_h = rect[2], rect[3]
        except: pass

        logger.info("キャリブレーション開始")

        yaw_samples, pitch_samples, face_x_samples, face_y_samples, screen_x, screen_y = [], [], [], [], [], []

        for idx, (nx, ny) in enumerate(points):
            px, py = int(nx * screen_w), int(ny * screen_h)
            collecting = False
            col_yaw, col_pitch, col_fx, col_fy = [], [], [], []

            while True:
                success, frame = cap.read()
                if not success:
                    time.sleep(0.01)
                    continue

                result = self._estimator.estimate(frame)
                canvas = np.full((screen_h, screen_w, 3), self.BG_COLOR, dtype=np.uint8)

                for j, (gx, gy) in enumerate(points):
                    color = (80, 80, 80) if j != idx else self.MARKER_COLOR
                    cv2.circle(canvas, (int(gx * screen_w), int(gy * screen_h)), self.MARKER_RADIUS // 2, color, -1)

                marker_color = self.MARKER_COLOR_ACTIVE if collecting else self.MARKER_COLOR
                cv2.circle(canvas, (px, py), self.MARKER_RADIUS, marker_color, -1)
                cv2.circle(canvas, (px, py), self.MARKER_RADIUS + 4, marker_color, 2)

                guide = f"Recording... ({len(col_yaw)}/{self.SAMPLE_FRAMES})" if collecting else f"Point {idx + 1}/{len(points)} - SPACE to record"
                cv2.putText(canvas, guide, (screen_w // 2 - 300, screen_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.TEXT_COLOR, 2)

                if result.is_blink_or_lost:
                    cv2.putText(canvas, "Face: NOT DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    st = f"Face: OK  yaw:{math.degrees(result.raw_yaw):+.1f} pitch:{math.degrees(result.raw_pitch):+.1f}"
                    cv2.putText(canvas, st, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow(self.WINDOW_NAME, canvas)
                key = cv2.waitKey(16) & 0xFF

                if key == 27:
                    cap.release()
                    cv2.destroyWindow(self.WINDOW_NAME)
                    return None

                if key == ord(" ") and not collecting and not result.is_blink_or_lost:
                    collecting = True
                    col_yaw.clear(); col_pitch.clear(); col_fx.clear(); col_fy.clear()

                if collecting and not result.is_blink_or_lost and result.bbox:
                    col_yaw.append(result.raw_yaw)
                    col_pitch.append(result.raw_pitch)
                    col_fx.append(result.bbox.center[0] / cam_w)
                    col_fy.append(result.bbox.center[1] / cam_h)

                    if len(col_yaw) >= self.SAMPLE_FRAMES:
                        yaw_samples.extend(col_yaw)
                        pitch_samples.extend(col_pitch)
                        face_x_samples.extend(col_fx)
                        face_y_samples.extend(col_fy)
                        screen_x.extend([nx] * self.SAMPLE_FRAMES)
                        screen_y.extend([ny] * self.SAMPLE_FRAMES)
                        break

        cap.release()
        cv2.destroyWindow(self.WINDOW_NAME)

        mapper = CalibratedGazeMapper.fit(yaw_samples, pitch_samples, face_x_samples, face_y_samples, screen_x, screen_y)
        return mapper

    def _generate_grid_points(self):
        return [(self.MARGIN + c * (1.0 - 2 * self.MARGIN) / (self.GRID_COLS - 1),
                 self.MARGIN + r * (1.0 - 2 * self.MARGIN) / (self.GRID_ROWS - 1))
                for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS)]


# =========================================================================
# Filters
# =========================================================================
class GazeFilter(ABC):
    @abstractmethod
    def apply(self, coord: GazeCoordinate) -> GazeCoordinate: pass
    @abstractmethod
    def reset(self) -> None: pass

class NoFilter(GazeFilter):
    def apply(self, coord: GazeCoordinate) -> GazeCoordinate: return coord
    def reset(self) -> None: pass

class KalmanFilter2D(GazeFilter):
    def __init__(self, process_noise=1e-4, measurement_noise=1e-2):
        self._Q = np.eye(4) * process_noise
        self._R = np.eye(2) * measurement_noise
        self._F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.float64)
        self._H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        self.reset()

    def apply(self, coord: GazeCoordinate) -> GazeCoordinate:
        z = np.array([coord.x, coord.y])
        if not self._initialized:
            self._state = np.array([coord.x, 0.0, coord.y, 0.0])
            self._initialized = True
            return coord
        pred_state = self._F @ self._state
        pred_cov = self._F @ self._covariance @ self._F.T + self._Q
        S = self._H @ pred_cov @ self._H.T + self._R
        K = pred_cov @ self._H.T @ np.linalg.inv(S)
        self._state = pred_state + K @ (z - self._H @ pred_state)
        self._covariance = (np.eye(4) - K @ self._H) @ pred_cov
        return GazeCoordinate(x=float(self._state[0]), y=float(self._state[2]))

    def reset(self) -> None:
        self._initialized = False
        self._state = np.zeros(4)
        self._covariance = np.eye(4)

class KalmanEMAFilter(GazeFilter):
    def __init__(self, process_noise=1e-4, measurement_noise=1e-2, ema_alpha=0.25):
        self._kalman = KalmanFilter2D(process_noise, measurement_noise)
        self._alpha = ema_alpha
        self.reset()

    def apply(self, coord: GazeCoordinate) -> GazeCoordinate:
        k_res = self._kalman.apply(coord)
        if self._px is None:
            self._px, self._py = k_res.x, k_res.y
            return k_res
        self._px = self._alpha * k_res.x + (1 - self._alpha) * self._px
        self._py = self._alpha * k_res.y + (1 - self._alpha) * self._py
        return GazeCoordinate(x=self._px, y=self._py)

    def reset(self):
        self._kalman.reset()
        self._px = self._py = None


# =========================================================================
# OSC Sender & Limiter
# =========================================================================
class OSCSender:
    def __init__(self, ip="127.0.0.1", port=8000):
        self._client = SimpleUDPClient(ip, port)

    def send_gaze(self, coord: GazeCoordinate):
        self._client.send_message("/gaze", [coord.x, coord.y, coord.z])

    def send_blink(self, value=1.0):
        self._client.send_message("/gaze/blink", value)

class FrameRateLimiter:
    def __init__(self, max_fps=30.0):
        self._interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._last_time = 0.0

    def wait(self):
        if self._interval <= 0: return
        now = time.perf_counter()
        remaining = self._interval - (now - self._last_time)
        if remaining > 0: time.sleep(remaining)
        self._last_time = time.perf_counter()


# =========================================================================
# Estimator — wraps the official L2CS-Net Pipeline (RetinaFace + ResNet50)
# =========================================================================
class L2CSPipelineEstimator:
    def __init__(self, weights_path: str | Path, device: str = "cpu") -> None:
        _device = torch.device(device)
        logger.info(f"Loading L2CS-Net Pipeline with RetinaFace onto {device}...")
        self._pipeline = Pipeline(
            weights=Path(weights_path),
            arch="ResNet50",
            device=_device,
            include_detector=True,
            confidence_threshold=0.5,
        )

    def estimate(self, frame_bgr: np.ndarray) -> GazeEstimationResult:
        try:
            results = self._pipeline.step(frame_bgr)
        except Exception:
            return GazeEstimationResult(is_blink_or_lost=True)

        if results.pitch.size == 0:
            return GazeEstimationResult(is_blink_or_lost=True)

        pitch = float(results.pitch[0])
        yaw = float(results.yaw[0])
        box = results.bboxes[0]
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = max(x1 + 1, int(box[2]))
        y2 = max(y1 + 1, int(box[3]))
        return GazeEstimationResult(
            raw_pitch=pitch,
            raw_yaw=yaw,
            bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
        )


# =========================================================================
# Preview Renderer
# =========================================================================
class PreviewRenderer:
    def __init__(self):
        self._prev_time = time.perf_counter()
        self._fps = 0.0

    def render(self, frame: np.ndarray, result: GazeEstimationResult) -> np.ndarray:
        now = time.perf_counter()
        self._fps = 1.0 / (now - self._prev_time) if now - self._prev_time > 0 else 0
        self._prev_time = now

        if result.is_blink_or_lost or result.bbox is None:
            cv2.putText(frame, "FACE NOT DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        bbox = result.bbox
        cv2.rectangle(frame, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max), (0, 255, 0), 2)
        cx, cy = bbox.center
        cv2.circle(frame, (cx, cy), 5, (255, 128, 0), -1)

        arrow_length = bbox.width * 1.5
        dx = -math.sin(result.raw_yaw) * math.cos(result.raw_pitch) * arrow_length
        dy = -math.sin(result.raw_pitch) * arrow_length
        cv2.arrowedLine(frame, (cx, cy), (int(cx + dx), int(cy + dy)), (0, 0, 255), 3, tipLength=0.25)

        cv2.putText(frame, f"yaw:{math.degrees(result.raw_yaw):+.1f} pitch:{math.degrees(result.raw_pitch):+.1f}",
                    (bbox.x_min, bbox.y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self._fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return frame


# =========================================================================
# Main Application
# =========================================================================
class GazeTrackingApp:
    def __init__(self, estimator, gaze_filter, osc_sender, gaze_mapper, camera_id=0, max_fps=30.0, show_preview=False):
        self._estimator = estimator
        self._filter = gaze_filter
        self._osc = osc_sender
        self._mapper = gaze_mapper
        self._camera_id = camera_id
        self._limiter = FrameRateLimiter(max_fps)
        self._show_preview = show_preview
        self._renderer = PreviewRenderer() if show_preview else None

    def run(self):
        cap = cv2.VideoCapture(self._camera_id)
        if not cap.isOpened(): sys.exit(1)

        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        try:
            while True:
                success, frame = cap.read()
                if not success: continue

                result = self._estimator.estimate(frame)

                if result.is_blink_or_lost or result.bbox is None:
                    self._osc.send_blink(1.0)
                else:
                    fx = result.bbox.center[0] / cam_w
                    fy = result.bbox.center[1] / cam_h
                    
                    raw_coord = self._mapper.map(result.raw_yaw, result.raw_pitch, fx, fy)
                    filtered = self._filter.apply(raw_coord).clipped()
                    self._osc.send_gaze(filtered)

                if self._renderer:
                    self._renderer.render(frame, result)
                    cv2.imshow("Hybrid Gaze Tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"): break

                self._limiter.wait()

        except KeyboardInterrupt: pass
        finally:
            cap.release()
            if self._show_preview: cv2.destroyAllWindows()


# =========================================================================
# CLI
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/L2CSNet_gaze360.pkl")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--osc-ip", type=str, default="127.0.0.1")
    parser.add_argument("--osc-port", type=int, default=8000)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--filter", type=str, default="kalman_ema")
    parser.add_argument("--ema-alpha", type=float, default=0.25)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calib-file", type=str, default="calibration.json")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    gaze_filter = KalmanEMAFilter(ema_alpha=args.ema_alpha) if args.filter == "kalman_ema" else NoFilter()
    estimator = L2CSPipelineEstimator(weights_path=args.weights, device=args.device)

    if args.calibrate:
        mapper = CalibrationRunner(estimator, args.camera).run()
        if mapper:
            with open(args.calib_file, "w") as f: json.dump(mapper.to_dict(), f)
    else:
        try:
            with open(args.calib_file, "r") as f: mapper = CalibratedGazeMapper.from_dict(json.load(f))
        except:
            mapper = LinearGazeMapper()

    app = GazeTrackingApp(estimator, gaze_filter, OSCSender(args.osc_ip, args.osc_port), mapper, args.camera, args.fps, args.preview)
    app.run()

if __name__ == "__main__":
    main()