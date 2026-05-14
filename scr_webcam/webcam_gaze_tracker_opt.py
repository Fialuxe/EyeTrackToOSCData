"""
gaze_tracker_v5.py  ── Multi-Pose Build (5 Phases)
=========================================
[Original v4 + Bug-Fix build features retained]

[NEW in this build: Extended Multi-Pose Calibration]
  45点キャリブレーション = 3×3グリッド × 5ポーズ（正面・左・右・上・下）

  PHASE 1 – FRONTAL  : yaw ∈ [-15°, +15°], pitch ∈ [-15°, +15°]
  PHASE 2 – LEFT     : yaw ∈ [-30°, -10°], pitch ∈ [-15°, +15°]
  PHASE 3 – RIGHT    : yaw ∈ [+10°, +30°], pitch ∈ [-15°, +15°]
  PHASE 4 – UP       : yaw ∈ [-15°, +15°], pitch ∈ [-20°,  0°]
  PHASE 5 – DOWN     : yaw ∈ [-15°, +15°], pitch ∈ [ +4°, +20°]

  NEW FUNCTIONS:
    draw_head_pose_hud()      — キャンバス上にヨー＆ピッチ矢印を常時表示
    draw_pose_bars()          — 現在のヨー/ピッチ vs ターゲットゾーンの2段バー (draw_yaw_barから拡張)
    wait_for_pose()           — 指定ポーズ（ヨー＆ピッチ）で安定するまで待機
    pulse_and_capture_gated() — ヨー＆ピッチゲート付きキャプチャ
    run_multi_pose_calibration() — 5フェーズオーケストレーター

  [FIXED BUGS]
    - CALIB_PHASES 全体に pitch_center / pitch_tol を追加
    - cv2.putText の引数エラーを修正
    - wait_for_pose, pulse_and_capture_gated のピッチ判定抜けを修正
"""
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont as _PILImageFont
import threading
import queue
import time
import argparse
import tkinter as tk
from collections import deque
from pythonosc.udp_client import SimpleUDPClient
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import torch
import torchvision
import torchvision.transforms as transforms
from l2cs import L2CS
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────
# Japanese/Unicode text rendering via Pillow
# ──────────────────────────────────────────────────────────────────────
_ja_font_cache: dict = {}

def _get_ja_font(size: int):
    if size not in _ja_font_cache:
        for path in [
            r"C:\Windows\Fonts\meiryo.ttc",
            r"C:\Windows\Fonts\YuGothM.ttc",
            r"C:\Windows\Fonts\msgothic.ttc",
            r"C:\Windows\Fonts\msmincho.ttc",
        ]:
            try:
                _ja_font_cache[size] = _PILImageFont.truetype(path, size)
                break
            except OSError:
                pass
        if size not in _ja_font_cache:
            _ja_font_cache[size] = _PILImageFont.load_default()
    return _ja_font_cache[size]

def put_text_ja(img: np.ndarray, text: str, org: tuple,
                font_size: int = 24, color_bgr=(255, 255, 255)) -> None:
    """Render text with Japanese/Unicode support via Pillow (in-place on img)."""
    pil = Image.fromarray(img[..., ::-1])   # BGR → RGB
    draw = ImageDraw.Draw(pil)
    font = _get_ja_font(font_size)
    fill = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))  # BGR → RGB
    draw.text(org, text, font=font, fill=fill)
    img[...] = np.array(pil)[..., ::-1]    # RGB → BGR


# ──────────────────────────────────────────────────────────────────────
# Landmark Indices
# ──────────────────────────────────────────────────────────────────────
LEFT_EYE_IDX = [
    107, 66, 105, 63, 70, 55, 65, 52, 53, 46,
    468, 469, 470, 471, 472,
    133, 33,
    173, 157, 158, 159, 160, 161, 246,
    155, 154, 153, 145, 144, 163, 7,
    243, 190, 56, 28, 27, 29, 30, 247,
    130, 25, 110, 24, 23, 22, 26, 112,
    244, 189, 221, 222, 223, 224, 225, 113,
    226, 31, 228, 229, 230, 231, 232, 233,
    193, 245, 128, 121, 120, 119, 118, 117,
    111, 35, 124, 143, 156,
]
RIGHT_EYE_IDX = [
    336, 296, 334, 293, 300, 285, 295, 282, 283, 276,
    473, 476, 475, 474, 477,
    362, 263,
    398, 384, 385, 386, 387, 388, 466,
    382, 381, 380, 374, 373, 390, 249,
    463, 414, 286, 258, 257, 259, 260, 467,
    359, 255, 339, 254, 253, 252, 256, 341,
    464, 413, 441, 442, 443, 444, 445, 342,
    446, 261, 448, 449, 450, 451, 452, 453,
    417, 465, 357, 350, 349, 348, 347, 346,
    340, 265, 353, 372, 383,
]
MUTUAL_IDX = [4, 10, 151, 9, 152, 234, 454, 58, 288]
ALL_LANDMARK_IDX = LEFT_EYE_IDX + RIGHT_EYE_IDX + MUTUAL_IDX

FACE_OVAL_IDS = [10,338,297,332,284,251,389,356,454,323,361,288,
                 397,365,379,378,400,377,152,148,176,149,150,136,
                 172,58,132,93,234,127,162,21,54,103,67,109]

# Define contours of face mesh
# Uses frozenset to ensure fast lookups
FACEMESH_CONTOURS = frozenset([
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),
    (356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),
    (379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),
    (149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),
    (234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10),
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),
    (155,133),(133,173),(173,157),(157,158),(158,159),(159,160),(160,161),
    (161,246),(246,33),
    (362,382),(382,381),(381,380),(380,374),(374,373),(373,390),(390,249),
    (249,263),(263,466),(466,388),(388,387),(387,386),(386,385),(385,384),
    (384,398),(398,362),
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),
    (321,375),(375,291),(291,409),(409,270),(270,269),(269,267),(267,0),
    (0,37),(37,39),(39,40),(40,185),(185,61),
    (46,53),(53,52),(52,65),(65,55),(55,70),(70,63),(63,105),(105,66),(66,107),(107,46),
    (276,283),(283,282),(282,295),(295,285),(285,300),(300,293),(293,334),
    (334,296),(296,336),(336,276),
])
FACEMESH_IRISES = frozenset([
    (468,469),(469,470),(470,471),(471,472),(472,468),
    (473,474),(474,475),(475,476),(476,477),(477,473),
])

# ──────────────────────────────────────────────────────────────────────
# Multi-Pose Phase Definitions
# ──────────────────────────────────────────────────────────────────────
CALIB_PHASES = [
    dict(
        name="FRONTAL",
        label_ja="正面",
        instruction_ja="正面をまっすぐ見てください",
        instruction_en="Face STRAIGHT to camera",
        yaw_center=0, yaw_tol=15,          # [-15, +15]
        pitch_center=0, pitch_tol=15,      # [-15, +15]
        color=(0, 220, 80),
    ),
    dict(
        name="LEFT",
        label_ja="左向き",
        instruction_ja="顔を左に向けてください（約20°）",
        instruction_en="Turn face LEFT ~20°",
        yaw_center=-20, yaw_tol=10,        # [-30, -10]
        pitch_center=0, pitch_tol=15,      # [-15, +15]
        color=(80, 180, 255),
    ),
    dict(
        name="RIGHT",
        label_ja="右向き",
        instruction_ja="顔を右に向けてください（約20°）",
        instruction_en="Turn face RIGHT ~20°",
        yaw_center=+20, yaw_tol=10,        # [+10, +30]
        pitch_center=0, pitch_tol=15,      # [-15, +15]
        color=(255, 160, 60),
    ),
    dict(
        name="UP",
        label_ja="上向き",
        instruction_ja="顔を少し上に向けてください",
        instruction_en="Tilt face slightly UP",
        yaw_center=0, yaw_tol=15,          # [-15, +15]
        pitch_center=-10, pitch_tol=10,    # [-20, 0]
        color=(255, 180, 180),
    ),
    dict(
        name="DOWN",
        label_ja="下向き",
        instruction_ja="顔を少し下に向けてください",
        instruction_en="Tilt face DOWN",
        yaw_center=0, yaw_tol=15,          # [-15, +15]
        pitch_center=+12, pitch_tol=8,     # [+4, +20]
        color=(120, 180, 255),
    ),
]

def phase_yaw_range(phase):
    c, t = phase["yaw_center"], phase["yaw_tol"]
    return c - t, c + t

def phase_pitch_range(phase):
    c, t = phase["pitch_center"], phase["pitch_tol"]
    return c - t, c + t


# ──────────────────────────────────────────────────────────────────────
# FaceMesh Drawing
# ──────────────────────────────────────────────────────────────────────
def draw_face_mesh(frame, landmarks, img_w, img_h,
                   draw_contours=True, draw_irises=True,
                   draw_all_points=True, draw_solvepnp=True,
                   flipped=False, alpha=0.55):
    h, w = frame.shape[:2]
    def px(lm):
        x = (1.0 - lm.x) if flipped else lm.x
        return int(x * img_w), int(lm.y * img_h)
    pts = np.array([px(lm) for lm in landmarks], dtype=np.int32)
    overlay = frame.copy()
    if draw_all_points:
        for p in pts:
            if 0 <= p[0] < w and 0 <= p[1] < h:
                cv2.circle(overlay, tuple(p), 1, (80, 80, 80), -1)
    if draw_contours:
        for i, j in FACEMESH_CONTOURS:
            if i < len(pts) and j < len(pts):
                cv2.line(overlay, tuple(pts[i]), tuple(pts[j]), (0, 200, 60), 1, cv2.LINE_AA)
    if draw_irises:
        for i, j in FACEMESH_IRISES:
            if i < len(pts) and j < len(pts):
                cv2.line(overlay, tuple(pts[i]), tuple(pts[j]), (0, 220, 220), 1, cv2.LINE_AA)
        for idx, col in [(468, (0,255,255)), (473, (0,255,255))]:
            if idx < len(pts):
                cv2.drawMarker(overlay, tuple(pts[idx]), col, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    if draw_solvepnp:
        labels = {1:"nose", 152:"chin", 33:"L_eye", 263:"R_eye", 61:"L_mth", 291:"R_mth"}
        for idx, name in labels.items():
            if idx < len(pts):
                p = tuple(pts[idx])
                cv2.circle(frame, p, 5, (0, 220, 255), -1)
                cv2.putText(frame, name, (p[0]+6, p[1]-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,220,255), 1, cv2.LINE_AA)
    for idx in ALL_LANDMARK_IDX:
        if idx < len(pts):
            p = tuple(pts[idx])
            if 0 <= p[0] < w and 0 <= p[1] < h:
                cv2.circle(frame, p, 2, (180, 100, 255), -1)


# ──────────────────────────────────────────────────────────────────────
# Head Pose HUD & Pose Bars
# ──────────────────────────────────────────────────────────────────────
def draw_head_pose_hud(canvas, head_yaw_deg, head_pitch_deg,
                       cx=None, cy=None, radius=55,
                       phase=None, in_gate=False):
    """ヘッドポーズ矢印サークルをキャンバスに描画する。"""
    h, w = canvas.shape[:2]
    if cx is None: cx = w - radius - 20
    if cy is None: cy = radius + 20

    # 背景円
    cv2.circle(canvas, (cx, cy), radius, (30, 30, 30), -1)
    rim_color = (0, 220, 80) if in_gate else (100, 100, 100)
    cv2.circle(canvas, (cx, cy), radius, rim_color, 2)

    # 十字線
    cv2.line(canvas, (cx - radius, cy), (cx + radius, cy), (50,50,50), 1)
    cv2.line(canvas, (cx, cy - radius), (cx, cy + radius), (50,50,50), 1)

    # 矢印（ヨー=水平、ピッチ=垂直）
    yaw_r   = np.radians(head_yaw_deg)
    pitch_r = np.radians(head_pitch_deg)
    arrow_len = int(radius * 0.78)
    dx =  int(np.sin(yaw_r)    * arrow_len)
    dy =  int(np.sin(pitch_r)  * arrow_len)   # 下向き正
    ex, ey = cx + dx, cy + dy
    arr_col = phase["color"] if phase else (0, 220, 80)
    cv2.arrowedLine(canvas, (cx, cy), (ex, ey), arr_col, 2, tipLength=0.35)
    cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), -1)

    # 数値
    txt = f"Y{head_yaw_deg:+.0f} P{head_pitch_deg:+.0f}"
    cv2.putText(canvas, txt,
                (cx - radius + 5, cy + radius + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, arr_col, 1, cv2.LINE_AA)

def draw_pose_bars(canvas, head_yaw_deg, head_pitch_deg, phase,
                   bar_x=None, bar_y=None, bar_w=None, bar_h=10,
                   yaw_range=(-50, 50), pitch_range=(-50, 50)):
    """
    ヨーとピッチ両方のバーを描画する。
    緑のゾーン = ターゲット範囲、カーソル = 現在の角度。
    """
    h, w = canvas.shape[:2]
    if bar_w is None: bar_w = w - 80
    if bar_x is None: bar_x = 40
    if bar_y is None: bar_y = h - (bar_h * 2) - 45  # 下部へ配置

    y_lo, y_hi = phase_yaw_range(phase)
    p_lo, p_hi = phase_pitch_range(phase)

    def draw_single_bar(val, v_min, v_max, t_lo, t_hi, y_offset, label_prefix):
        # バー背景
        cv2.rectangle(canvas, (bar_x, y_offset), (bar_x + bar_w, y_offset + bar_h),
                      (30, 30, 30), -1)
        cv2.rectangle(canvas, (bar_x, y_offset), (bar_x + bar_w, y_offset + bar_h),
                      (70, 70, 70), 1)

        def to_px(deg):
            t = (deg - v_min) / (v_max - v_min)
            return bar_x + int(np.clip(t, 0, 1) * bar_w)

        # ターゲットゾーン（フィルド）
        lo_px = to_px(t_lo)
        hi_px = to_px(t_hi)
        cv2.rectangle(canvas, (lo_px, y_offset), (hi_px, y_offset + bar_h),
                      (0, 60, 0), -1)
        cv2.rectangle(canvas, (lo_px, y_offset), (hi_px, y_offset + bar_h),
                      phase["color"], 1)

        # カーソル
        cur_px = to_px(val)
        in_zone = t_lo <= val <= t_hi
        cur_col = phase["color"] if in_zone else (80, 80, 255)
        cv2.line(canvas, (cur_px, y_offset - 4), (cur_px, y_offset + bar_h + 4),
                 cur_col, 3)

        # 0° マーク
        zero_px = to_px(0)
        cv2.line(canvas, (zero_px, y_offset), (zero_px, y_offset + bar_h),
                 (120, 120, 120), 1)

        # ラベル
        label_col = phase["color"] if in_zone else (150, 150, 150)
        txt = f"{label_prefix} {val:+.1f} deg   target [{t_lo:.0f} ~ {t_hi:.0f}]"
        cv2.putText(canvas, txt, (bar_x, y_offset - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_col, 1, cv2.LINE_AA)

    # 上段にYaw、下段にPitchを描画
    draw_single_bar(head_yaw_deg, yaw_range[0], yaw_range[1], y_lo, y_hi, bar_y, "Yaw")
    draw_single_bar(head_pitch_deg, pitch_range[0], pitch_range[1], p_lo, p_hi, bar_y + bar_h + 24, "Pitch")


# ──────────────────────────────────────────────────────────────────────
# AsyncCamera
# ──────────────────────────────────────────────────────────────────────
class AsyncCamera:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.q = queue.Queue(maxsize=2)
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            if not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(frame)

    def read(self):
        return self.q.get() if not self.q.empty() else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


# ──────────────────────────────────────────────────────────────────────
# Kalman + EMA Filter
# ──────────────────────────────────────────────────────────────────────
class KalmanEMAFilter:
    def __init__(self, ema_alpha=0.35, Q=1e-2, R=0.5):
        self.ema_alpha = ema_alpha
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix   = np.array(
                                                [[1,0,0,0],
                                                [0,1,0,0]], 
                                                np.float32)
        self.kf.transitionMatrix    = np.array(
                                                [[1,0,1,0],
                                                [0,1,0,1],
                                                [0,0,1,0],
                                                [0,0,0,1]], 
                                                np.float32)
        self.kf.processNoiseCov     = np.diag([Q, Q, Q*10, Q*10]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(R)
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self.ema  = None
        self.init = False

    def reset(self, x, y):
        self.kf.statePre  = np.array([[x],[y],[0],[0]], np.float32)
        self.kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
        self.ema  = np.array([x, y], np.float32)
        self.init = True

    def update(self, x, y):
        if not self.init:
            self.reset(x, y)
            return x, y
        m = np.array([[x],[y]], np.float32)
        self.kf.predict()
        e = self.kf.correct(m)
        kx, ky = float(e[0,0]), float(e[1,0])
        self.ema = self.ema_alpha * np.array([kx, ky], np.float32) \
                 + (1 - self.ema_alpha) * self.ema
        return float(self.ema[0]), float(self.ema[1])


# ──────────────────────────────────────────────────────────────────────
# BBoxStabilizer
# ──────────────────────────────────────────────────────────────────────
class BBoxStabilizer:
    def __init__(self, alpha=0.20):
        self.alpha = alpha
        self.bbox  = None

    def update(self, b):
        if self.bbox is None:
            self.bbox = np.array(b, np.float32)
        else:
            self.bbox = self.alpha * np.array(b, np.float32) + (1-self.alpha)*self.bbox
        return [int(v) for v in self.bbox]


# ──────────────────────────────────────────────────────────────────────
# GazeEstimator
# ──────────────────────────────────────────────────────────────────────
class GazeEstimator:
    FEAT_FACE_NORM = slice(0,   513)
    FEAT_HEAD_POS  = slice(513, 516)
    FEAT_L2CS      = slice(516, 518)
    FEAT_IRIS      = slice(518, 522)
    FEAT_DIM       = 522

    def __init__(self, img_w, img_h, model_path="models/L2CSNet_gaze360.pkl"):
        self.img_w = img_w
        self.img_h = img_h
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.bbox_stab = BBoxStabilizer(alpha=0.20)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16   = torch.cuda.is_available()
        self.l2cs   = L2CS(block=torchvision.models.resnet.Bottleneck,
                           layers=[3,4,6,3], num_bins=90)
        self.l2cs.load_state_dict(torch.load(model_path, map_location=self.device))
        self.l2cs.to(self.device).eval()
        if self.fp16: self.l2cs.half()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224), antialias=True),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.idx_tensor = None
        self._ear_history = deque(maxlen=60)
        self._blink_ratio = 0.75

    def _ear(self, lm):
        def _e(outer, inner, top, bot):
            w = np.linalg.norm(np.array([lm[outer].x, lm[outer].y])
                              -np.array([lm[inner].x, lm[inner].y])) + 1e-9
            h = np.linalg.norm(np.array([lm[top].x, lm[top].y])
                              -np.array([lm[bot].x,  lm[bot].y]))
            return h / w
        return (_e(33,133,159,145) + _e(263,362,386,374)) / 2.0

    def _face_normalized_features(self, lm_list, all_points_px):
        left_corner  = all_points_px[33]
        right_corner = all_points_px[263]
        top_of_head  = all_points_px[10]
        eye_center = (left_corner + right_corner) / 2.0

        x_axis = right_corner - left_corner
        x_axis /= np.linalg.norm(x_axis) + 1e-9

        y_approx = eye_center - top_of_head
        y_approx /= np.linalg.norm(y_approx) + 1e-9
        y_axis = y_approx - np.dot(y_approx, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis) + 1e-9

        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-9

        R_face = np.column_stack((x_axis, y_axis, z_axis))

        shifted = all_points_px - eye_center
        rotated = (R_face.T @ shifted.T).T

        inter_eye = np.linalg.norm(right_corner - left_corner)
        if inter_eye > 1e-7:
            rotated /= inter_eye

        subset = rotated[ALL_LANDMARK_IDX]
        flat   = subset.flatten()

        yaw   = np.arctan2( R_face[0, 2], R_face[2, 2])
        pitch = np.arctan2(-R_face[1, 2], np.sqrt(R_face[0,2]**2 + R_face[2,2]**2))
        roll  = np.arctan2( R_face[1, 0], R_face[0, 0])

        features = np.concatenate([flat, [yaw, pitch, roll]])  # (513,)
        return features.astype(np.float64), R_face

    def _tight_crop(self, lm, rgb):
        pts = np.array([[lm[i].x*self.img_w, lm[i].y*self.img_h]
                        for i in FACE_OVAL_IDS], np.float32)
        x0,y0 = pts[:,0].min(), pts[:,1].min()
        x1,y1 = pts[:,0].max(), pts[:,1].max()
        pw = (x1-x0)*0.10; ph = (y1-y0)*0.10
        x0,y0 = max(0,x0-pw), max(0,y0-ph)
        x1,y1 = min(self.img_w,x1+pw), min(self.img_h,y1+ph)
        sx0,sy0,sx1,sy1 = self.bbox_stab.update([x0,y0,x1,y1])
        sx0,sy0 = max(0,sx0), max(0,sy0)
        sx1,sy1 = min(self.img_w,sx1), min(self.img_h,sy1)
        return rgb[sy0:sy1, sx0:sx1]

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None, False, None

        lm = res.multi_face_landmarks[0].landmark
        all_points = np.array([[l.x, l.y, l.z] for l in lm], np.float32)

        ear = self._ear(lm)
        self._ear_history.append(ear)
        thr = (float(np.mean(self._ear_history)) * self._blink_ratio
               if len(self._ear_history) >= 15 else 0.18)
        is_blinking = ear < thr

        # Head pose
        p_L = all_points[234]
        p_R = all_points[454]
        p_T = all_points[10]
        p_B = all_points[152]

        x_axis = p_R - p_L
        x_axis /= np.linalg.norm(x_axis) + 1e-9
        y_approx = p_B - p_T
        y_approx /= np.linalg.norm(y_approx) + 1e-9
        y_axis = y_approx - np.dot(y_approx, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis) + 1e-9
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-9
        R_head = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)

        head_yaw_deg   = float(np.degrees(np.arctan2( R_head[0,2], R_head[2,2])))
        head_pitch_deg = float(np.degrees(np.arctan2(-R_head[1,2],
                                          np.sqrt(R_head[0,2]**2 + R_head[2,2]**2))))
        head_roll_deg  = float(np.degrees(np.arctan2( R_head[1,0], R_head[0,0])))

        # L2CS
        crop = self._tight_crop(lm, rgb)
        l2cs_yaw_local = l2cs_pitch_local = 0.0
        if crop.size > 0:
            if self.idx_tensor is None:
                self.idx_tensor = torch.arange(90, dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                t = self.transform(crop).unsqueeze(0).to(self.device)
                if self.fp16: t = t.half()
                gp, gy = self.l2cs(t)
                p_s = torch.softmax(gp, dim=1)
                y_s = torch.softmax(gy, dim=1)
                l2cs_pitch_local = (torch.sum(p_s * self.idx_tensor) * 4 - 180).item()
                l2cs_yaw_local   = (torch.sum(y_s * self.idx_tensor) * 4 - 180).item()

        # World-compensated gaze
        yaw_r   = np.radians(l2cs_yaw_local)
        pitch_r = np.radians(l2cs_pitch_local)
        gaze_cam = np.array([
             np.cos(pitch_r) * np.sin(yaw_r),
             np.sin(pitch_r),
             np.cos(pitch_r) * np.cos(yaw_r),
        ], dtype=np.float64)
        gaze_face_local = R_head.T @ gaze_cam
        world_yaw_deg   = float(np.degrees(np.arctan2(gaze_face_local[0], gaze_face_local[2])))
        world_pitch_deg = float(np.degrees(np.arctan2(gaze_face_local[1],
                                           np.sqrt(gaze_face_local[0]**2 + gaze_face_local[2]**2))))

        # Iris offsets
        l_eye_center  = (all_points[33]  + all_points[133]) / 2.0
        r_eye_center  = (all_points[362] + all_points[263]) / 2.0
        l_iris_local  = R_head.T @ (all_points[468] - l_eye_center)
        r_iris_local  = R_head.T @ (all_points[473] - r_eye_center)
        l_eye_width   = np.linalg.norm(all_points[33]  - all_points[133]) + 1e-9
        r_eye_width   = np.linalg.norm(all_points[362] - all_points[263]) + 1e-9
        iris_l_dx = l_iris_local[0] / l_eye_width
        iris_l_dy = l_iris_local[1] / l_eye_width
        iris_r_dx = r_iris_local[0] / r_eye_width
        iris_r_dy = r_iris_local[1] / r_eye_width

        eye_center_global = (all_points[33] + all_points[263]) / 2.0
        head_x    = float(eye_center_global[0])
        head_y    = float(eye_center_global[1])
        head_size = float(np.linalg.norm(all_points[263] - all_points[33]))

        # Face-normalized features
        all_points_px = all_points.copy()
        all_points_px[:, 0] *= self.img_w
        all_points_px[:, 1] *= self.img_h
        face_norm_feats, R_face = self._face_normalized_features(lm, all_points_px)

        features = np.concatenate([
            face_norm_feats,
            [head_x, head_y, head_size],
            [l2cs_yaw_local, l2cs_pitch_local],
            [iris_l_dx, iris_l_dy, iris_r_dx, iris_r_dy],
        ]).astype(np.float64)

        dbg = GazeDebugInfo(
            head_yaw=head_yaw_deg, head_pitch=head_pitch_deg, head_roll=head_roll_deg,
            l2cs_yaw=l2cs_yaw_local, l2cs_pitch=l2cs_pitch_local,
            world_yaw=world_yaw_deg, world_pitch=world_pitch_deg,
            ear=ear, is_blinking=is_blinking,
            landmarks=lm,
            R_head=R_head,
        )
        return features, is_blinking, dbg


# ──────────────────────────────────────────────────────────────────────
# GazeCalibrator (Multi-Model Evaluation & Auto-Selection)
# ──────────────────────────────────────────────────────────────────────
class GazeCalibrator:
    HIGH_DIM_THRESHOLD     = 50
    HIGH_DIM_DEFAULT_ALPHA = 200.0
    LOW_DIM_DEFAULT_ALPHA  = 50.0

    def __init__(self, alpha=None, outlier_mad_k=2.5):
        self._alpha_override     = alpha
        self._outlier_k          = outlier_mad_k
        self.models_with_weights = []   # [(pipeline, weight), ...]
        self.best_model_name     = ""
        self.is_calibrated       = False
        self._n_features         = None
        self.calib_features      = None
        self.calib_targets       = None

    def _auto_alpha(self, n_features):
        if self._alpha_override is not None:
            return self._alpha_override
        return (self.HIGH_DIM_DEFAULT_ALPHA if n_features >= self.HIGH_DIM_THRESHOLD
                else self.LOW_DIM_DEFAULT_ALPHA)

    def _build_ridge_pipeline(self, n_features, alpha):
        if n_features >= self.HIGH_DIM_THRESHOLD:
            return make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=True))
        else:
            return make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False),
                StandardScaler(),
                Ridge(alpha=alpha, fit_intercept=True),
            )

    def _reject_outliers(self, X, y):
        n_feat = X.shape[1]
        if n_feat >= 522:
            l2cs_slice = slice(516, 518)
        elif n_feat >= 18:
            l2cs_slice = slice(n_feat - 6, n_feat - 4)
        else:
            l2cs_slice = slice(0, n_feat)

        unique_targets = np.unique(y, axis=0)
        X_out, y_out, dropped = [], [], 0
        k = self._outlier_k

        for tgt in unique_targets:
            mask = np.all(y == tgt, axis=1)
            Xi = X[mask]; yi = y[mask]
            if len(Xi) <= 4:
                X_out.append(Xi); y_out.append(yi)
                continue
            sub  = Xi[:, l2cs_slice].astype(np.float64)
            med  = np.median(sub, axis=0)
            dist = np.linalg.norm(sub - med, axis=1)
            mad  = np.median(np.abs(dist - np.median(dist))) + 1e-9
            good = dist <= (np.median(dist) + k * mad * 1.4826)
            dropped += int(np.sum(~good))
            X_out.append(Xi[good]); y_out.append(yi[good])

        return np.vstack(X_out), np.vstack(y_out), dropped

    def fit(self, features, targets_px):
        X = np.array(features,   dtype=np.float64)
        y = np.array(targets_px, dtype=np.float64)
        n_feat = X.shape[1]
        self._n_features = n_feat

        X_clean, y_clean, n_dropped = self._reject_outliers(X, y)
        print(f"\n[Data Prep] Outlier rejection: {n_dropped} frames removed "
              f"({len(X_clean)}/{len(X)} kept)")

        if len(X_clean) < 50:
            print("  WARNING: Too few clean samples for reliable multi-model evaluation.")
            X_clean, y_clean = X, y

        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )

        alpha     = self._auto_alpha(n_feat)
        dim_label = ('high-dim/Ridge' if n_feat >= self.HIGH_DIM_THRESHOLD
                     else 'low-dim/Poly+Ridge')

        # SVR does not natively support multi-output, so wrap with MultiOutputRegressor
        candidate_models = {
            "Ridge (Baseline)": self._build_ridge_pipeline(n_feat, alpha),
            "SVR (RBF Kernel)": make_pipeline(
                StandardScaler(),
                MultiOutputRegressor(SVR(kernel='rbf', C=10.0, gamma='scale'))
            ),
            "RandomForest": make_pipeline(
                StandardScaler(),
                RandomForestRegressor(n_estimators=100, max_depth=10,
                                      random_state=42, n_jobs=-1)
            ),
            "MLP (Neural Net)": make_pipeline(
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                             early_stopping=True, random_state=42)
            ),
        }

        print(f"\n=== Model Evaluation (Train: {len(X_train)} / Test: {len(X_test)}) ===")
        print(f"  Feature dim  : {n_feat}  ({dim_label})")
        print(f"  Ridge alpha  : {alpha}")

        eval_results = []   # [(name, test_error, pipeline)]
        for name, pipeline in candidate_models.items():
            t_start = time.time()
            pipeline.fit(X_train, y_train)
            pred          = pipeline.predict(X_test)
            errors        = np.linalg.norm(pred - y_test, axis=1)
            mean_px_error = float(np.mean(errors))
            t_elapsed     = time.time() - t_start
            print(f"  {name:<22} : Test Error = {mean_px_error:5.1f} px  ({t_elapsed:.2f}s)")
            eval_results.append((name, mean_px_error, pipeline))

        eval_results.sort(key=lambda x: x[1])
        best_name = eval_results[0][0]

        # Inverse-error weights: better models get proportionally higher weight
        test_errors = np.array([r[1] for r in eval_results])
        inv_errors  = 1.0 / (test_errors + 1e-9)
        weights     = inv_errors / inv_errors.sum()

        print(f"\n  Ensemble weights (best → worst):")
        for (name, err, _), w in zip(eval_results, weights):
            print(f"    {name:<22} : err={err:5.1f}px  weight={w:.3f}")

        print("  Re-training all models on ALL clean data...")
        self.models_with_weights = []
        for (name, _, pipeline), w in zip(eval_results, weights):
            pipeline.fit(X_clean, y_clean)
            self.models_with_weights.append((pipeline, float(w)))

        pred_all = self._ensemble_predict(X_clean)
        rmse = np.sqrt(np.mean((pred_all - y_clean) ** 2, axis=0))
        print(f"  Ensemble RMSE: x={rmse[0]:.1f}px  y={rmse[1]:.1f}px  "
              f"(diagonal={np.hypot(rmse[0], rmse[1]):.1f}px)")

        self.best_model_name = best_name
        self.calib_features  = X_clean
        self.calib_targets   = y_clean
        self.is_calibrated   = True

    def _ensemble_predict(self, X):
        pred = np.zeros((X.shape[0], 2), dtype=np.float64)
        for pipeline, w in self.models_with_weights:
            pred += w * pipeline.predict(X)
        return pred

    def predict(self, feature):
        if not self.is_calibrated:
            return None
        X = np.array(feature, dtype=np.float64).reshape(1, -1)
        return self._ensemble_predict(X)[0]

    def per_point_accuracy(self):
        if not self.is_calibrated:
            return []
        results = []
        unique = np.unique(self.calib_targets, axis=0)
        for tgt in unique:
            mask      = np.all(self.calib_targets == tgt, axis=1)
            mean_feat = self.calib_features[mask].mean(axis=0)
            pred      = self.predict(mean_feat)
            err       = float(np.linalg.norm(pred - tgt))
            results.append((tgt[0], tgt[1], pred[0], pred[1], err))
        return results


class GazeDebugInfo:
    __slots__ = ['head_yaw','head_pitch','head_roll',
                 'l2cs_yaw','l2cs_pitch',
                 'world_yaw','world_pitch',
                 'ear','is_blinking','landmarks','R_head']
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)


# ──────────────────────────────────────────────────────────────────────
# Calibration Grid
# ──────────────────────────────────────────────────────────────────────
def build_calibration_grid(W, H, rows=3, cols=3,
                           margin_x=0.05, margin_y=0.05,
                           top_offset_px=0, order="serpentine"):
    mx     = int(W * margin_x)
    my     = int(H * margin_y) + top_offset_px
    my_bot = int(H * margin_y)
    gw     = W - 2 * mx
    gh     = H - my - my_bot
    sx     = gw / (cols - 1) if cols > 1 else 0
    sy     = gh / (rows - 1) if rows > 1 else 0
    pts    = []
    for r in range(rows):
        row = [(mx + int(c * sx), my + int(r * sy)) for c in range(cols)]
        if order == "serpentine" and r % 2 == 1:
            row = row[::-1]
        pts.extend(row)
    return pts


# ──────────────────────────────────────────────────────────────────────
# wait_for_pose
# ──────────────────────────────────────────────────────────────────────
def wait_for_pose(estimator, cam, W, H, phase, win_name,
                  stable_dur=2.0):
    yaw_lo, yaw_hi = phase_yaw_range(phase)
    pitch_lo, pitch_hi = phase_pitch_range(phase)
    in_since = None
    phase_col = phase["color"]

    print(f"\n[POSE GATE] Waiting for {phase['name']}  "
          f"yaw ∈ [{yaw_lo:.0f}°, {yaw_hi:.0f}°], pitch ∈ [{pitch_lo:.0f}°, {pitch_hi:.0f}°]")

    while True:
        frame = cam.read()
        if frame is None: continue

        feat, blink, dbg = estimator.process(frame)
        now = time.time()

        canvas = np.zeros((H, W, 3), np.uint8)

        if dbg is not None and not blink:
            yaw = dbg.head_yaw
            pitch = dbg.head_pitch
            in_zone = (
                yaw_lo   <= yaw   <= yaw_hi and
                pitch_lo <= pitch <= pitch_hi
            )

            if in_zone:
                if in_since is None: in_since = now
                elapsed = now - in_since
                if elapsed >= stable_dur:
                    return True
                # カウントダウン円
                t = elapsed / stable_dur
                ease = t*t*(3-2*t)
                ang  = int(360 * ease)
                cv2.ellipse(canvas, (W//2, H//2 - 60), (60, 60),
                            0, -90, -90+ang, phase_col, 6)
                cv2.putText(canvas, f"{stable_dur - elapsed:.1f}s",
                            (W//2 - 28, H//2 - 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, phase_col, 2)
            else:
                in_since = None

            # ヨー＆ピッチバー
            draw_pose_bars(canvas, yaw, pitch, phase)

            # 矢印HUD（中央下）
            draw_head_pose_hud(canvas, yaw, pitch,
                               cx=W//2, cy=H//2 + 80, radius=70,
                               phase=phase, in_gate=in_zone)
        else:
            in_since = None
            cv2.putText(canvas, "Face not detected",
                        (W//2 - 180, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (50,50,255), 3)

        # 指示テキスト
        phase_label = f"Phase: {phase['label_ja']}  ({phase['name']})"
        put_text_ja(canvas, phase_label,
                    (W//2 - 200, 32),
                    font_size=32, color_bgr=phase_col)
        put_text_ja(canvas, phase["instruction_ja"],
                    (W//2 - 280, 88),
                    font_size=26, color_bgr=(220, 220, 220))
        cv2.putText(canvas, phase["instruction_en"],
                    (W//2 - 240, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 160, 160), 1, cv2.LINE_AA)

        # ゾーン状態
        if dbg is not None:
            in_zone = (yaw_lo <= dbg.head_yaw <= yaw_hi and
                       pitch_lo <= dbg.head_pitch <= pitch_hi)
            status_txt = "[OK] IN ZONE  hold still" if in_zone else "< adjust head pose >"
            status_col = phase_col if in_zone else (180, 180, 180)
            cv2.putText(canvas, status_txt,
                        (W//2 - 160, H//2 - 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_col, 2, cv2.LINE_AA)

        cv2.imshow(win_name, canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            return False


# ──────────────────────────────────────────────────────────────────────
# pulse_and_capture_gated  (ヨー＆ピッチゲート付き)
# ──────────────────────────────────────────────────────────────────────
def pulse_and_capture_gated(estimator, cam, pts, W, H, phase,
                             pulse_secs=0.8, capture_secs=1.5,
                             top_bar_px=56, win_name="Calibration",
                             phase_idx=0, n_phases=5,
                             min_frames_per_point=8):
    yaw_lo, yaw_hi = phase_yaw_range(phase)
    pitch_lo, pitch_hi = phase_pitch_range(phase)
    feats, targs   = [], []
    total          = len(pts)
    TOP_BAR        = top_bar_px
    INSTRUCTION    = f"{phase['label_ja']}: {phase['instruction_ja']}"
    phase_col      = phase["color"]

    for i, (x, y) in enumerate(pts):
        for attempt in range(3):   # 最大3回リトライ
            # ── pulse ────────────────────────────────────────────────────────
            ps = time.time()
            while True:
                dt = time.time() - ps
                if dt > pulse_secs: break
                frame = cam.read()
                if frame is None: continue
                canvas = np.zeros((H, W, 3), np.uint8)
                feat0, _, dbg0 = estimator.process(frame)
                if dbg0:
                    in_g = (yaw_lo <= dbg0.head_yaw <= yaw_hi and
                            pitch_lo <= dbg0.head_pitch <= pitch_hi)
                    draw_head_pose_hud(canvas, dbg0.head_yaw, dbg0.head_pitch,
                                       phase=phase, in_gate=in_g)
                    draw_pose_bars(canvas, dbg0.head_yaw, dbg0.head_pitch, phase)
                r = 12 + int(12 * abs(np.sin(np.pi * dt / 0.5)))
                cv2.circle(canvas, (x, y), r, phase_col, -1)
                _draw_top_bar_multi(canvas, W, TOP_BAR, i, total,
                                    INSTRUCTION, phase, phase_idx, n_phases,
                                    collecting=False)
                cv2.imshow(win_name, canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    return None

            # ── capture ──────────────────────────────────────────────────────
            cs = time.time()
            point_feats = []
            while True:
                dt = time.time() - cs
                if dt > capture_secs: break
                frame = cam.read()
                if frame is None: continue

                canvas = np.zeros((H, W, 3), np.uint8)
                feat, is_blink, dbg = estimator.process(frame)
                in_gate = False
                if dbg is not None:
                    in_gate = (yaw_lo <= dbg.head_yaw <= yaw_hi and
                               pitch_lo <= dbg.head_pitch <= pitch_hi)
                    draw_head_pose_hud(canvas, dbg.head_yaw, dbg.head_pitch,
                                       phase=phase, in_gate=in_gate)
                    draw_pose_bars(canvas, dbg.head_yaw, dbg.head_pitch, phase)
                cv2.circle(canvas, (x, y), 18, phase_col, -1)
                ease = (dt/capture_secs)**2 * (3 - 2*(dt/capture_secs))
                ang  = int(360 * (1 - ease))
                cv2.ellipse(canvas, (x, y), (36, 36), 0, -90, -90+ang, (255,255,255), 3)

                # ゲート外ならフレームを捨てて警告
                if not in_gate and dbg is not None:
                    cv2.putText(canvas, "HEAD POSE OUT OF RANGE",
                                (W//2 - 220, y - 50 if y > 80 else y + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,255), 2)

                if feat is not None and not is_blink and in_gate:
                    point_feats.append(feat)

                label_y = y + 50 if y < H - 60 else y - 30
                cv2.putText(canvas, f"{len(point_feats)}f",
                            (x + 44, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
                _draw_top_bar_multi(canvas, W, TOP_BAR, i, total,
                                    INSTRUCTION, phase, phase_idx, n_phases,
                                    collecting=True)
                cv2.imshow(win_name, canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    return None

            if len(point_feats) >= min_frames_per_point:
                feats.extend(point_feats)
                targs.extend([[x, y]] * len(point_feats))
                print(f"  [{phase['name']}] Point {i+1:2d}/{total}: "
                      f"({x:4d},{y:4d})  {len(point_feats)} frames  "
                      f"(attempt {attempt+1})")
                break
            else:
                print(f"  [{phase['name']}] Point {i+1}/{total}: only "
                      f"{len(point_feats)} frames (need {min_frames_per_point}), retry...")
                retry_canvas = np.zeros((H, W, 3), np.uint8)
                cv2.putText(retry_canvas, "Too few frames — adjust head pose and retry",
                            (W//2 - 340, H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,50,255), 2)
                cv2.imshow(win_name, retry_canvas)
                cv2.waitKey(1000)

    return feats, targs


def _draw_top_bar_multi(canvas, W, bar_h, current_idx, total,
                         instruction, phase, phase_idx, n_phases,
                         collecting=False):
    cv2.rectangle(canvas, (0, 0), (W, bar_h), (18, 18, 18), -1)

    # ポイント進捗バー
    fill = int((current_idx / total) * (W - 40))
    col  = phase["color"] if collecting else tuple(c//2 for c in phase["color"])
    cv2.rectangle(canvas, (20, bar_h - 18), (20 + fill, bar_h - 6), col, -1)
    cv2.rectangle(canvas, (20, bar_h - 18), (W - 20, bar_h - 6), (70, 70, 70), 1)

    # フェーズドット
    dot_x = W - 160
    for pi in range(n_phases):
        c = CALIB_PHASES[pi]["color"]
        filled = pi <= phase_idx
        cv2.circle(canvas, (dot_x + pi * 22, 18), 7,
                   c if filled else (40,40,40), -1 if filled else 1)

    # テキスト
    put_text_ja(canvas, instruction,
                (20, bar_h - 38),
                font_size=18, color_bgr=(220, 220, 220))
    counter = f"{current_idx + 1} / {total}"
    (tw, _), _ = cv2.getTextSize(counter, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    cv2.putText(canvas, counter,
                (W - tw - 180, bar_h - 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (160, 220, 255), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────
# run_multi_pose_calibration  ─ 5フェーズオーケストレーター
# ──────────────────────────────────────────────────────────────────────
def run_multi_pose_calibration(estimator, cam, calib, W, H, win_name,
                                rows=3, cols=3, margin=0.05,
                                pulse_secs=0.8, capture_secs=1.5,
                                top_bar_px=56,
                                min_frames_per_point=8):
    TOP_BAR = top_bar_px
    pts = build_calibration_grid(W, H,
                                  rows=rows, cols=cols,
                                  margin_x=margin, margin_y=margin,
                                  top_offset_px=TOP_BAR)
    total_pts = len(pts) * len(CALIB_PHASES)
    print(f"\n=== Extended Multi-Pose Calibration ===")
    print(f"  Grid   : {rows}×{cols} = {len(pts)} points per pose")
    print(f"  Phases : {len(CALIB_PHASES)} ({', '.join(p['name'] for p in CALIB_PHASES)})")
    print(f"  Total  : {total_pts} points")
    print(f"  Feature: {GazeEstimator.FEAT_DIM} dims")

    all_feats  = []
    all_targs  = []

    for phase_idx, phase in enumerate(CALIB_PHASES):
        print(f"\n--- Phase {phase_idx+1}/{len(CALIB_PHASES)}: {phase['name']} "
              f"({phase['label_ja']}) ---")

        ok = wait_for_pose(estimator, cam, W, H, phase,
                           win_name, stable_dur=2.0)
        if not ok:
            return False

        ready_canvas = np.zeros((H, W, 3), np.uint8)
        put_text_ja(ready_canvas,
                    f"{phase['label_ja']}  キャリブレーション開始！",
                    (W//2 - 300, H//2 - 28),
                    font_size=32, color_bgr=phase["color"])
        cv2.imshow(win_name, ready_canvas)
        cv2.waitKey(600)

        result = pulse_and_capture_gated(
            estimator, cam, pts, W, H, phase,
            pulse_secs=pulse_secs,
            capture_secs=capture_secs,
            top_bar_px=TOP_BAR,
            win_name=win_name,
            phase_idx=phase_idx,
            n_phases=len(CALIB_PHASES),
            min_frames_per_point=min_frames_per_point,
        )
        if result is None:
            return False

        feats, targs = result
        all_feats.extend(feats)
        all_targs.extend(targs)
        print(f"  Phase {phase_idx+1} done: {len(feats)} frames collected")

    n_samples = len(all_feats)
    n_feat    = len(all_feats[0]) if all_feats else 0
    print(f"\n=== Training ===")
    print(f"  Total samples : {n_samples}")
    print(f"  Feature dim   : {n_feat}")
    print(f"  Targets       : {len(pts)} unique screen positions")

    if n_samples < 30:
        print("  ERROR: Too few samples. Recalibrate.")
        return False

    train_canvas = np.zeros((H, W, 3), np.uint8)
    cv2.putText(train_canvas, f"Training...  {n_samples} samples",
                (W//2 - 250, H//2), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255,255,255), 2)
    cv2.imshow(win_name, train_canvas)
    cv2.waitKey(1)

    calib.fit(all_feats, all_targs)
    return True


# ──────────────────────────────────────────────────────────────────────
# Post-calibration accuracy display
# ──────────────────────────────────────────────────────────────────────
def show_calibration_accuracy(calib, W, H, win_name, duration=3.0):
    accuracy_data = calib.per_point_accuracy()
    if not accuracy_data:
        return

    canvas = np.zeros((H, W, 3), np.uint8)
    total_err = 0.0
    for tx, ty, px_, py_, err in accuracy_data:
        total_err += err
        color = (0, 220, 80) if err < 60 else (50, 50, 255)
        cv2.circle(canvas, (int(tx), int(ty)), 14, (180,180,180), 2)
        cv2.circle(canvas, (int(px_), int(py_)), 8, color, -1)
        cv2.line(canvas, (int(tx), int(ty)), (int(px_), int(py_)), color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{err:.0f}px", (int(px_)+10, int(py_)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    mean_err = total_err / max(len(accuracy_data), 1)
    label = f"Mean error: {mean_err:.1f}px   (circle=target, dot=prediction)"
    cv2.putText(canvas, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
    cv2.putText(canvas, "Starting tracking...", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 100), 1)

    t0 = time.time()
    while time.time() - t0 < duration:
        cv2.imshow(win_name, canvas)
        if cv2.waitKey(30) & 0xFF == 27:
            break


# ──────────────────────────────────────────────────────────────────────
# Debug Overlay (tracking mode)
# ──────────────────────────────────────────────────────────────────────
def _draw_debug_overlay(frame, dbg, fx, fy, feat):
    h, w = frame.shape[:2]
    panel_w = 360
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, 240), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    def put(text, row, color=(220,220,220)):
        cv2.putText(frame, text, (10, 24 + row*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)

    yaw_ok   = abs(dbg.head_yaw)   < 45
    pitch_ok = abs(dbg.head_pitch) < 45
    put(f"HEAD  yaw ={dbg.head_yaw:+6.1f}  pitch={dbg.head_pitch:+6.1f}  roll={dbg.head_roll:+5.1f}",
        0, (100,255,100) if yaw_ok and pitch_ok else (80,80,255))
    put(f"L2CS  yaw ={dbg.l2cs_yaw:+6.1f}  pitch={dbg.l2cs_pitch:+6.1f}  [cam-space]",  1, (255,200,80))
    put(f"WORLD yaw ={dbg.world_yaw:+6.1f}  pitch={dbg.world_pitch:+6.1f}  [head-comp]", 2, (80,200,255))
    put(f"GAZE  x   ={fx:+6.3f}  y    ={fy:+6.3f}", 3, (0,255,120))
    blink_str = "BLINK" if dbg.is_blinking else "open"
    put(f"EAR   {dbg.ear:.3f}  {blink_str}", 4,
        (80, 80, 255) if dbg.is_blinking else (180,180,180))
    feat_norm = float(np.linalg.norm(feat[:513])) if len(feat) >= 513 else 0.0
    put(f"FEAT  dim={len(feat)}  |face_norm|={feat_norm:.2f}", 5, (160,160,160))

    cx, cy = w - 100, 100
    _draw_gaze_arrow(frame, cx, cy, dbg.head_yaw, dbg.head_pitch,
                     color=(100,255,100), label="HEAD", radius=70)
    cx2, cy2 = w - 100, 260
    _draw_gaze_arrow(frame, cx2, cy2, dbg.world_yaw, dbg.world_pitch,
                     color=(80,200,255), label="WORLD", radius=70)


def _draw_gaze_arrow(frame, cx, cy, yaw_deg, pitch_deg, color, label, radius=60):
    cv2.circle(frame, (cx, cy), radius, (40,40,40), -1)
    cv2.circle(frame, (cx, cy), radius, color, 1)
    yaw_r   = np.radians(yaw_deg)
    pitch_r = np.radians(pitch_deg)
    dx =  int( np.sin(yaw_r)   * radius * 0.8)
    dy =  int( np.sin(pitch_r) * radius * 0.8)
    ex, ey = cx + dx, cy + dy
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), color, 2, tipLength=0.3)
    cv2.circle(frame, (cx, cy), 4, (255,255,255), -1)
    cv2.putText(frame, label,
                (cx - radius, cy + radius + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"y{yaw_deg:+.0f} p{pitch_deg:+.0f}",
                (cx - radius, cy + radius + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────
# Wait for Face Detection + Countdown
# ──────────────────────────────────────────────────────────────────────
def wait_for_face_countdown(estimator, cam, W, H, win_name, dur=2.0):
    fd_start = None
    while True:
        frame = cam.read()
        if frame is None: continue
        feat, blink, _ = estimator.process(frame)
        face_ok = feat is not None and not blink
        canvas = np.zeros((H, W, 3), np.uint8)
        now = time.time()
        if face_ok:
            if fd_start is None: fd_start = now
            elapsed = now - fd_start
            if elapsed >= dur:
                return True
            t = elapsed / dur
            ease = t*t*(3-2*t)
            ang  = int(360 * (1-ease))
            cv2.ellipse(canvas, (W//2, H//2), (55,55), 0, -90, -90+ang, (0,220,80), -1)
            cv2.putText(canvas, "Hold still...",
                        (W//2-100, H//2+80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2)
        else:
            fd_start = None
            cv2.putText(canvas, "Face not detected",
                        (W//2-200, H//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50,50,255), 3)
        cv2.imshow(win_name, canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            return False


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def get_screen_size():
    root = tk.Tk()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibrate',    action='store_true',
                    help='Run multi-pose calibration (frontal + left + right + up + down, 45 points)')
    ap.add_argument('--preview',      action='store_true')
    ap.add_argument('--debug',        action='store_true')
    ap.add_argument('--mesh',         action='store_true')
    ap.add_argument('--kalman_ema',   type=float, default=0.35)
    ap.add_argument('--osc_ip',       type=str,   default='127.0.0.1')
    ap.add_argument('--osc_port',     type=int,   default=8000)
    ap.add_argument('--model',        type=str,   default='models/L2CSNet_gaze360.pkl')
    ap.add_argument('--ridge_alpha',  type=float, default=None)
    ap.add_argument('--outlier_k',    type=float, default=2.5)
    ap.add_argument('--grid_rows',    type=int,   default=3,
                    help='Grid rows per pose (default 3; 3×3×5poses = 45 points)')
    ap.add_argument('--grid_cols',    type=int,   default=3,
                    help='Grid cols per pose (default 3)')
    ap.add_argument('--margin',       type=float, default=0.05)
    ap.add_argument('--pulse_secs',   type=float, default=0.8)
    ap.add_argument('--capture_secs', type=float, default=1.5)
    ap.add_argument('--min_frames',   type=int,   default=8,
                    help='Min gated frames per calibration point (default 8)')
    ap.add_argument('--show_accuracy', action='store_true', default=True)
    ap.add_argument('--yaw_sign_flip', action='store_true',
                    help='Flip yaw sign convention if LEFT/RIGHT phases are swapped on your camera')
    args = ap.parse_args()

    osc      = SimpleUDPClient(args.osc_ip, args.osc_port)
    W, H     = get_screen_size()
    cam      = AsyncCamera(src=0, width=1280, height=720)
    est      = GazeEstimator(1280, 720, model_path=args.model)
    calib    = GazeCalibrator(alpha=args.ridge_alpha, outlier_mad_k=args.outlier_k)
    smoother = KalmanEMAFilter(ema_alpha=args.kalman_ema)

    if args.yaw_sign_flip:
        for ph in CALIB_PHASES:
            ph["yaw_center"] *= -1
        print("NOTE: yaw_center signs flipped by --yaw_sign_flip")

    last_x, last_y = W/2, H/2
    mode = "CALIBRATION" if args.calibrate else "TRACKING"
    WIN  = "Calibration" if args.calibrate else "Preview"

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    if args.calibrate:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        if mode == "CALIBRATION":
            # 顔検出待機
            ok = wait_for_face_countdown(est, cam, W, H, WIN, dur=2.0)
            if not ok:
                return

            # 5フェーズキャリブレーション
            calib_ok = run_multi_pose_calibration(
                est, cam, calib, W, H, WIN,
                rows=args.grid_rows,
                cols=args.grid_cols,
                margin=args.margin,
                pulse_secs=args.pulse_secs,
                capture_secs=args.capture_secs,
                min_frames_per_point=args.min_frames,
            )

            if not calib_ok or not calib.is_calibrated:
                fail_canvas = np.zeros((H, W, 3), np.uint8)
                cv2.putText(fail_canvas, "Calibration failed. Re-run with --calibrate.",
                            (W//2 - 380, H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,50,255), 2)
                cv2.imshow(WIN, fail_canvas)
                cv2.waitKey(3000)
                return

            # 精度表示
            if args.show_accuracy:
                show_calibration_accuracy(calib, W, H, WIN, duration=3.5)

            cv2.destroyWindow(WIN)
            cv2.waitKey(1)
            mode = "TRACKING"
            if args.preview:
                cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)

        # ── TRACKING ──────────────────────────────────────────────────────────
        while mode == "TRACKING":
            frame = cam.read()
            if frame is None: continue

            feat, is_blink, dbg = est.process(frame)
            display = cv2.flip(frame, 1)

            sx, sy = last_x, last_y
            if feat is not None and not is_blink:
                pred = calib.predict(feat)
                if pred is not None:
                    rx, ry = pred[0], pred[1]
                    sx, sy = smoother.update(rx, ry)
                    sx = float(np.clip(sx, 0, W))
                    sy = float(np.clip(sy, 0, H))
                    last_x, last_y = sx, sy

            fx, fy = sx/W, sy/H
            osc.send_message("/gaze", [fx, fy])

            if args.preview:
                px_ = int(sx * display.shape[1] / W)
                py_ = int(sy * display.shape[0] / H)

                if args.mesh and dbg is not None and hasattr(dbg, 'landmarks'):
                    draw_face_mesh(
                        display, dbg.landmarks,
                        img_w=display.shape[1], img_h=display.shape[0],
                        flipped=True,
                        draw_contours=True, draw_irises=True,
                        draw_all_points=True, draw_solvepnp=True,
                    )

                # 視線ドット
                col = (0, 80, 255) if is_blink else (0, 255, 60)
                cv2.circle(display, (px_, py_), 18, col, -1)
                cv2.circle(display, (px_, py_), 18, (255,255,255), 2)

                # ヘッドポーズ HUD（右上コーナー）
                if dbg is not None:
                    draw_head_pose_hud(display, dbg.head_yaw, dbg.head_pitch,
                                       phase=None, in_gate=False)

                if args.debug and dbg is not None:
                    _draw_debug_overlay(display, dbg, fx, fy,
                                        feat if feat is not None
                                        else np.zeros(GazeEstimator.FEAT_DIM))

                cv2.imshow("Preview", display)
                if cv2.waitKey(1) & 0xFF == 27: break
            else:
                if cv2.waitKey(1) & 0xFF == 27: break

    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()