#!/usr/bin/env python3
"""Geometric gaze-to-screen projection test.

Projects L2CS-Net gaze angles onto screen coordinates using face bounding box
height as a depth proxy — no calibration needed.

Physical model
--------------
  mm_per_px   = FACE_HEIGHT_MM / face_height_pixels
  dx_on_screen = -DISTANCE_MM * tan(yaw)           (mm)
  dy_on_screen = -DISTANCE_MM * tan(pitch)/cos(yaw) (mm, correct 3D projection)
  pixel_offset = mm_offset / mm_per_px

Note: the Roboflow demo uses arccos(yaw)*tan(pitch) for dy which is incorrect;
the right formula is tan(pitch)/cos(yaw).

Usage
-----
  python geometric_gaze_test.py --preview
  python geometric_gaze_test.py --distance 600 --face-height 220 --osc-port 8000
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from l2cs import Pipeline
from pythonosc.udp_client import SimpleUDPClient


# ---------------------------------------------------------------------------
# Geometric projection
# ---------------------------------------------------------------------------

def gaze_to_screen_norm(
    yaw: float,
    pitch: float,
    face_h_px: float,
    cam_w: int,
    cam_h: int,
    distance_mm: float,
    face_height_mm: float,
    cam_offset_x_px: float = 0.0,
    cam_offset_y_px: float = 0.0,
) -> tuple[float, float]:
    """Return normalised screen coordinate (x, y) in [0, 1].

    cam_offset_x/y_px: how many pixels the camera centre is to the right of /
    below the screen centre. Positive = camera is right of / below screen centre.
    """
    if face_h_px <= 0:
        return 0.5, 0.5

    mm_per_px = face_height_mm / face_h_px

    dx_mm = -distance_mm * math.tan(yaw)

    # Guard against cos(yaw) → 0 (looking almost 90° sideways)
    cos_yaw = math.cos(yaw)
    cos_yaw = math.copysign(max(abs(cos_yaw), 1e-3), cos_yaw)
    dy_mm = -distance_mm * math.tan(pitch) / cos_yaw

    dx_px = dx_mm / mm_per_px
    dy_px = dy_mm / mm_per_px

    # Screen centre in camera-image pixel space
    screen_cx = cam_w / 2.0 - cam_offset_x_px
    screen_cy = cam_h / 2.0 - cam_offset_y_px

    gaze_x = (screen_cx + dx_px) / cam_w
    gaze_y = (screen_cy + dy_px) / cam_h
    return float(gaze_x), float(gaze_y)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quadrant(nx: float, ny: float) -> str:
    h = "left" if nx < 0.33 else ("right" if nx > 0.67 else "center")
    v = "top"  if ny < 0.33 else ("bottom" if ny > 0.67 else "center")
    return f"{v}-{h}"


def _draw(frame: np.ndarray, results, cam_w: int, cam_h: int,
          distance_mm: float, face_height_mm: float,
          cam_offset_x: float, cam_offset_y: float,
          fps: float, osc: SimpleUDPClient | None) -> None:

    if results is None or results.pitch.size == 0:
        cv2.putText(frame, "NO FACE DETECTED", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return

    pitch = float(results.pitch[0])
    yaw   = float(results.yaw[0])
    box   = results.bboxes[0]

    x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
    x2 = max(x1 + 1, int(box[2])); y2 = max(y1 + 1, int(box[3]))
    face_h_px = y2 - y1
    face_cx, face_cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Face box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)

    # Gaze arrow from face centre
    arrow_len = face_h_px * 1.5
    adx = int(-math.sin(yaw)   * math.cos(pitch) * arrow_len)
    ady = int(-math.sin(pitch) * arrow_len)
    cv2.arrowedLine(frame, (face_cx, face_cy),
                    (face_cx + adx, face_cy + ady),
                    (0, 60, 255), 2, tipLength=0.2)

    # Geometric projection
    nx, ny = gaze_to_screen_norm(
        yaw, pitch, face_h_px, cam_w, cam_h,
        distance_mm, face_height_mm, cam_offset_x, cam_offset_y,
    )

    if osc:
        osc.send_message("/gaze", [nx, ny, 0.0])

    # Gaze dot (clipped to frame for display)
    gaze_px = int(np.clip(nx * cam_w, 0, cam_w - 1))
    gaze_py = int(np.clip(ny * cam_h, 0, cam_h - 1))
    cv2.circle(frame, (gaze_px, gaze_py), 22, (255, 255, 255), 3)
    cv2.circle(frame, (gaze_px, gaze_py), 18, (0, 80, 255), -1)

    # Labels
    quad = _quadrant(nx, ny)
    mm_per_px = face_height_mm / face_h_px if face_h_px > 0 else 0
    depth_est = distance_mm  # physical constant, shown for reference

    lines = [
        f"yaw {math.degrees(yaw):+6.1f}  pitch {math.degrees(pitch):+6.1f}  deg",
        f"screen ({nx:.3f}, {ny:.3f})  [{quad}]",
        f"face_h {face_h_px}px  ~{mm_per_px:.1f}mm/px  dist {depth_est:.0f}mm",
    ]
    ty = y1 - 10 - 18 * (len(lines) - 1)
    for line in lines:
        cv2.putText(frame, line, (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 0), 1, cv2.LINE_AA)
        ty += 18

    cv2.putText(frame, f"FPS {fps:4.1f}", (8, cam_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights",   default="models/L2CSNet_gaze360.pkl")
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--camera",    type=int,   default=0)
    parser.add_argument("--distance",  type=float, default=600.0,
                        help="Distance from face to screen in mm (default: 600)")
    parser.add_argument("--face-height", type=float, default=250.0,
                        help="Physical face height in mm (default: 250)")
    parser.add_argument("--cam-offset-x", type=float, default=0.0,
                        help="Camera centre offset from screen centre, horizontal in px "
                             "(positive = camera is to the right of screen centre)")
    parser.add_argument("--cam-offset-y", type=float, default=0.0,
                        help="Camera centre offset from screen centre, vertical in px "
                             "(positive = camera is below screen centre)")
    parser.add_argument("--osc-ip",   default="127.0.0.1")
    parser.add_argument("--osc-port", type=int, default=0,
                        help="OSC UDP port (0 = disabled)")
    parser.add_argument("--fps",      type=float, default=30.0)
    args = parser.parse_args()

    print("Loading L2CS-Net + RetinaFace pipeline …")
    pipeline = Pipeline(
        weights=Path(args.weights),
        arch="ResNet50",
        device=torch.device(args.device),
        include_detector=True,
        confidence_threshold=0.5,
    )

    osc = SimpleUDPClient(args.osc_ip, args.osc_port) if args.osc_port else None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera.")

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    interval = 1.0 / args.fps

    print(f"Camera {cam_w}x{cam_h}  |  distance {args.distance}mm  "
          f"face-height {args.face_height}mm  "
          f"cam-offset ({args.cam_offset_x}, {args.cam_offset_y})px")
    print("Press Q to quit.")

    prev = time.perf_counter()
    fps  = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        try:
            results = pipeline.step(frame)
        except Exception:
            results = None

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now

        _draw(frame, results, cam_w, cam_h,
              args.distance, args.face_height,
              args.cam_offset_x, args.cam_offset_y,
              fps, osc)

        cv2.imshow("Geometric Gaze Test  [Q=quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # soft FPS cap
        elapsed = time.perf_counter() - now
        if elapsed < interval:
            time.sleep(interval - elapsed)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
