from dataclasses import dataclass, field
from typing import Literal
import pathlib

BASE_DIR = pathlib.Path(__file__).parent


@dataclass
class Config:
    # Camera
    camera_index: int   = 0
    capture_width: int  = 1280
    capture_height: int = 720
    capture_fps: int    = 60

    # OSC
    osc_ip:   str = "127.0.0.1"
    osc_port: int = 8000
    send_fps: int = 30

    # L2CS-Net
    l2cs_weights: pathlib.Path = BASE_DIR / "models" / "L2CSNet_gaze360.pkl"
    l2cs_arch: str             = "ResNet50"
    l2cs_device: str           = "cpu"

    # MediaPipe FaceMesh
    mp_refine_landmarks: bool          = True
    mp_max_num_faces: int              = 1
    mp_min_detection_confidence: float = 0.7
    mp_min_tracking_confidence: float  = 0.5

    # Head Pose (solvePnP)
    # 3D face model points [mm]. Origin = nose tip. OpenCV coords (X right, Y down, Z into screen).
    # Corresponding MediaPipe landmark indices: nose=1, chin=152, left_eye=263, right_eye=33, left_mouth=287, right_mouth=57
    face_3d_pts_mm: list = field(default_factory=lambda: [
        [ 0.0,    0.0,    0.0  ],
        [ 0.0,  -63.6,  -12.5 ],
        [-43.3,  32.7,  -26.0 ],
        [ 43.3,  32.7,  -26.0 ],
        [-28.9, -28.9,  -24.1 ],
        [ 28.9, -28.9,  -24.1 ],
    ])
    face_landmark_indices: list = field(default_factory=lambda:
        [1, 152, 263, 33, 287, 57])

    # Blink (EAR 6-point)
    # Right eye: P1=33, P2=159, P3=158, P4=133, P5=153, P6=145
    # Left eye:  P1=362, P2=385, P3=387, P4=263, P5=373, P6=380
    right_eye_ear_indices: list = field(default_factory=lambda:
        [33, 159, 158, 133, 153, 145])
    left_eye_ear_indices: list  = field(default_factory=lambda:
        [362, 385, 387, 263, 373, 380])
    ear_threshold: float   = 0.21
    ear_consec_frames: int = 2

    # Filter
    filter_type: Literal["kalman", "kde", "hybrid"] = "hybrid"

    # Kalman
    kalman_process_noise: float = 1e-3
    kalman_obs_noise: float     = 1e-1

    # KDE
    kde_bandwidth: float  = 0.03
    kde_window_size: int  = 60

    # EMA (used inside hybrid)
    ema_alpha: float = 0.35

    # Screen Projection
    screen_width: int         = 1920
    screen_height: int        = 1080
    screen_distance_mm: float = 600.0
    camera_fov_h_deg: float   = 60.0

    # Calibration
    calib_dwell_time: float = 1.5
    calib_grid_size:  int   = 5       # NxN grid; 5 → 25 points, 3 → 9 points
    calib_tps_lambda: float = 1e-3    # TPS regularisation (larger = smoother, less overshoot)
