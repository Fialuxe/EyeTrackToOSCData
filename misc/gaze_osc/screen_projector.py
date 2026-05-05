import numpy as np


class ScreenProjector:
    def __init__(self, cfg, homography=None):
        self._D     = cfg.screen_distance_mm
        self._fov   = np.radians(cfg.camera_fov_h_deg)
        self._fov_v = self._fov * (cfg.screen_height / cfg.screen_width)
        self.homography = homography  # np.ndarray (3,3) or None

    def project(self, gaze_vec: np.ndarray,
                eye_pos_mm: np.ndarray = None):
        """
        Args:
            gaze_vec   : np.ndarray (3,) unit vector in camera coords (OpenCV: X-right, Y-down, Z-forward)
            eye_pos_mm : np.ndarray (3,) eye 3D position [mm] in camera coords, or None (uses origin)

        Returns:
            u, v : float in [0.0, 1.0] — normalized screen coordinates
        """
        if eye_pos_mm is None:
            eye_pos_mm = np.zeros(3)

        gz = gaze_vec[2]
        if gz < 1e-9:
            return 0.5, 0.5

        # screen_distance_mm is the distance from the eye to the screen plane
        t = self._D / gz

        x_hit = eye_pos_mm[0] + t * gaze_vec[0]
        y_hit = eye_pos_mm[1] + t * gaze_vec[1]

        x_range = self._D * np.tan(self._fov   / 2.0)
        y_range = self._D * np.tan(self._fov_v / 2.0)
        u =  x_hit / (2.0 * x_range) + 0.5
        v =  y_hit / (2.0 * y_range) + 0.5

        if self.homography is not None:
            p = self.homography @ np.array([u, v, 1.0])
            u = p[0] / (p[2] + 1e-9)
            v = p[1] / (p[2] + 1e-9)

        return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))


def angles_to_vector(pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """
    L2CS-Net (Gaze360 weights) output → unit gaze vector in OpenCV camera coords
    (X=right, Y=down, Z=into screen).

    Gaze360 convention (from l2cs/utils.py gazeto3d):
      pitch > 0: looking down  → +Y in OpenCV
      yaw   > 0: looking to subject's right (camera's left) → -X in OpenCV
    """
    p, y = pitch_rad, yaw_rad
    v = np.array([
        -np.cos(p) * np.sin(y),
         np.sin(p),
         np.cos(p) * np.cos(y),
    ], dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-9)
