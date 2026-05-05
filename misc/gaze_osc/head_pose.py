import numpy as np
import cv2


class HeadPoseEstimator:
    def __init__(self, cfg, frame_w: int, frame_h: int):
        self._model_pts = np.array(cfg.face_3d_pts_mm, dtype=np.float64)
        self._lm_idx    = cfg.face_landmark_indices

        # Heuristic: focal length = image width. Use cv2.calibrateCamera() for accuracy.
        focal = float(frame_w)
        cx, cy = frame_w / 2.0, frame_h / 2.0
        self._cam_mat = np.array([
            [focal, 0,     cx],
            [0,     focal, cy],
            [0,     0,     1 ],
        ], dtype=np.float64)
        self._dist = np.zeros((4, 1), dtype=np.float64)

        self._prev_rvec = None
        self._prev_tvec = None

    def estimate(self, landmarks_px: np.ndarray):
        """
        Returns:
            tvec    : np.ndarray (3,1) face center in camera coords [mm]
            rot_mat : np.ndarray (3,3) rotation matrix (reference only — not used for gaze correction with Gaze360 weights)
        """
        img_pts   = landmarks_px[self._lm_idx].astype(np.float64)
        use_guess = self._prev_rvec is not None

        ok, rvec, tvec = cv2.solvePnP(
            self._model_pts,
            img_pts,
            self._cam_mat,
            self._dist,
            rvec=self._prev_rvec,
            tvec=self._prev_tvec,
            useExtrinsicGuess=use_guess,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return np.zeros((3, 1)), np.eye(3)

        self._prev_rvec = rvec.copy()
        self._prev_tvec = tvec.copy()
        rot_mat, _ = cv2.Rodrigues(rvec)
        return tvec, rot_mat
