import numpy as np


def _ear(pts: np.ndarray, idx: list) -> float:
    """
    6-point Eye Aspect Ratio: EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
    idx order: [P1, P2, P3, P4, P5, P6]
    """
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]
    vertical   = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2.0 * np.linalg.norm(p1 - p4)
    return float(vertical / (horizontal + 1e-9))


class BlinkDetector:
    def __init__(self, cfg):
        self._right_idx = cfg.right_eye_ear_indices
        self._left_idx  = cfg.left_eye_ear_indices
        self._threshold = cfg.ear_threshold
        self._consec    = cfg.ear_consec_frames
        self._counter   = 0
        self._blink_now = False

    def update(self, landmarks_px) -> bool:
        """
        Args:
            landmarks_px: np.ndarray (478, 2) or None

        Returns:
            True  → send /gaze/blink 1.0
            False → send /gaze normally

        Landmark loss (landmarks_px is None) is treated as a blink.
        """
        if landmarks_px is None:
            return True

        ear_r = _ear(landmarks_px, self._right_idx)
        ear_l = _ear(landmarks_px, self._left_idx)
        avg   = (ear_r + ear_l) / 2.0

        if avg < self._threshold:
            self._counter += 1
            if self._counter >= self._consec:
                self._blink_now = True
        else:
            self._blink_now = False
            self._counter   = 0

        return self._blink_now
