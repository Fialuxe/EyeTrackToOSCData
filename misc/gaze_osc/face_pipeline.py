import mediapipe as mp
import numpy as np
import cv2


class FacePipeline:
    def __init__(self, cfg):
        mp_fm = mp.solutions.face_mesh
        self._face_mesh = mp_fm.FaceMesh(
            max_num_faces=cfg.mp_max_num_faces,
            refine_landmarks=cfg.mp_refine_landmarks,
            min_detection_confidence=cfg.mp_min_detection_confidence,
            min_tracking_confidence=cfg.mp_min_tracking_confidence,
        )

    def process(self, bgr_frame: np.ndarray):
        """
        Returns:
            landmarks_px : np.ndarray (478, 2) float32 or None
                           478 points when refine_landmarks=True
            (h, w)       : frame size
        """
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return None, (h, w)

        lm = results.multi_face_landmarks[0].landmark
        pts = np.array(
            [[p.x * w, p.y * h] for p in lm],
            dtype=np.float32,
        )
        return pts, (h, w)
