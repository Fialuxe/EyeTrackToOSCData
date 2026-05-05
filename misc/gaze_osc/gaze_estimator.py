import torch
import numpy as np
from l2cs import Pipeline, select_device


class GazeEstimator:
    def __init__(self, cfg):
        device = select_device(cfg.l2cs_device, batch_size=1)
        self._pipeline = Pipeline(
            weights=cfg.l2cs_weights,
            arch=cfg.l2cs_arch,
            device=device,
        )

    def predict(self, bgr_frame: np.ndarray):
        """
        Args:
            bgr_frame: full BGR frame — do NOT pass a cropped ROI

        Returns:
            pitch_rad : float (radians), positive = looking down. None on failure.
            yaw_rad   : float (radians), positive = looking right. None on failure.
            bbox      : np.ndarray (4,) [x1,y1,x2,y2] or None
        """
        with torch.no_grad():
            results = self._pipeline.step(bgr_frame)

        if results is None:
            return None, None, None
        if results.pitch is None or len(results.pitch) == 0:
            return None, None, None

        pitch_rad = float(results.pitch[0])
        yaw_rad   = float(results.yaw[0])

        bbox = None
        if results.bboxes is not None and len(results.bboxes) > 0:
            bbox = results.bboxes[0]

        return pitch_rad, yaw_rad, bbox
