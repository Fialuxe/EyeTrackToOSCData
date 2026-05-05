import numpy as np
from .kalman import GazeKalmanFilter


class GazeHybridFilter:
    """
    Kalman + EMA hybrid.
    S_t = alpha * x_kalman + (1 - alpha) * S_{t-1}

    ema_alpha guide:
        0.2  → high stability, more lag
        0.35 → balanced (default)
        0.6  → fast response, more noise
    """

    def __init__(self, cfg):
        self._alpha  = cfg.ema_alpha
        self._kalman = GazeKalmanFilter(cfg)
        self._ema    = None

    def update(self, measurement) -> np.ndarray:
        """
        Args:
            measurement: array-like (2,) [x, y]
        Returns:
            np.ndarray (2,) smoothed [x, y]
        """
        x_k = self._kalman.update(measurement)

        if self._ema is None:
            self._ema = x_k.copy()
        else:
            self._ema = self._alpha * x_k + (1.0 - self._alpha) * self._ema

        return self._ema.copy()
