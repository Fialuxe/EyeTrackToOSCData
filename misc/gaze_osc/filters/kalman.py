import numpy as np

try:
    from filterpy.kalman import KalmanFilter as _FPKalman
    _FILTERPY = True
except ImportError:
    _FILTERPY = False


class GazeKalmanFilter:
    """
    Kalman filter for 2D gaze coordinates.
    State: [x, y, vx, vy]  Observation: [x, y]
    Falls back to a hand-rolled implementation if filterpy is unavailable.
    """

    def __init__(self, cfg):
        self._q    = cfg.kalman_process_noise
        self._r    = cfg.kalman_obs_noise
        self._init = False
        dt = 1.0 / 30.0

        if _FILTERPY:
            kf = _FPKalman(dim_x=4, dim_z=2)
            kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
            kf.H = np.array([[1,0,0,0],[0,1,0,0]], float)
            kf.R = np.eye(2) * self._r
            kf.Q = np.eye(4) * self._q
            kf.P = np.eye(4)
            self._kf     = kf
            self._use_fp = True
        else:
            self._F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
            self._H = np.array([[1,0,0,0],[0,1,0,0]], float)
            self._R = np.eye(2) * self._r
            self._Q = np.eye(4) * self._q
            self._P = np.eye(4)
            self._x = np.zeros(4)
            self._use_fp = False

    def update(self, measurement) -> np.ndarray:
        """
        Args:
            measurement: array-like (2,) [x, y]
        Returns:
            np.ndarray (2,) filtered [x, y]
        """
        z = np.array(measurement, dtype=float)

        if not self._init:
            if self._use_fp:
                self._kf.x = np.array([z[0], z[1], 0.0, 0.0])
            else:
                self._x = np.array([z[0], z[1], 0.0, 0.0])
            self._init = True

        if self._use_fp:
            self._kf.predict()
            self._kf.update(z)
            return self._kf.x[:2].copy()

        # Predict
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        # Update
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ (z - self._H @ self._x)
        self._P = (np.eye(4) - K @ self._H) @ self._P
        return self._x[:2].copy()
