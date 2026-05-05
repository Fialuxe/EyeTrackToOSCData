import numpy as np
from collections import deque
from scipy.stats import gaussian_kde


class GazeKDEFilter:
    """
    KDE filter: outputs the density peak of the last `window_size` samples.
    Best suited for fixation estimation rather than smooth pursuit.
    """

    def __init__(self, cfg):
        self._bw  = cfg.kde_bandwidth
        self._buf = deque(maxlen=cfg.kde_window_size)
        self._res = 50

    def update(self, measurement) -> np.ndarray:
        """
        Args:
            measurement: array-like (2,) [x, y]
        Returns:
            np.ndarray (2,) KDE density peak
        """
        self._buf.append(np.array(measurement, dtype=float))

        if len(self._buf) < 5:
            return np.array(measurement, dtype=float)

        data = np.stack(self._buf, axis=1)  # (2, N)
        try:
            kde = gaussian_kde(data, bw_method=self._bw)
        except np.linalg.LinAlgError:
            return np.array(measurement, dtype=float)

        x_arr = np.linspace(data[0].min(), data[0].max(), self._res)
        y_arr = np.linspace(data[1].min(), data[1].max(), self._res)
        xg, yg = np.meshgrid(x_arr, y_arr)
        dens = kde(np.stack([xg.ravel(), yg.ravel()])).reshape(self._res, self._res)
        idx  = np.unravel_index(dens.argmax(), dens.shape)
        return np.array([x_arr[idx[1]], y_arr[idx[0]]], dtype=float)
