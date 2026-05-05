from .kalman import GazeKalmanFilter
from .kde    import GazeKDEFilter
from .hybrid import GazeHybridFilter


def build_filter(cfg):
    """Factory: returns a filter object with an update(measurement) -> np.ndarray (2,) method."""
    if cfg.filter_type == "kalman":
        return GazeKalmanFilter(cfg)
    elif cfg.filter_type == "kde":
        return GazeKDEFilter(cfg)
    else:
        return GazeHybridFilter(cfg)
