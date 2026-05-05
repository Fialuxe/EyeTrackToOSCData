import numpy as np
import time

try:
    from scipy.interpolate import RBFInterpolator
    _SCIPY = True
except ImportError:
    _SCIPY = False


class _TPSWarp:
    """
    Thin-plate spline warp in UV space using scipy RBFInterpolator.
    Falls back to a plain numpy homography if scipy is unavailable.
    """

    def __init__(self, src: np.ndarray, dst: np.ndarray, lam: float):
        if _SCIPY:
            # RBFInterpolator with TPS kernel, one interpolator per output dimension
            self._rbf_u = RBFInterpolator(src, dst[:, 0], kernel="thin_plate_spline",
                                           smoothing=lam * len(src))
            self._rbf_v = RBFInterpolator(src, dst[:, 1], kernel="thin_plate_spline",
                                           smoothing=lam * len(src))
            self._use_scipy = True
        else:
            import cv2
            s32 = src.astype(np.float32)
            d32 = dst.astype(np.float32)
            self._H, _ = cv2.findHomography(s32, d32, cv2.RANSAC, 5.0)
            self._use_scipy = False

    def __call__(self, u: float, v: float):
        if self._use_scipy:
            p = np.array([[u, v]])
            u_out = float(self._rbf_u(p))
            v_out = float(self._rbf_v(p))
        else:
            import numpy as np
            p = self._H @ np.array([u, v, 1.0])
            u_out = float(p[0] / (p[2] + 1e-9))
            v_out = float(p[1] / (p[2] + 1e-9))
        return float(np.clip(u_out, 0.0, 1.0)), float(np.clip(v_out, 0.0, 1.0))


class Calibrator:
    def __init__(self, cfg):
        g      = cfg.calib_grid_size
        margin = 0.1
        step   = (1.0 - 2 * margin) / max(g - 1, 1)
        self.TARGETS_UV = [
            (round(margin + c * step, 4), round(margin + r * step, 4))
            for r in range(g)
            for c in range(g)
        ]

        self._dwell  = cfg.calib_dwell_time
        self._lambda = cfg.calib_tps_lambda
        self._idx    = 0
        self._t0     = time.time()
        self._buf    = []
        self._src    = []
        self._dst    = []
        self._done   = False
        self._warp   = None

    # ── public API ────────────────────────────────────────────

    @property
    def current_target_uv(self):
        if self._idx < len(self.TARGETS_UV):
            return self.TARGETS_UV[self._idx]
        return None

    @property
    def total_targets(self) -> int:
        return len(self.TARGETS_UV)

    @property
    def current_index(self) -> int:
        return self._idx

    @property
    def dwell_progress(self) -> float:
        """0.0 → 1.0 fraction of dwell time elapsed for the current target."""
        return min((time.time() - self._t0) / self._dwell, 1.0)

    def update(self, u_raw: float, v_raw: float):
        if self._done or self._idx >= len(self.TARGETS_UV):
            return

        self._buf.append([u_raw, v_raw])

        if time.time() - self._t0 >= self._dwell:
            avg = np.mean(self._buf, axis=0)
            self._src.append(avg.tolist())
            self._dst.append(list(self.TARGETS_UV[self._idx]))
            self._buf.clear()
            self._idx += 1
            self._t0   = time.time()

            if self._idx >= len(self.TARGETS_UV):
                self._compute()
                self._done = True

    def is_done(self) -> bool:
        return self._done

    def get_warp(self):
        """Returns callable (u, v) -> (u, v), or None if calibration incomplete."""
        return self._warp

    # ── private ───────────────────────────────────────────────

    def _compute(self):
        src = np.array(self._src, dtype=np.float64)
        dst = np.array(self._dst, dtype=np.float64)
        self._warp = _TPSWarp(src, dst, self._lambda)
