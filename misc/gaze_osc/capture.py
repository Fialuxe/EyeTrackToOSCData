import cv2
import threading


class FrameCapture:
    def __init__(self, cfg):
        cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.capture_height)
        cap.set(cv2.CAP_PROP_FPS,          cfg.capture_fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap   = cap
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._t     = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        self._stop.set()
        self._t.join(timeout=1.0)
        self._cap.release()
