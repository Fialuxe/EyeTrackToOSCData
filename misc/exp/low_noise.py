"""
Tobii EyeX OSC Server (32-bit Python)
Refactored for thread-safe state management, EMA filtering, and blink detection.
"""

import time
import os
import sys
import ctypes
import argparse

# 1. Setup paths and load DLLs
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    import clr
    clr.AddReference("EyeXFramework")
    import EyeXFramework
    import Tobii.EyeX.Framework
except ImportError:
    sys.exit("Error: 'pythonnet' is not installed. Run: python -m pip install pythonnet")
except Exception as e:
    sys.exit(f"Error loading EyeXFramework.dll: {e}\nEnsure DLLs are in the same folder.")

try:
    from pythonosc import udp_client
except ImportError:
    sys.exit("Error: 'python-osc' is not installed. Run: python -m pip install python-osc")

# --- Default Configurations ---
OSC_IP = "127.0.0.1"
OSC_PORT = 8000
TARGET_FPS = 30.0
TRACKING_TIMEOUT_SEC = 0.15  # Tobiiがデータを送ってこない場合のタイムアウト（まばたき判定）


def get_screen_dimensions():
    """Windows APIを使用して解像度を取得する"""
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


class ExponentialMovingAverage:
    """
    高周波ノイズを抑制するためのEMAフィルタ。
    """
    def __init__(self, alpha):
        self.alpha = alpha
        self.last_x = None
        self.last_y = None

    def step(self, x, y):
        if self.last_x is None:
            self.last_x, self.last_y = x, y
            return x, y
            
        filtered_x = self.alpha * x + (1.0 - self.alpha) * self.last_x
        filtered_y = self.alpha * y + (1.0 - self.alpha) * self.last_y
        
        self.last_x, self.last_y = filtered_x, filtered_y
        return filtered_x, filtered_y

    def reset(self):
        """ロスト復帰時に古い状態に引っ張られないようリセットする"""
        self.last_x = None
        self.last_y = None


class GazeTracker:
    """
    Tobiiからのコールバックを受け取り、フィルタリングされた視線状態を安全に保持する。
    """
    def __init__(self, screen_width, screen_height, ema_alpha):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.filter = ExponentialMovingAverage(ema_alpha)
        
        self._norm_x = 0.5
        self._norm_y = 0.5
        self._last_update_time = 0.0

    def on_gaze_data_received(self, sender, e):
        """Tobiiフレームワークから非同期で呼ばれるコールバック"""
        if self.screen_width <= 0 or self.screen_height <= 0:
            return

        # 画面外の異常値を除外して正規化
        raw_nx = max(0.0, min(1.0, e.X / self.screen_width))
        raw_ny = max(0.0, min(1.0, e.Y / self.screen_height))

        filtered_nx, filtered_ny = self.filter.step(raw_nx, raw_ny)

        self._norm_x = filtered_nx
        self._norm_y = filtered_ny
        self._last_update_time = time.perf_counter()

    def get_current_state(self):
        """
        現在の視線座標と、データが有効（まばたきしていない）かを返す。
        Returns: (x, y, is_tracking_valid)
        """
        time_since_last_update = time.perf_counter() - self._last_update_time
        is_valid = time_since_last_update < TRACKING_TIMEOUT_SEC
        
        if not is_valid:
            self.filter.reset()
            
        return self._norm_x, self._norm_y, is_valid


def main():
    parser = argparse.ArgumentParser(description="Tobii EyeX OSC Server")
    parser.add_argument("--ema-alpha", type=float, default=0.15, help="Smoothing strength (0.0 to 1.0). Lower is smoother.")
    args = parser.parse_args()

    screen_width, screen_height = get_screen_dimensions()
    print(f"Detected Display: {screen_width}x{screen_height}")

    tracker = GazeTracker(screen_width, screen_height, args.ema_alpha)
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    
    host = EyeXFramework.EyeXHost()
    host.Start()
    
    stream = host.CreateGazePointDataStream(Tobii.EyeX.Framework.GazePointDataMode.LightlyFiltered)
    stream.Next += tracker.on_gaze_data_received
    
    print(f"Streaming to {OSC_IP}:{OSC_PORT} at {TARGET_FPS} FPS.")
    print("Press [Ctrl+C] to exit.")

    frame_interval = 1.0 / TARGET_FPS
    last_print_time = time.perf_counter()

    try:
        while True:
            start_time = time.perf_counter()

            nx, ny, is_valid = tracker.get_current_state()

            if is_valid:
                osc_client.send_message("/gaze", [float(nx), float(ny), 0.0])
                status = f"({nx:.3f}, {ny:.3f})"
            else:
                osc_client.send_message("/gaze/blink", 1.0)
                status = "BLINK / LOST"

            # 1秒ごとにステータスを出力
            if start_time - last_print_time >= 1.0:
                print(f"\r[Tobii] {status}          ", end="", flush=True)
                last_print_time = start_time

            # 目標FPSの維持 (perf_counter を使用して精度向上)
            elapsed = time.perf_counter() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        host.Dispose()

if __name__ == "__main__":
    main()