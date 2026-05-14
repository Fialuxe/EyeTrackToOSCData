import time
import os
import sys
import argparse
import ctypes

# 32bits test
is_32bit = sys.maxsize <= 2**32 - 1
if not is_32bit:
    sys.exit("This script requires 32-bit Python environment. if you installed 64-bit python, please install 32-bit python.")

# pythonnet for Tobii .NET SDK
import clr

# OSC
try:
    from pythonosc import udp_client
except ImportError as e:
    print(f"pythonosc not installed. Run: pip install python-osc", file=sys.stderr)
    sys.exit(1)

# Numpy
try:
    import numpy as np
except ImportError as e:
    print(f"numpy not installed. Run: pip install numpy", file=sys.stderr)
    sys.exit(1)

# --- OSC Configuration ---
OSC_IP = "127.0.0.1"
OSC_PORT = 8000

# --- DLL Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    clr.AddReference("EyeXFramework")
    import EyeXFramework
    import Tobii.EyeX.Framework
except Exception as e:
    print(f"Failed to load DLL: {e}", file=sys.stderr)
    print("Please make sure you are running a 32-bit Python environment.", file=sys.stderr)
    sys.exit(1)

# --- Global Variables ---
current_norm_x = 0.5
current_norm_y = 0.5
screen_width = 0
screen_height = 0
osc_client = None

# --- Filters ---
class KalmanEMA:
    """2-D Kalman (position + velocity) -> EMA."""
    def __init__(self, R: float = 1e-2, q_ratio: float = 0.01, alpha: float = 0.25):
        Q = R * q_ratio
        self._F = np.array([[1, 1, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]], dtype=np.float64)
        self._H = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0]], dtype=np.float64)
        self._Q = np.eye(4) * Q
        self._R = np.eye(2) * R
        self._alpha = alpha
        self.reset()

    def update(self, x: float, y: float):
        z = np.array([x, y])
        if not self._init:
            self._state = np.array([x, 0.0, y, 0.0])
            self._cov = np.eye(4)
            self._init = True
            self._ex, self._ey = x, y
            return x, y

        sp = self._F @ self._state
        cp = self._F @ self._cov @ self._F.T + self._Q
        S = self._H @ cp @ self._H.T + self._R
        K = cp @ self._H.T @ np.linalg.inv(S)
        self._state = sp + K @ (z - self._H @ sp)
        self._cov = (np.eye(4) - K @ self._H) @ cp

        kx, ky = float(self._state[0]), float(self._state[2])
        self._ex = self._alpha * kx + (1 - self._alpha) * self._ex
        self._ey = self._alpha * ky + (1 - self._alpha) * self._ey
        return self._ex, self._ey

    def reset(self):
        self._state = np.zeros(4)
        self._cov = np.eye(4)
        self._init = False
        self._ex = self._ey = None

gaze_filter = KalmanEMA(R=1e-2, q_ratio=0.01, alpha=0.25)


def get_screen_resolution():
    """Windows APIを使用して解像度を取得する（Headlessモード用）"""
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def gaze_handler(sender, e):
    """Tobiiからのデータ受信コールバック"""
    global current_norm_x, current_norm_y

    if screen_width > 0 and screen_height > 0:
        nx = e.X / screen_width
        ny = e.Y / screen_height

        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))

        fx, fy = gaze_filter.update(nx, ny)

        current_norm_x = max(0.0, min(1.0, fx))
        current_norm_y = max(0.0, min(1.0, fy))


def send_osc_data():
    """OSC送信処理の共通化"""
    if osc_client:
        try:
            osc_client.send_message("/gaze", [float(current_norm_x), float(current_norm_y), 0.0])
        except Exception as e:
            print(f"OSC Send Error: {e}")


def update_gui(root, canvas, coord_label, gaze_circle):
    """GUIモード用のループ"""
    coord_label.config(text=f"Normalized Gaze: X={current_norm_x:.3f}, Y={current_norm_y:.3f}")

    canvas_w = canvas.winfo_width()
    canvas_h = canvas.winfo_height()
    cx = current_norm_x * canvas_w
    cy = current_norm_y * canvas_h

    r = 15
    canvas.coords(gaze_circle, cx - r, cy - r, cx + r, cy + r)

    send_osc_data()
    root.after(16, update_gui, root, canvas, coord_label, gaze_circle)


def main():
    global screen_width, screen_height, osc_client

    parser = argparse.ArgumentParser(description="Tobii Gaze OSC Streamer")
    parser.add_argument("--headless", action="store_true", help="Disable GUI window and run in terminal only")
    args = parser.parse_args()

    print(f"Creating OSC client: {OSC_IP}:{OSC_PORT}")
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

    if args.headless:
        # Headlessモードのセットアップ
        screen_width, screen_height = get_screen_resolution()
        print(f"Headless mode enabled. Detected Resolution: {screen_width}x{screen_height}")
    else:
        # GUIモードのセットアップ
        import tkinter as tk
        root = tk.Tk()
        root.title("Tobii Gaze OSC Streamer (32bit)")
        root.geometry("800x600")

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        coord_label = tk.Label(root, text="Waiting for data...", font=("Arial", 16))
        coord_label.pack(pady=10)

        canvas = tk.Canvas(root, bg="black")
        canvas.pack(fill=tk.BOTH, expand=True)

        gaze_circle = canvas.create_oval(0, 0, 0, 0, fill="red", outline="white", width=2)

    # --- Tobii Setup ---
    print("Initializing Tobii Host...")
    try:
        host = EyeXFramework.EyeXHost()
        host.Start()
        stream = host.CreateGazePointDataStream(Tobii.EyeX.Framework.GazePointDataMode.LightlyFiltered)
        stream.Next += gaze_handler
    except Exception as e:
        print(f"Failed to start Tobii Host: {e}")
        print("Continuing with dummy data for test purposes...")

    # --- Start Loops ---
    if args.headless:
        print("Streaming started in background. Press Ctrl+C to exit.")
        try:
            while True:
                send_osc_data()
                time.sleep(0.016)  # ~60FPS
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
    else:
        print("Streaming started. Close the window to exit cleanly.")
        update_gui(root, canvas, coord_label, gaze_circle)
        root.mainloop()
        print("Shutting down...")

    # --- Shutdown Hook ---
    try:
        host.Dispose()
    except:
        pass
    print("Done.")

if __name__ == "__main__":
    main()