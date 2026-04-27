"""
Tobii EyeX OSC Server (32-bit Python)
This script loads the Tobii EyeX Framework via pythonnet, reads gaze data,
and streams the normalized coordinates via OSC to 127.0.0.1:8000.
"""

import time
import os
import sys
import ctypes

# 1. Setup paths to load DLLs
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 2. Import pythonnet and load the DLLs
try:
    import clr
    clr.AddReference("EyeXFramework")
    import EyeXFramework
    import Tobii.EyeX.Framework
except ImportError:
    print("Error: 'pythonnet' is not installed. Please run:")
    print("  python -m pip install pythonnet")
    sys.exit(1)
except Exception as e:
    print(f"Error loading EyeXFramework.dll: {e}")
    print("Make sure Tobii.EyeX.Client.Net20.dll, Tobii.EyeX.Client.dll, and EyeXFramework.dll are in the same folder.")
    sys.exit(1)

# 3. Import python-osc
try:
    from pythonosc import udp_client
except ImportError:
    print("Error: 'python-osc' is not installed. Please run:")
    print("  python -m pip install python-osc")
    sys.exit(1)

# --- CONFIGURATION ---
OSC_IP = "127.0.0.1"
OSC_PORT = 8000
TARGET_FPS = 30.0

# --- SHARED STATE ---
current_norm_x = 0.5
current_norm_y = 0.5
is_data_received = False

# Get Screen Dimensions using Windows API (No Tkinter required)
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)
print(f"Detected Display Resolution: {screen_width}x{screen_height}")

def gaze_handler(sender, e):
    """
    Callback fired by the Tobii Framework whenever new gaze data is available.
    Runs in a background thread.
    """
    global current_norm_x, current_norm_y, is_data_received
    
    if screen_width > 0 and screen_height > 0:
        # Normalize and clamp coordinates
        nx = e.X / screen_width
        ny = e.Y / screen_height
        
        current_norm_x = max(0.0, min(1.0, nx))
        current_norm_y = max(0.0, min(1.0, ny))
        is_data_received = True

def main():
    print(f"Initializing Tobii EyeX Framework OSC Server...")
    print(f"Target OSC: {OSC_IP}:{OSC_PORT}")
    
    # Setup OSC Client
    client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    
    # Setup Tobii Host
    host = EyeXFramework.EyeXHost()
    host.Start()
    
    # Create the data stream
    stream = host.CreateGazePointDataStream(Tobii.EyeX.Framework.GazePointDataMode.LightlyFiltered)
    stream.Next += gaze_handler
    
    print("Tobii Stream initialized. Broadcasting starting... Press [Ctrl+C] to stop.")
    
    frame_interval = 1.0 / TARGET_FPS
    last_print_time = time.time()
    
    try:
        while True:
            start_time = time.time()
            
            # 1. Send OSC Data
            # Format: /gaze x y pupil
            # Since pupil size isn't available in this callback, we use 0.0
            client.send_message("/gaze", [float(current_norm_x), float(current_norm_y), 0.0])
            
            # 2. Print debug log every 1 second
            if start_time - last_print_time >= 1.0:
                if is_data_received:
                    status = f"Gaze: x={current_norm_x:.3f} y={current_norm_y:.3f}"
                else:
                    status = "Gaze: WAITING FOR DATA..."
                    
                print(f"\r[Tobii OSC] {status}           ", end="", flush=True)
                last_print_time = start_time
                
            # 3. Maintain Target FPS
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        # Clean up resources
        host.Dispose()
        print("Tobii Host Disposed. Exiting.")

if __name__ == "__main__":
    main()
