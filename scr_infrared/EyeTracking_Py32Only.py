import time
import os
import sys
import tkinter as tk

# pythonnet for Tobii .NET SDK
import clr

# 32bits test
is_32bit = sys.maxsize <= 2**32 -  1
if not is_32bit:
    sys.exit("This script requires 32-bit Python environment. if you installed 64-bit python, please install 32-bit python.")


# OSC
try:
    from pythonosc import udp_client
except ImportError as e:
    print(f"pythonosc not installed. Run: pip install python-osc", file=sys.stderr)
    sys.exit(1)


# --- OSC Configuration ---
# 127.0.0.1 is localhost - change if Unity is running on another machine
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
current_norm_x = 0.5  # Normalized X (0.0 to 1.0)
current_norm_y = 0.5  # Normalized Y (0.0 to 1.0)
screen_width = 1920   # Will be updated by Tkinter
screen_height = 1080  # Will be updated by Tkinter
osc_client = None


def gaze_handler(sender, e):
    """
    Background thread handler called by Tobii whenever new gaze data arrives.
    """
    global current_norm_x, current_norm_y
    
    if screen_width > 0 and screen_height > 0:
        # Normalize to 0.0 ~ 1.0
        nx = e.X / screen_width
        ny = e.Y / screen_height
        
        # Clamp to bounds to prevent out-of-screen errors
        current_norm_x = max(0.0, min(1.0, nx))
        current_norm_y = max(0.0, min(1.0, ny))


def update_gui():
    """
    Periodic GUI loop that updates the dot and sends OSC data (~60 FPS).
    """
    # Frame variables
    coord_label.config(text=f"Normalized Gaze: X={current_norm_x:.3f}, Y={current_norm_y:.3f}")
    
    canvas_w = canvas.winfo_width()
    canvas_h = canvas.winfo_height()
    
    cx = current_norm_x * canvas_w
    cy = current_norm_y * canvas_h
    
    # Redraw standard circle
    r = 15
    canvas.coords(gaze_circle, cx - r, cy - r, cx + r, cy + r)
    
    # Send OSC (x, y, dummy pupil parameter 0.0)
    if osc_client:
        try:
            osc_client.send_message("/gaze", [float(current_norm_x), float(current_norm_y), 0.0])
        except Exception as e:
            print(f"OSC Send Error: {e}")
            
    # Loop back in 16ms
    root.after(16, update_gui)


def main():
    global screen_width, screen_height, osc_client
    global root, coord_label, canvas, gaze_circle
    
    print(f"Creating OSC client: {OSC_IP}:{OSC_PORT}")
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

    # --- GUI Setup ---
    root = tk.Tk()
    root.title("Tobii Gaze OSC Streamer (32bit)")
    root.geometry("800x600")
    
    # Setup correct screen resolution using the Tk root
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

    print("Streaming started. Close the window to exit cleanly.")

    # --- Start Loops ---
    update_gui()
    root.mainloop()

    # --- Shutdown Hook ---
    print("Shutting down...")
    try:
        host.Dispose()
    except:
        pass
    print("Done.")


if __name__ == "__main__":
    main()
