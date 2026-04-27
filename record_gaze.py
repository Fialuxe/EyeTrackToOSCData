import time
import csv
import cv2
import numpy as np
import os

# Eyetrax imports (あなたの環境に合わせてください)
try:
    from eyetrax.gaze import GazeEstimator
    from eyetrax.calibration import run_9_point_calibration
    from eyetrax.utils.video import camera, iter_frames
    from eyetrax.utils.screen import get_screen_size
except ImportError:
    raise SystemExit("eyetrax not installed.")

def record_data():
    # ファイル名 (タイムスタンプ付き)
    filename = f"gaze_log_{int(time.time())}.csv"
    
    # モデルのロード
    gaze_estimator = GazeEstimator()
    run_9_point_calibration(gaze_estimator)
    screen_width, screen_height = get_screen_size()
    
    print(f"=== Recording to {filename} ===")
    print("SPACE: Start/Stop Recording | ESC: Quit")
    
    recording = False
    
    # CSVファイルの準備
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "raw_x", "raw_y"]) # ヘッダー
        
        with camera(0) as cap:
            for frame in iter_frames(cap):
                current_time = time.time()
                
                # 視線推定 (フィルタなしのRaw値を取得)
                features, blink = gaze_estimator.extract_features(frame)
                
                display_frame = frame.copy()
                
                if features is not None and not blink:
                    gaze_point = gaze_estimator.predict(np.array([features]))[0]
                    x, y = gaze_point
                    
                    # 画面内の座標に正規化 (0.0 - 1.0)
                    norm_x = x / screen_width
                    norm_y = y / screen_height
                    
                    # 録画中ならCSVに書き込み
                    if recording:
                        writer.writerow([current_time, norm_x, norm_y])
                        cv2.circle(display_frame, (50, 50), 20, (0, 0, 255), -1) # Rec Indicator
                    
                    # 画面に現在の視点を表示（確認用）
                    ix, iy = int(x), int(y)
                    cv2.circle(display_frame, (ix, iy), 10, (0, 255, 0), 2)

                # UI表示
                status_text = "RECORDING" if recording else "PAUSED (Press SPACE)"
                cv2.putText(display_frame, status_text, (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Gaze Recorder", display_frame)
                
                key = cv2.waitKey(1)
                if key == 27: # ESC
                    break
                elif key == 32: # SPACE
                    recording = not recording
                    print(f"Recording: {recording}")

if __name__ == "__main__":
    record_data()