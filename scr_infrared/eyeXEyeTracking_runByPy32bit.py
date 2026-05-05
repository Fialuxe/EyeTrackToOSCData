import time
import os
import sys
import tkinter as tk

# pythonnetのインポート
import clr

# DLLの読み込み設定
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    clr.AddReference("EyeXFramework")
    import EyeXFramework
    import Tobii.EyeX.Framework
except Exception as e:
    print(f"DLLの読み込みに失敗しました: {e}")
    sys.exit(1)

# --- グローバル変数（GUIとバックグラウンド処理で共有） ---
current_norm_x = 0.5  # 正規化されたX座標 (0.0 ~ 1.0)
current_norm_y = 0.5  # 正規化されたY座標 (0.0 ~ 1.0)
screen_width = 1920   # 初期値 (後でTkinterから取得して上書きします)
screen_height = 1080  # 初期値

def gaze_handler(sender, e):
    """ Tobiiからデータが送られてくるたびに呼ばれる関数 (バックグラウンド) """
    global current_norm_x, current_norm_y
    
    if screen_width > 0 and screen_height > 0:
        # 1. 画面サイズで割って 0.0 ~ 1.0 に正規化
        nx = e.X / screen_width
        ny = e.Y / screen_height
        
        # 2. 画面外を見たときのエラーを防ぐため、0.0〜1.0の範囲に収める (Clamp処理)
        current_norm_x = max(0.0, min(1.0, nx))
        current_norm_y = max(0.0, min(1.0, ny))

def update_gui():
    """ GUIの画面を定期的に書き換える関数 (メインスレッド) """
    # 数値をテキストとして表示
    coord_label.config(text=f"正規化座標: X={current_norm_x:.3f}, Y={current_norm_y:.3f}")
    
    # キャンバス（描画エリア）の現在のサイズを取得
    canvas_w = canvas.winfo_width()
    canvas_h = canvas.winfo_height()
    
    # 正規化座標(0~1)を、キャンバス上のピクセル座標に変換
    cx = current_norm_x * canvas_w
    cy = current_norm_y * canvas_h
    
    # 赤い丸の描画位置を更新
    r = 15 # 丸の半径
    canvas.coords(gaze_circle, cx - r, cy - r, cx + r, cy + r)
    
    # 約16ミリ秒後（約60fps）に再びこの関数を呼び出す
    root.after(16, update_gui)

def main():
    global screen_width, screen_height

    # --- 1. GUIのセットアップ ---
    global root, coord_label, canvas, gaze_circle
    root = tk.Tk()
    root.title("Tobii Gaze Viewer (32bit)")
    root.geometry("800x600") # ウィンドウの初期サイズ
    
    # PCの実際の画面解像度を取得
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # 座標表示用ラベル
    coord_label = tk.Label(root, text="データ待機中...", font=("Arial", 16))
    coord_label.pack(pady=10)

    # 視線描画用キャンバス（黒背景）
    canvas = tk.Canvas(root, bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)

    # 視線を示す赤い丸を作成（最初は見えない位置に）
    gaze_circle = canvas.create_oval(0, 0, 0, 0, fill="red", outline="white", width=2)

    # --- 2. Tobiiのセットアップ ---
    print("Tobii ホストを初期化中...")
    host = EyeXFramework.EyeXHost()
    host.Start()
    stream = host.CreateGazePointDataStream(Tobii.EyeX.Framework.GazePointDataMode.LightlyFiltered)
    stream.Next += gaze_handler

    print("ストリーミング開始。ウィンドウを閉じると終了します。")

    # --- 3. GUIループ開始 ---
    update_gui()     # 定期更新ループをスタート
    root.mainloop()  # GUIを表示して待機（ここでプログラムはブロックされます）

    # --- 4. 終了処理 ---
    print("終了処理を実行します...")
    host.Dispose()
    print("終了しました。")

if __name__ == "__main__":
    main()