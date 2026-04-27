import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==========================================
# 設定
# ==========================================
# ステップ1で作成されたCSVファイルを指定してください
csv_file = 'gaze_log_1768890546.csv'  # <--- ここを書き換える

# 図のデザイン設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.5

# ==========================================
# カルマンフィルタの実装
# ==========================================
def apply_kalman(data):
    # OpenCVのKalmanFilterを使用
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    
    # パラメータ調整（ここが「滑らかさ」の肝）
    # processNoiseCov: 小さいほどモデル(慣性)を重視 -> 遅れるが滑らか
    cv2.setIdentity(kalman.processNoiseCov, 1e-7)
    # measurementNoiseCov: 大きいほど観測(ノイズ)を無視 -> ノイズが消える
    cv2.setIdentity(kalman.measurementNoiseCov, 10)
    
    # 初期化
    kalman.statePost = np.array([data[0][0], data[0][1], 0, 0], np.float32)
    
    filtered = []
    for x, y in data:
        prediction = kalman.predict()
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        estimated = kalman.correct(measurement)
        filtered.append(estimated[:2].flatten())
        
    return np.array(filtered)

def remove_stagnation(data, threshold=0.01):
    if len(data) == 0:
        return data
    clean_data = [data[0]]
    for i in range(1, len(data)):
        # 直前に採用された点との距離を計算
        dist = np.linalg.norm(data[i] - clean_data[-1])
        if dist > threshold:
            clean_data.append(data[i])
    return np.array(clean_data)

# ==========================================
# メイン処理
# ==========================================
def main():
    # 1. データ読み込み
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found. Please run record_gaze.py first.")
        return

    raw_data = df[['raw_x', 'raw_y']].values
    timestamps = df['timestamp'].values
    timestamps = timestamps - timestamps[0] # 開始時間を0にする

    # 2. フィルタ適用
    # 2. 停滞（グチャッとした部分）の削除 / Stagnation Removal
    clean_raw_data = remove_stagnation(raw_data, threshold=0.02)
    # clean_timestamps = ... (タイムスタンプも合わせるべきだが、描画用なら簡易対応でも可)
    # ここでは簡易的にフィルタ後のデータだけを使う（タイムスタンプはずれるので注意）
    
    # 3. フィルタ適用 (Clean dataに対して行う)
    filtered_data = apply_kalman(clean_raw_data)
    
    # visualization用にデータを差し替え
    raw_data = clean_raw_data
    # timestampの長さが合わなくなるので調整 (単に等間隔とみなすか、フィルタするか)
    # 今回はPlot用に配列長を合わせる
    timestamps = timestamps[:len(raw_data)]

    # 3. プロット作成 (出版品質のレイアウト)
    fig = plt.figure(figsize=(10, 8), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1])

    # --- 上段: 2D Spatial Plot (画面上の軌跡) ---
    ax1 = plt.subplot(gs[0])
    
    # Raw Data: 薄いグレーで「ノイズの雲」として表現
    ax1.plot(raw_data[:, 0], raw_data[:, 1], 
             color='#999999', alpha=0.9, linewidth=1, label='Raw Input (No Filter)')
    ax1.scatter(raw_data[:, 0], raw_data[:, 1], 
                color='#999999', s=10, alpha=0.3)

    # Filtered Data: 濃い青ではっきりとした「意図」として表現
    ax1.plot(filtered_data[:, 0], filtered_data[:, 1], 
             color='#0055AA', linewidth=1, alpha=0.9, label='Kalman Filtered')

    # デザイン調整
    ax1.set_title("(A) Spatial Gaze Path", fontsize=14, fontweight='bold', loc='left')
    ax1.set_xlabel("Screen X Coordinate", fontsize=10)
    ax1.set_ylabel("Screen Y Coordinate", fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(1, 0) # 画面座標系に合わせてY軸を反転（上が0）
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True, loc='upper right')

    # --- 下段: Temporal Plot (X軸の動きの時間変化) ---
    ax2 = plt.subplot(gs[1])
    
    # Raw
    ax2.plot(timestamps, raw_data[:, 0], 
             color='#999999', alpha=0.6, linewidth=1, label='Raw X')
    # Filtered
    ax2.plot(timestamps, filtered_data[:, 0], 
             color='#0055AA', linewidth=1, label='Stabilized X')

    # デザイン調整
    ax2.set_title("(B) Temporal Stability (X-axis)", fontsize=14, fontweight='bold', loc='left')
    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.set_ylabel("X Position", fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # 注釈を入れる（効果的！）
    # ※ データに合わせて位置(xy)を調整してください
    mid_time = timestamps[-1] / 2
    ax2.annotate('Jitter Reduction', xy=(mid_time, 0.5), xytext=(mid_time, 0.2),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 ha='center', fontsize=10)

    plt.tight_layout()
    output_filename = 'gaze_comparison_poster.png'
    plt.savefig(output_filename)
    print(f"Saved plot to {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()