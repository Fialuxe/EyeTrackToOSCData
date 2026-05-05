import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==========================================
# 1. データ生成 (A地点 -> B地点へのサッカード)
# ==========================================
def generate_saccade_data(n_frames=100, switch_frame=30):
    # A地点 (画面左側: 0.2, 0.5) から B地点 (画面右側: 0.8, 0.5) へ
    pos_a = np.array([0.2, 0.5])
    pos_b = np.array([0.8, 0.5])
    
    true_positions = []
    measurements = []
    
    # ノイズの強さ
    noise_std = 0.02  # 画面サイズの2%程度の揺れ
    
    for t in range(n_frames):
        # あるタイミングでAからBへ瞬間移動（サッカード）
        if t < switch_frame:
            true_pos = pos_a
        else:
            true_pos = pos_b
            
        true_positions.append(true_pos)
        
        # ノイズを付加
        noise = np.random.randn(2) * noise_std
        measurements.append(true_pos + noise)
        
    return np.array(true_positions), np.array(measurements)

# ==========================================
# 2. カルマンフィルタ処理
# ==========================================
def run_kalman_filter(measurements):
    # 状態: [x, y, vx, vy] (位置と速度)
    kalman = cv2.KalmanFilter(4, 2)
    
    # 観測行列: 状態から [x, y] だけを見る
    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)
    
    # 遷移行列: 等速運動モデル (x + vx*dt)
    kalman.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    # パラメータ調整（ここが重要）
    # Process Noise: 小さいほどモデル（慣性）を信じる -> 滑らかになる
    cv2.setIdentity(kalman.processNoiseCov, 1e-4)
    # Measurement Noise: 大きいほど観測（ノイズ）を無視する
    cv2.setIdentity(kalman.measurementNoiseCov, 1e-2)
    
    # 初期化
    kalman.statePost = np.zeros((4, 1), dtype=np.float32)
    kalman.statePost[0] = measurements[0][0]
    kalman.statePost[1] = measurements[0][1]
    
    estimated = []
    for meas in measurements:
        kalman.predict()
        est = kalman.correct(meas.reshape(2, 1).astype(np.float32))
        estimated.append(est[:2].flatten())
        
    return np.array(estimated)

# ==========================================
# 3. 描画 (シンプルで見やすく)
# ==========================================
def plot_results(true_data, raw_data, filtered_data):
    # ポスター向けのデザイン（背景暗め、線太め）
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # --- 上段：2D軌跡 (Screen View) ---
    ax1.set_title("Spatial View: Cursor Movement on Screen", fontsize=16, color='white', pad=15)
    
    # ターゲットAとBを描画
    ax1.scatter(true_data[0][0], true_data[0][1], s=300, c='gray', marker='o', label='Start (A)')
    ax1.scatter(true_data[-1][0], true_data[-1][1], s=400, c='gray', marker='*', label='Target (B)')
    
    # Raw Data (薄い青、細い線、透明度あり)
    ax1.plot(raw_data[:, 0], raw_data[:, 1], c='cyan', lw=1.5, alpha=0.4, label='Raw Input (No Filter)')
    ax1.scatter(raw_data[:, 0], raw_data[:, 1], c='cyan', s=10, alpha=0.4)
    
    # Kalman Data (明るい緑、太い線)
    ax1.plot(filtered_data[:, 0], filtered_data[:, 1], c='lime', lw=4, alpha=0.9, label='Kalman Filtered')
    
    # 装飾
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.2, 0.8)
    ax1.set_xlabel("Screen X", fontsize=12)
    ax1.set_ylabel("Screen Y", fontsize=12)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.2)
    
    # --- 下段：時系列 (Time Domain View) ---
    ax2.set_title("Temporal View: Stability & Responsiveness (X-axis)", fontsize=16, color='white', pad=15)
    
    frames = range(len(raw_data))
    
    # 理想的な動き（ステップ関数）
    ax2.plot(frames, true_data[:, 0], 'w--', lw=1, alpha=0.5, label='Ideal Step')
    
    # Raw Data
    ax2.plot(frames, raw_data[:, 0], c='cyan', lw=1, alpha=0.5, label='Raw Jitter')
    
    # Kalman Data
    ax2.plot(frames, filtered_data[:, 0], c='lime', lw=3, label='Stabilized Signal')
    
    # 装飾
    ax2.set_xlim(0, len(raw_data))
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Time (Frames)", fontsize=12)
    ax2.set_ylabel("X Position", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('kalman_simple_comparison.png', dpi=300)
    print("画像が保存されました: kalman_simple_comparison.png")
    # plt.show() # 確認用

if __name__ == "__main__":
    true_path, raw_path = generate_saccade_data()
    filtered_path = run_kalman_filter(raw_path)
    plot_results(true_path, raw_path, filtered_path)