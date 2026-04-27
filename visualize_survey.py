import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_metrics(df):
    """Calculate Jitter and RMSE"""
    # Jitter: Velocity variance
    dx = df['gaze_x'].diff().dropna()
    dy = df['gaze_y'].diff().dropna()
    velocity = np.sqrt(dx**2 + dy**2)
    jitter = velocity.std()
    
    # RMSE: Error from Target (if target exists)
    if 'target_x' in df.columns and not df['target_x'].isnull().all():
        err_x = df['gaze_x'] - df['target_x']
        err_y = df['gaze_y'] - df['target_y']
        rmse = np.sqrt((err_x**2 + err_y**2).mean())
    else:
        rmse = np.nan
        
    return jitter, rmse

def visualize_survey():
    parser = argparse.ArgumentParser(description="Visualize and compare gaze survey data.")
    parser.add_argument("files", nargs='+', help="List of CSV files to compare")
    args = parser.parse_args()

    plt.figure(figsize=(12, 5))
    
    # CHI Standard Style (Academic/Publication Ready)
    # White background, readable fonts, no grid clutter
    params = {
        'axes.labelsize': 12,
        'font.size': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'font.family': 'sans-serif', # Arial or Helvetica is standard for CHI
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'grid.color': 'gray'
    }
    plt.rcParams.update(params)

    # Subplot 1: Time vs Position (X-axis tracking)
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Gaze Tracking: Time vs X Position", fontweight='bold')
    ax1.set_xlabel("Time (frames)")
    ax1.set_ylabel("X Position (0.0 - 1.0)")
    ax1.set_ylim(0, 1)

    # Subplot 2: Jitter/Error Profile
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Jitter (Velocity) Profile", fontweight='bold')
    ax2.set_xlabel("Time (frames)")
    ax2.set_ylabel("Frame-to-Frame Jump")

    # Colors suitable for white background (High contrast)
    colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7'] # Colorblind friendly palette
    
    target_plotted = False

    print(f"{'File':<30} | {'Mode':<10} | {'Jitter (Std)':<12} | {'Accuracy (RMSE)':<15}")
    print("-" * 75)

    for i, file_path in enumerate(args.files):
        try:
            df = pd.read_csv(file_path)
            
            # --- Discard first 30 frames (Warmup/Calibration noise) ---
            if len(df) > 30:
                df = df.iloc[30:].reset_index(drop=True)
            
            # Helper to handle empty strings/nans if any
            df['gaze_x'] = pd.to_numeric(df['gaze_x'], errors='coerce')
            df['gaze_y'] = pd.to_numeric(df['gaze_y'], errors='coerce')
            df['target_x'] = pd.to_numeric(df['target_x'], errors='coerce')
            
            # Drop rows with missing gaze data
            df = df.dropna(subset=['gaze_x', 'gaze_y'])
            
            label = f"Session {i+1}"
            mode = "Unknown"
            if 'mode' in df.columns:
                mode = df['mode'].iloc[0]
                label = f"{mode.upper()}"

            jitter, rmse = calculate_metrics(df)
            
            print(f"{file_path:<30} | {mode:<10} | {jitter:.6f}     | {rmse:.6f}")

            color = colors[i % len(colors)]
            
            # Plot Target Path (Dashed) - Only once
            if not target_plotted and 'target_x' in df.columns:
                ax1.plot(range(len(df)), df['target_x'], color='black', linestyle='--', linewidth=2, alpha=0.6, label='Target (Ideal)')
                target_plotted = True

            # Plot Gaze Path
            ax1.plot(range(len(df)), df['gaze_x'], label=label, color=color, linewidth=2, alpha=0.9)

            # Plot Velocity (Jitter)
            dx = df['gaze_x'].diff().fillna(0)
            dy = df['gaze_y'].diff().fillna(0)
            velocity = np.sqrt(dx**2 + dy**2)
            velocity_smooth = velocity.rolling(window=5).mean()
            
            ax2.plot(range(len(df)), velocity_smooth, label=label, color=color, linewidth=1, alpha=0.8)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    ax1.legend()
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("survey_comparison.png")
    print("\nComparison plot saved to 'survey_comparison.png'")
    plt.show()

if __name__ == "__main__":
    visualize_survey()
