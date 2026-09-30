import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

def plot_metrics(csv_files, labels, output_path):
    # Set style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = [
        ('mse', 'MSE (↓)', False),
        ('psnr', 'PSNR (↑)', True),
        ('ssim', 'SSIM (↑)', True),
        ('lpips', 'LPIPS (↓)', False)
    ]
    
    colors = sns.color_palette("husl", len(csv_files))
    
    for i, (csv_file, label) in enumerate(zip(csv_files, labels)):
        df = pd.read_csv(csv_file)
        
        # Determine history length from the first sample
        # history_length is the count of is_context == True for a single sample
        first_sample_id = df['sample_id'].iloc[0]
        h_len = df[(df['sample_id'] == first_sample_id) & (df['is_context'] == True)].shape[0]
        
        # Calculate steps after context: relative_idx = 0 is the first GENERATED frame
        df['pred_step'] = df['frame_idx'] - h_len
        
        # Filter to keep only generated frames (where pred_step >= 0)
        gen_df = df[df['pred_step'] >= 0].copy()
        
        # Group by pred_step and average across samples
        summary = gen_df.groupby('pred_step').mean().reset_index()
        
        for j, (col, title, higher_better) in enumerate(metrics):
            ax = axes[j]
            ax.plot(summary['pred_step'], summary[col], marker='o', label=label, color=colors[i], linewidth=2)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Steps After Context (0 = First Generated Frame)', fontsize=12)
            if i == 0:
                ax.set_ylabel('Value', fontsize=12)

    # Add legends and cleanup
    for ax in axes:
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot metrics from multiple rollout evaluation CSVs.")
    parser.add_argument("--csvs", type=str, nargs='+', required=True, 
                        help="Paths to CSV files (e.g., metrics_h1.csv metrics_h2.csv ...)")
    parser.add_argument("--output", type=str, default="rollout_comparison.png", 
                        help="Path to save the resulting plot")
    
    args = parser.parse_args()
    
    # Generate labels from filenames if possible
    labels = []
    for csv in args.csvs:
        name = Path(csv).stem
        if 'metrics_h' in name:
            h_val = name.split('_h')[-1]
            labels.append(f"History Length: {h_val}")
        else:
            labels.append(name)
            
    plot_metrics(args.csvs, labels, args.output)

if __name__ == "__main__":
    main()
