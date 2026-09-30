import os
import argparse
import numpy as np
import pandas as pd
import torch
import imageio
from tqdm import tqdm
from pathlib import Path

# Add project root to path
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from utils.metrics import Evaluator

def load_video(path):
    """Load video using imageio and return as torch tensor [T, C, H, W] in [-1, 1]."""
    reader = imageio.get_reader(path)
    frames = []
    for frame in reader:
        frames.append(frame)
    reader.close()
    video = np.stack(frames)  # [T, H, W, C]
    # Convert to [0, 1] then to [-1, 1]
    video = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
    video = video * 2.0 - 1.0
    return video

def main():
    parser = argparse.ArgumentParser(description="Evaluate rollout videos per frame.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing _gen.mp4 and _gt.mp4 files")
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv", help="Path to save the results CSV")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--history_length", type=int, default=1, help="Number of history frames (context)")
    
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Directory {video_dir} does not exist.")
        return

    # Initialize Evaluator
    # We don't need I3D for frame-by-frame MSE, SSIM, LPIPS
    evaluator = Evaluator(device=args.device)
    
    # Find all sample pairs
    gen_files = sorted(list(video_dir.glob("*_gen.mp4")))
    
    results = []
    
    print(f"Found {len(gen_files)} samples. Starting evaluation...")
    
    for gen_path in tqdm(gen_files):
        sample_id = gen_path.name.replace("_gen.mp4", "").replace("sample_", "")
        gt_path = video_dir / f"sample_{sample_id}_gt.mp4"
        
        if not gt_path.exists():
            print(f"Warning: Ground truth for {gen_path.name} not found. Skipping.")
            continue
            
        # Load videos
        try:
            video_gen = load_video(gen_path).to(args.device)
            video_gt = load_video(gt_path).to(args.device)
        except Exception as e:
            print(f"Error loading {sample_id}: {e}")
            continue
            
        # Ensure they have the same length
        T = min(video_gen.shape[0], video_gt.shape[0])
        
        for t in range(T):
            frame_gen = video_gen[t:t+1] # [1, C, H, W]
            frame_gt = video_gt[t:t+1]   # [1, C, H, W]
            
            with torch.no_grad():
                mse = evaluator.compute_mse(frame_gen, frame_gt)
                psnr = evaluator.compute_psnr(frame_gen, frame_gt)
                ssim = evaluator.compute_ssim(frame_gen, frame_gt)
                lpips_val = evaluator.compute_lpips(frame_gen, frame_gt)
                
            results.append({
                "sample_id": sample_id,
                "frame_idx": t,
                "is_context": t < args.history_length,
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips_val
            })
            
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    
    # Print summary (averaging over samples per frame)
    summary = df.groupby("frame_idx")[["mse", "psnr", "ssim", "lpips"]].mean()
    print("\nSummary (Mean per frame index across all samples):")
    print(summary)

if __name__ == "__main__":
    main()
