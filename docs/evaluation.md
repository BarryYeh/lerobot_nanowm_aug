# Evaluation

How to evaluate a trained world model: standalone metric runs, sampling pipelines, and the headline results we report on each domain.

## Setup

Same as training — the eval path uses the same dependencies:

```bash
./nanowm sync
```

i3d weights for FVD (one-time; FID uses InceptionV3, which downloads itself on first use):

```bash
mkdir -p pretrained_models/i3d
curl -L "https://www.dropbox.com/scl/fi/c5nfs6c422nlpj880jbmh/i3d_torchscript.pt?rlkey=x5xcjsrz0818i4qxyoglp5bb8&dl=1" \
    -o pretrained_models/i3d/i3d_torchscript.pt
```

## Quick start

### Evaluate a checkpoint on a fixed validation subset

```bash
./nanowm python src/main.py experiment=evaluate_only \
    dataset=dino_wm/pusht model=nanowm_b2 \
    experiment.resume_from_checkpoint=<path/to/ckpt> \
    experiment.evaluation.batch_size=1 \
    dataset.loader.validation_fixed_subset_size=256 \
    dataset.loader.validation_fixed_subset_seed=42
```

We provide example scripts under `src/scripts/eval/`, including
`dino_wm_*.sh`, `rt1.sh`, `csgo.sh`, and ablation evals under `abl_*.sh`.

`experiment=evaluate_only` sets `tasks=[evaluate]` and `validation_size: null` (use the full val set, optionally constrained by `validation_fixed_subset_size`). Outputs land under `${RESULTS_DIR}/<run_dir>/`:

```
<run_dir>/
├── eval_videos/step_<step>/<sample>/   # pred_video.mp4, gt_video.mp4, raw_data.npz
├── tb/                                 # val_eval/{psnr,ssim,lpips,fid,fvd} scalars
└── .hydra/                             # composed config snapshot
```

Metric values are not written to a file. They are logged as `val_eval/*` to TensorBoard (and to W&B when enabled) and printed to the run log as `Epoch end val <metric>: <value>`.

### Standalone metric calculation

If you already have rollout videos, compute metrics directly:

```bash
./nanowm python src/sample/evaluate_metrics.py \
    --video_dir /path/to/rollout_results \
    --history_length 1 \
    --output_csv metrics.csv
```

Plot a comparison across runs (e.g., different history lengths):

```bash
./nanowm python src/sample/plot_metrics.py \
    --csvs metrics_h1.csv metrics_h2.csv metrics_h3.csv \
    --output rollout_comparison.png
```

## Sampling modes

The model supports two scheduling modes during sampling:

<div align="center">

| `model.scheduling_mode` | Behavior |
|:------------------------|:---------|
| `sequential` (default) | Frame-by-frame autoregressive denoising. Highest quality. |
| `full_sequence` | Denoise all frames jointly (DDIM over the whole clip). Faster, slightly lower quality. |

</div>

DDIM steps are controlled by `model.num_sampling_steps` (250 default for sequential; 50 is a sensible setting for full_sequence). Switch modes via:

```bash
./nanowm python src/main.py experiment=evaluate_only ... model.scheduling_mode=full_sequence \
    model.num_sampling_steps=50
```

## Metric definitions

Metrics are computed on predicted frames only; context frames are excluded.

<div align="center">

| Metric | Direction | Notes |
|:-------|:----------|:------|
| **PSNR** | ↑ | per-pixel MSE, in dB |
| **SSIM** | ↑ | structural similarity, [0, 1] |
| **LPIPS** | ↓ | learned perceptual distance (AlexNet) |
| **FID** | ↓ | Fréchet Inception Distance, per-frame InceptionV3 pool3 features (`pytorch-fid`) |

</div>

PSNR/SSIM/LPIPS are per-clip averages. FID is *not*: predicted frames from every clip are pooled into a single set of InceptionV3 features and one Fréchet distance is computed over the pool (256 clips × 3 predicted frames = 768 frames for the 4-frame configs). That is far below the sample count FID is usually reported at, so these values are comparable across rows of this table but not against FID numbers from other papers.

**FVD** is computed only when a clip has at least 10 predicted frames. It therefore runs on the 16-frame CSGO configs and is skipped for the 4-frame DINO-WM and RT-1 configs below, which is why those rows report no FVD. The i3d model path (FVD only) comes from `${PRETRAINED_MODELS_DIR}/i3d/i3d_torchscript.pt` or the relative fallback `pretrained_models/i3d/i3d_torchscript.pt`.

## Results on shipped checkpoints

Standardized eval: 256 fixed val samples (seed 42), 250 DDIM steps, sequential scheduling.

### DINO-WM environments (NanoWM-B/2)

<div align="center">

| Dataset | Steps | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|:--------|:------|:-------|:-------|:--------|:------|
| Point Maze | 30k | 36.74 | 0.984 | 0.019 | 9.66 |
| Wall | 15k | 34.05 | 0.994 | 0.010 | 2.64 |
| PushT | 100k | 33.19 | 0.982 | 0.016 | 13.63 |
| Rope | 15k | 31.63 | 0.953 | 0.056 | 35.20 |
| Granular | 15k | 26.08 | 0.917 | 0.073 | 40.05 |

</div>

### RT-1 (fractal)

<div align="center">

| Model | Steps | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|:------|:------|:-------|:-------|:--------|:------|
| NanoWM-B/2 | 300k | 24.36 | 0.787 | 0.180 | 35.08 |

</div>

## Reproducing these numbers

Each row is a single eval command. Replace `<ckpt>` with the corresponding HF model checkpoint path (after downloading) and `<dataset>` with the matching `dino_wm/{point_maze,wall,pusht,rope,granular}` or `rt1/rt1`.

```bash
./nanowm python src/main.py experiment=evaluate_only \
    dataset=<dataset> model=nanowm_b2 \
    experiment.resume_from_checkpoint=<ckpt> \
    experiment.evaluation.batch_size=1 \
    dataset.loader.validation_fixed_subset_size=256 \
    dataset.loader.validation_fixed_subset_seed=42
```

For ablation arms (per-axis comparison), see [training.md](training.md#design-choices).

## See also

- [training.md](training.md) — design choices + ablation tables
- [applications/long_rollout.md](applications/long_rollout.md) — long-horizon autoregressive rollout (CSGO 50-frame demo)
- [applications/planning.md](applications/planning.md) — model-predictive control (success-rate metric)
- [config_system.md](config_system.md) — Hydra config reference
