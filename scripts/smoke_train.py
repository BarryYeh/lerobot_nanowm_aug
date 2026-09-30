#!/usr/bin/env python3
"""Run the real PushT train/resume/eval entrypoints on tiny synthetic inputs.

No dataset, pretrained weights, tracking account, or network access is needed.
The randomly initialized tiny VAE tests integration, not prediction quality.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import torchvision
from diffusers import AutoencoderKL

ROOT = Path(__file__).resolve().parents[1]


def run_smoke(directory, compile_model=False):
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a CUDA GPU. Use doctor + pytest on CPU CI.")
    torch.manual_seed(123)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for split, count in (("train", 2), ("val", 1)):
        data = directory / "data" / "pusht_noise" / split
        (data / "obses").mkdir(parents=True)
        torch.save(torch.randn(count, 6, 5), data / "states.pth")
        torch.save(torch.randn(count, 6, 2), data / "rel_actions.pth")
        torch.save(torch.full((count,), 6), data / "seq_lengths.pth")
        for index in range(count):
            torchvision.io.write_video(
                str(data / "obses" / f"episode_{index:03d}.mp4"),
                torch.randint(0, 256, (6, 32, 32, 3), dtype=torch.uint8),
                fps=5, video_codec="libx264",
            )
    vae_path = directory / "vae"
    AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(32, 32, 32, 32), layers_per_block=1,
        latent_channels=4, norm_num_groups=32, sample_size=32,
    ).save_pretrained(vae_path)

    # Deliberately pass relative paths: Hydra changes cwd before creating datasets.
    common = [
        "~local", "experiment=dino_wm_pusht", "dataset=dino_wm/pusht",
        "model=nanowm_s2", "model.image_size=32", "model.latent_size=4",
        "model.num_frames=2", "dataset.frame_interval=1",
        f"dataset_dir={json.dumps(os.path.relpath(directory / 'data', ROOT))}",
        f"vae_model_path={json.dumps(os.path.relpath(vae_path, ROOT))}",
        "experiment.training.max_steps=1", "experiment.training.batch_size=1",
        "experiment.training.log_every=1", "experiment.training.val_every_n_steps=1",
        "experiment.training.checkpointing.latest.every_n_train_steps=1",
        "experiment.infra.num_workers=0",
        f"experiment.infra.compile={str(compile_model).lower()}",
        "experiment.infra.mixed_precision=true",
        "dataset.loader.validation_size=1",
        "experiment.evaluation.metrics.evaluate=false",
        "experiment.evaluation.save_videos=false", "wandb.enabled=false",
    ]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled")
    # Bound CPU work during compilation on shared hosts.
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

    def launch(name, overrides):
        command = [sys.executable, str(ROOT / "src/main.py"), *common,
                   f"hydra.run.dir={json.dumps(str(directory / name))}", *overrides]
        result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
        (directory / f"{name}.log").write_text(result.stdout)
        print(result.stdout, flush=True)
        result.check_returncode()

    launch("train", [])
    checkpoint = next((directory / "train/checkpoints/latest").glob("*.ckpt"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["global_step"] == 1 and state["optimizer_states"]
    assert all(torch.isfinite(t).all() for t in state["state_dict"].values() if torch.is_tensor(t))
    launch("resume", ["experiment.training.max_steps=2",
                      f"experiment.resume_from_checkpoint={json.dumps(str(checkpoint))}"])
    resumed = next((directory / "resume/checkpoints/latest").glob("*.ckpt"))
    assert torch.load(resumed, map_location="cpu", weights_only=False)["global_step"] == 2
    launch("eval", ["experiment.tasks=[evaluate]",
                    f"experiment.resume_from_checkpoint={json.dumps(str(resumed))}"])
    report = {"status": "passed", "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "compile": compile_model,
              "train_steps": 1, "resumed_step": 2, "evaluation": "passed"}
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory to retain logs and checkpoints")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    args = parser.parse_args()
    if args.output:
        if args.output.exists():
            parser.error("--output must be a new directory, so prior evidence cannot be overwritten")
        run_smoke(args.output, args.compile_model)
    else:
        with tempfile.TemporaryDirectory(prefix="nanowm-smoke-") as temporary:
            run_smoke(temporary, args.compile_model)
