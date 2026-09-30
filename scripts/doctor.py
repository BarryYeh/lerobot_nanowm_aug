#!/usr/bin/env python3
"""Offline environment check. Does not download weights or access experiment data."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
import tempfile
from pathlib import Path
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def check(require_cuda=False, rt1=False, cuda_dtype="bfloat16"):
    import torch
    import torchvision
    from pytorch_lightning import Trainer

    for module in (
        "diffusers", "transformers", "timm", "lpips", "piqa", "decord", "cv2",
        "av", "h5py", "pandas", "moviepy.editor", "experiments", "wm_datasets",
    ):
        importlib.import_module(module)
    from diffusers import AutoencoderKL
    from transformers import Dinov2Model

    # These constructors catch the exact API incompatibility from PR #17.
    Trainer(accelerator="cpu", devices=1, precision="bf16", logger=False,
            enable_checkpointing=False, enable_progress_bar=False)
    Trainer(accelerator="cpu", devices=1, precision="32", logger=False,
            enable_checkpointing=False, enable_progress_bar=False)

    # Exercise binary video dependencies, rather than just importing them.
    import av
    from decord import VideoReader
    with tempfile.TemporaryDirectory(prefix="nanowm-doctor-") as directory:
        path = str(Path(directory) / "video.mp4")
        frames = torch.randint(0, 256, (4, 32, 32, 3), dtype=torch.uint8)
        torchvision.io.write_video(path, frames, fps=4, video_codec="libx264")
        assert len(VideoReader(path)) == 4
        with av.open(path) as container:
            assert sum(1 for _ in container.decode(video=0)) == 4
        if rt1:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.datasets.video_utils import decode_video_frames
            decoded = decode_video_frames(path, [0.0, 0.25, 0.5], 1e-4, backend="pyav")
            assert decoded.shape == (3, 3, 32, 32)

    cuda = torch.cuda.is_available()
    if require_cuda and not cuda:
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and CUDA_VISIBLE_DEVICES.")
    gpus = []
    if cuda:
        # Run kernels on every visible GPU; an import alone misses missing sm_120 code.
        for index in range(torch.cuda.device_count()):
            device = torch.device("cuda", index)
            with torch.cuda.device(device):
                x = torch.randn(32, 32, device=device, requires_grad=True)
                # The training check defaults to BF16. FP32 inference also runs on T4.
                autocast = torch.autocast("cuda", dtype=torch.bfloat16) if cuda_dtype == "bfloat16" else nullcontext()
                with autocast:
                    loss = (x @ x).float().square().mean()
                loss.backward()
                torch.cuda.synchronize(device)
                assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
            gpus.append({"name": torch.cuda.get_device_name(index),
                         "capability": torch.cuda.get_device_capability(index)})
    return {
        "status": "passed", "python": platform.python_version(),
        "platform": platform.platform(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "gpus": gpus, "rt1": rt1,
        "cuda_test_dtype": cuda_dtype,
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torchvision", "pytorch-lightning", "diffusers", "transformers",
            "huggingface-hub", "numpy", "av", "eva-decord",
        )},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--rt1", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cuda-dtype", choices=["bfloat16", "float32"], default="bfloat16",
                        help="GPU kernel check precision; use float32 for FP32 inference on pre-Ampere GPUs")
    args = parser.parse_args()
    try:
        report = check(args.require_cuda, args.rt1, args.cuda_dtype)
    except Exception:
        print("Environment check failed. Re-run ./nanowm sync (add --extra rt1 for RT-1).\n"
              "See docs/environment.md for the supported driver/platform requirements.",
              file=sys.stderr)
        raise
    payload = json.dumps(report, indent=2) + "\n"
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
