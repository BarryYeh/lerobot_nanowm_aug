# Reproducible environment

The supported training runtime is **Linux x86_64, Python 3.11.13, PyTorch
2.7.1 + CUDA 12.8, torchvision 0.22.1, and Lightning 1.9.5**. The same CUDA
wheels support RTX 3090/3090 Ti (Ampere) and RTX 5090 (Blackwell). Use a current
NVIDIA driver; the validated baseline is Linux driver 580.173.02. A local CUDA
toolkit, Conda CUDA packages, and `LD_LIBRARY_PATH` edits are not required.
`torch.compile` additionally needs a host C/C++ compiler (for example `gcc`/`g++`).
The RT-1 extra also builds LeRobot's `evdev` dependency and needs a compiler plus
Linux input headers (Ubuntu's `build-essential` and `linux-libc-dev` packages).
Use glibc 2.28 or newer; CI checks Ubuntu 22.04, and the GPU hosts use Ubuntu
24.04. macOS, Windows without WSL2, and Linux ARM are outside this locked runtime.

Video loading keeps the `decord` API through the pinned
[EVA Decord fork](https://github.com/georgia-tech-db/eva-decord). The original
0.6.0 Linux wheel incorrectly declares a CPython 3.6 ABI inside its metadata,
which makes dependency checkers reject Python 3.11 and triggers repeated
reinstalls. EVA Decord 0.6.1 fixes those wheel tags. PyAV is pinned to 14.2.0
because later 14.x releases do not provide the required Linux Python 3.11 wheel.

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (CI uses
0.12.0), then run from a checkout:

```bash
./nanowm sync
./nanowm doctor --require-cuda
./nanowm python src/main.py experiment=dino_wm_pusht dataset=dino_wm/pusht model=nanowm_b2
```

`nanowm` downloads the pinned Python when necessary, creates an isolated `.venv`,
and uses `uv.lock` without updating it. It also starts commands at the repository
root, so SSH, a new terminal, or a batch job uses the same interpreter and paths.
An activated Conda environment is unnecessary. Existing launchers work through
`./nanowm bash src/scripts/train/main/dino_wm_pusht.sh` (set `GPUS=0` for one GPU).

For RT-1/LeRobot:

```bash
./nanowm sync --extra rt1
./nanowm doctor --require-cuda --rt1
./nanowm python src/main.py experiment=rt1 dataset=rt1/rt1 model=nanowm_b2
```

The RT-1 extra is locked together with core dependencies. Running commands
preserves installed extras; `./nanowm sync` intentionally resets to core, and
`./nanowm sync --extra rt1` restores RT-1. The data source explicitly uses PyAV,
whose wheel bundles FFmpeg libraries. LeRobot also installs TorchCodec, but the
default data path does not load its system-dependent decoder. A custom
`+dataset.loader.video_backend=torchcodec` requires compatible system FFmpeg
shared libraries and is outside the default validation path.

## Configure a machine once

```bash
cp src/configs/local/paths.yaml.example src/configs/local/paths.yaml
# Uncomment dataset_dir / rt1_data_root / results_dir etc. and set your paths.
```

The file is gitignored and automatically loaded by Hydra. Precedence is **CLI
overrides > local YAML > environment variables > repository defaults**. If an
environment variable should control a setting, leave its YAML line commented.
Use `~local` to ignore the machine file for a reproducibility check.

Relative input paths are resolved against the original invocation directory,
before model/data loading, so Hydra's run-directory change cannot redirect them.
With `./nanowm`, that base is always the repository root. Hugging Face repo IDs
remain repo IDs. `.env` is not implicitly loaded; use the local YAML or export
variables in the job environment. Scheduler variables such as `CUDA_VISIBLE_DEVICES`
are preserved. This path configuration applies to the Hydra entrypoint;
standalone sampling scripts retain their documented CLI arguments.

## Verify before changing dependencies

```bash
./nanowm doctor --require-cuda --output results/environment-report.json
./nanowm python -m pytest -q
./nanowm python scripts/smoke_train.py --output results/environment-smoke
# Also exercise the default compilation path (use a new output directory):
./nanowm python scripts/smoke_train.py --compile --output results/environment-smoke-compiled
```

The GPU test runs the actual PushT training entrypoint, writes a checkpoint,
resumes optimizer state from step 1 to step 2, and performs standalone evaluation.
It generates tiny videos and an untrained tiny VAE locally, with Hub access and
W&B disabled. This checks runtime integration, not model quality or published
checkpoint accuracy. `doctor` also runs bf16 CUDA forward/backward kernels and
encodes/decodes video. Its JSON report records versions, GPU architecture, and the
lock hash without dumping environment variables or credentials.

Every PR runs fresh core and RT-1 installations on GitHub-hosted Linux runners,
checks dependency consistency, imports the real training stack, exercises video
decoding and Lightning precision, and runs CPU model/config regression tests.
GPU tests are run on the two hardware generations before accepting a dependency
update. The workflow does not require exposing lab machines to public PR jobs.

To update dependencies, edit `pyproject.toml`, run `uv lock`, and commit both
files. Keep `.python-version` in sync with the validated Python patch version.
Normal users run `sync --locked` via `nanowm`, never `uv lock --upgrade` or an
unbounded `pip install -U`. Dependency maintenance will still be needed as
drivers/hardware evolve; the lock and checks prevent silent drift.

## Migration and scope

`environment.yml` has been retired: two solvers must not own PyTorch. Existing
Conda environments can be left intact; use `./nanowm sync` to create the new
project environment. Do not copy installed site-packages between machines.

This baseline covers training, sampling, evaluation, and optional RT-1 loading.
MuJoCo/PyFlex planning simulators and Depth Anything 3 have separate native stacks;
follow their application guides in separate environments rather than installing
unbounded simulator/xformers packages into the validated training environment.

The original failures are tracked in [PR #16](https://github.com/simchowitzlabpublic/nano-world-model/pull/16)
and [PR #17](https://github.com/simchowitzlabpublic/nano-world-model/pull/17).
The dependency lock incorporates compatible diffusion/Hub/LeRobot versions;
both Trainer call sites use Lightning 1.9's `bf16` mixed-precision spelling.
See [PyTorch's Blackwell/CUDA 12.8 release notes](https://pytorch.org/blog/pytorch-2-7/)
and [uv's PyTorch index guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
