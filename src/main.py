import signal
import os
os.environ["PL_DISABLE_SUBPROCESS_LOGGING"] = "1"
import hydra
import torch
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from utils.runtime_paths import resolve_runtime_paths

from experiments import build_experiment
def signal_handler(sig, frame):
    print(f"\nInterrupt received. Force quitting.")
    os._exit(0)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    resolve_runtime_paths(cfg, get_original_cwd())
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    torch.set_float32_matmul_precision('high')
    experiment = build_experiment(cfg)
    experiment.exec()


if __name__ == "__main__":
    main()
