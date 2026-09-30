"""Resolve local inputs before Hydra's working-directory change can affect them."""

from pathlib import Path

from omegaconf import OmegaConf


def resolve_runtime_paths(cfg, original_cwd):
    base = Path(original_cwd).resolve()

    def absolute(value):
        path = Path(str(value)).expanduser()
        return str((base / path).resolve())

    # Resolve roots first, so interpolated dataset paths inherit absolute roots.
    keys = [
        "dataset_dir", "csgo_data_dir", "rt1_data_root", "pretrained_models_dir",
        "results_dir", "dataset.loader.root",
        "dataset.loader.validation_fixed_subset_path",
        "experiment.evaluation.metrics.i3d_model_path",
        "experiment.pretrained", "experiment.resume_from_checkpoint",
        "vjepa21_model_path", "vjepa2_repo_path", "latent_codec.repo_path",
    ]
    if OmegaConf.select(cfg, "dataset.name") != "rt1":
        keys += ["dataset.loader.data_path_train", "dataset.loader.data_path_val"]
    for key in keys:
        value = OmegaConf.select(cfg, key)
        if value:
            OmegaConf.update(cfg, key, absolute(value), merge=False)

    # Model locations can be local directories OR Hugging Face repo IDs.
    for key in ("vae_model_path", "webdino_model_path", "latent_codec.model_path", "ckpt_path"):
        value = OmegaConf.select(cfg, key)
        if value and (
            str(value).startswith(("/", "./", "../", "~"))
            or (base / str(value)).exists()
        ):
            OmegaConf.update(cfg, key, absolute(value), merge=False)
    return cfg
