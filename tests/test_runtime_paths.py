from pathlib import Path
import shutil

from hydra import compose, initialize_config_dir

from utils.runtime_paths import resolve_runtime_paths


CONFIGS = Path(__file__).resolve().parents[1] / "src" / "configs"


def test_optional_local_paths_and_cli_precedence(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    shutil.copytree(CONFIGS, configs, ignore=shutil.ignore_patterns("paths.yaml"))
    monkeypatch.setenv("DATASET_DIR", "env-data")
    with initialize_config_dir(config_dir=str(configs), version_base=None):
        assert compose(config_name="config").dataset_dir == "env-data"
    (configs / "local" / "paths.yaml").write_text(
        "# @package _global_\ndataset_dir: local-data\nrt1_data_root: local-rt1\n"
    )
    with initialize_config_dir(config_dir=str(configs), version_base=None):
        cfg = compose(config_name="config", overrides=["dataset=rt1/rt1"])
        assert cfg.dataset_dir == "local-data"
        assert cfg.dataset.loader.root == "local-rt1"
        cfg = compose(config_name="config", overrides=["dataset_dir=cli-data"])
        assert cfg.dataset_dir == "cli-data"


def test_paths_survive_hydra_chdir_and_hub_ids_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_DIR", "relative-data")
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "dataset=dino_wm/pusht", "~local", "vae_model_path=./weights/vae",
            "experiment.pretrained=./checkpoints/model.pt",
            "ckpt_path=knightnemo/nanowm-b2-dino-wm-point-maze-30k",
        ])
    run_dir = tmp_path / "results" / "run"
    run_dir.mkdir(parents=True)
    monkeypatch.chdir(run_dir)
    resolve_runtime_paths(cfg, tmp_path)
    assert cfg.dataset.loader.data_path_train == str(tmp_path / "relative-data/pusht_noise/train")
    assert cfg.vae_model_path == str(tmp_path / "weights/vae")
    assert cfg.experiment.pretrained == str(tmp_path / "checkpoints/model.pt")
    assert cfg.webdino_model_path == "facebook/webssl-dino300m-full2b-224"
    assert cfg.ckpt_path == "knightnemo/nanowm-b2-dino-wm-point-maze-30k"
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        rt1 = compose(config_name="config", overrides=["dataset=rt1/rt1", "~local"])
    repo_id = rt1.dataset.loader.data_path
    resolve_runtime_paths(rt1, tmp_path)
    assert rt1.dataset.loader.data_path == repo_id
    assert Path(rt1.dataset.loader.root).is_absolute()
