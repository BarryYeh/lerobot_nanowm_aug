"""Exercise both production Trainer constructors on CPU CI, including precision."""

import pytest
import torch
from hydra import compose, initialize_config_dir
from pathlib import Path
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import TensorDataset

import experiments.train_experiment as training


class TinyModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        return self.model(batch[0]).float().square().mean()

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.mark.parametrize("mixed", [True, False])
def test_training_and_evaluation_trainer_api(tmp_path, monkeypatch, mixed):
    configs = Path(__file__).resolve().parents[1] / "src/configs"
    with initialize_config_dir(config_dir=str(configs), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "~local", "experiment.training.max_steps=1", "experiment.training.batch_size=1",
            "experiment.training.val_every_n_steps=1", "experiment.infra.num_workers=0",
            f"experiment.infra.mixed_precision={str(mixed).lower()}", "wandb.enabled=false",
        ])
    experiment = training.TrainExperiment(cfg)
    experiment._seed = 123
    module = TinyModule()
    data = TensorDataset(torch.ones(2, 2))
    monkeypatch.setattr(experiment, "_setup_common", lambda need_train: {
        "experiment_dir": str(tmp_path), "checkpoint_dir": str(tmp_path / "checkpoints"),
        "loggers": [], "train_dataset": data, "val_dataset": data,
        "pl_module": module, "eval_callbacks": [],
    })
    trainers = []

    def cpu_trainer(**kwargs):
        # Only hardware selection changes; production precision/callback/loop args are retained.
        kwargs.update(accelerator="cpu", devices=1, enable_progress_bar=False)
        instance = Trainer(**kwargs)
        trainers.append(instance)
        return instance

    monkeypatch.setattr(training, "Trainer", cpu_trainer)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    experiment.training()
    assert trainers[0].global_step == 1
    cfg.experiment.resume_from_checkpoint = "already-loaded-by-setup.ckpt"
    experiment.evaluate()
    assert len(trainers) == 2
