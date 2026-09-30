"""Regression tests for planning rollouts.

The planning module imports the optional model/codec stacks at module import
time.  These tests load it with tiny stand-ins so the rollout bookkeeping can
be exercised without downloading a checkpoint or starting a simulator.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "planning" / "diffusion_world_model.py"
)


def _load_module(monkeypatch):
    """Load the planning module while replacing optional dependencies."""
    diffusion_pkg = types.ModuleType("diffusion")
    diffusion_pkg.__path__ = []
    diffusion_sample = types.ModuleType("diffusion.df_sample")
    diffusion_sample.dfot_sample = lambda **_: None
    diffusion_gaussian = types.ModuleType("diffusion.gaussian_diffusion")
    diffusion_gaussian.GaussianDiffusion = object
    latent_codecs = types.ModuleType("latent_codecs")
    latent_codecs.LatentCodec = object

    for name, module in {
        "diffusion": diffusion_pkg,
        "diffusion.df_sample": diffusion_sample,
        "diffusion.gaussian_diffusion": diffusion_gaussian,
        "latent_codecs": latent_codecs,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("planning_rollout_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class _IdentityCodec:
    precision = "fp32"

    @staticmethod
    def encode(frames):
        return frames


def _args(num_frames=4, n_context_frames=2):
    return SimpleNamespace(
        model=SimpleNamespace(
            num_sampling_steps=2,
            num_frames=num_frames,
            n_context_frames=n_context_frames,
            scheduling_mode="sequential",
        ),
        experiment=SimpleNamespace(
            diffusion=SimpleNamespace(history_stabilization_level=0.0),
            infra=SimpleNamespace(vae_precision="fp32"),
        ),
    )


def test_rollout_preserves_history_and_action_alignment(monkeypatch):
    module = _load_module(monkeypatch)
    calls = []

    def fake_dfot_sample(**kwargs):
        call_index = len(calls)
        calls.append(
            {
                "context": kwargs["context"].clone(),
                "action": kwargs["model_kwargs"]["action"].clone(),
            }
        )
        batch, frames, channels, height, width = kwargs["shape"]
        context = kwargs["context"]
        output = torch.zeros(batch, frames, channels, height, width)
        output[:, :2] = context
        output[:, 2:] = 10 * (call_index + 1) + torch.arange(
            frames - 2, dtype=output.dtype
        ).view(1, -1, 1, 1, 1)
        return output

    monkeypatch.setattr(module, "dfot_sample", fake_dfot_sample)
    world_model = module.DiffusionWorldModel(
        _TinyModel(), _IdentityCodec(), object(), _args()
    )

    initial = torch.full((1, 1, 1, 1, 1), 9.0)
    actions = torch.arange(1, 6, dtype=torch.float32).view(1, 5, 1)
    observations, _ = world_model.rollout({"visual": initial}, actions)

    assert [tuple(call["context"].shape) for call in calls] == [
        (1, 2, 1, 1, 1),
        (1, 2, 1, 1, 1),
        (1, 2, 1, 1, 1),
    ]
    assert [call["context"].flatten().tolist() for call in calls] == [
        [9.0, 9.0],
        [10.0, 11.0],
        [20.0, 21.0],
    ]
    assert [call["action"].flatten().tolist() for call in calls] == [
        [0.0, 1.0, 2.0, 0.0],
        [0.0, 3.0, 4.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
    ]
    assert observations["visual"].shape == (1, 6, 1)
    assert observations["visual"].flatten().tolist() == [
        9.0,
        10.0,
        11.0,
        20.0,
        21.0,
        30.0,
    ]


def test_rollout_left_pads_partial_initial_history(monkeypatch):
    module = _load_module(monkeypatch)
    calls = []

    def fake_dfot_sample(**kwargs):
        calls.append(kwargs["context"].clone())
        batch, frames, channels, height, width = kwargs["shape"]
        return torch.zeros(batch, frames, channels, height, width)

    monkeypatch.setattr(module, "dfot_sample", fake_dfot_sample)
    world_model = module.DiffusionWorldModel(
        _TinyModel(), _IdentityCodec(), object(), _args(num_frames=4, n_context_frames=3)
    )

    # Explicitly make the temporal dimension unambiguous: [B=1, T=2, C=1, H=1, W=1].
    initial = torch.tensor([1.0, 2.0]).view(1, 2, 1, 1, 1)
    actions = torch.tensor([[[7.0]]])
    world_model.rollout({"visual": initial}, actions)

    assert calls[0].flatten().tolist() == [1.0, 1.0, 2.0]


def test_rollout_rejects_non_generating_context_window(monkeypatch):
    module = _load_module(monkeypatch)
    world_model = module.DiffusionWorldModel(
        _TinyModel(), _IdentityCodec(), object(), _args(num_frames=2, n_context_frames=2)
    )

    initial = torch.zeros(1, 1, 1, 1, 1)
    actions = torch.zeros(1, 1, 1)
    with pytest.raises(ValueError, match="n_context_frames"):
        world_model.rollout({"visual": initial}, actions)
