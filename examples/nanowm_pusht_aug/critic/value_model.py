"""Value (V) / reward-model (RM) network on NanoWM SD-VAE latents, top-wall band.

Plan: value-model-opus5-5-sonnet5-floofy-engelbart.md, section "Model". ONLY torch + numpy imports:
this file is loaded both in the lerobot training env and in the nanowm env for the later search, and
the nanowm env pins an older torch (2.1.2). Do not add `torch.compile`, newer `torch.func` APIs, or
any lerobot import here.

Architecture (fixed by the plan; do not change without Opus sign-off):
  ValueNet   latent (B,3,4,32,32) -> reshape (B,12,32,32)
             Conv(12->32,k3,p1) SiLU -> Conv(32->64,k3,s2,p1) SiLU -> Conv(64->64,k3,s2,p1) SiLU
             -> global avg pool -> Linear(64->128)
             concat normalized agent_state (4) -> Linear(132->256) SiLU -> Linear(256->256) SiLU
             heads: V (1 logit), RM (1 logit), pose (4: cx/512, cy/512, sin theta, cos theta)
  OracleNet  same trunk/heads minus the conv encoder; input = [agent_state_norm(4), cog_x/512,
             cog_y/512, sin(theta), cos(theta)] (8-dim)
  Normalizer per-VAE-channel mean/std (4 values, broadcast over the 3 stacked frames) + agent-state
             stats (pos/512; vel/train-std), fit on the TRAIN split only, serializable with the
             ensemble checkpoint.
  ValueEnsemble / OracleEnsemble
             K members + a shared Normalizer. `score(...)` returns the SAILOR-style mean-of-2-random
             (search-time) and mean/std-over-all-K (deterministic, used for every gate metric).

state convention: `agent_state` = [x, y, vx, vy] (sim units, pixel-scale positions). `cog_state` =
[agent_x, agent_y, cog_x, cog_y, theta] (see eval/ood_failure_map.py:sim_state), used only for
the pose auxiliary target and the OracleNet input.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

POS_SCALE = 512.0  # PushT sim coordinates are in [0, 512]


def pose_target(cog_state: np.ndarray) -> np.ndarray:
    """cog_state (...,5) [agent_x,agent_y,cog_x,cog_y,theta] -> pose target (...,4)
    [cog_x/512, cog_y/512, sin(theta), cos(theta)], the ValueNet pose head's regression target."""
    cog_state = np.asarray(cog_state, np.float64)
    cx, cy, th = cog_state[..., 2], cog_state[..., 3], cog_state[..., 4]
    return np.stack([cx / POS_SCALE, cy / POS_SCALE, np.sin(th), np.cos(th)], axis=-1).astype(np.float32)


def _torch_load(path, map_location):
    """`weights_only=False` is our own trusted checkpoint format (plain dicts/tensors/lists); older
    torch (nanowm env, 2.1.2) does not have the kwarg at all, so fall back if it's missing."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)  # nosec B614
    except TypeError:
        return torch.load(path, map_location=map_location)  # nosec B614


class Normalizer:
    """Per-channel latent mean/std (4,) and agent-state pos-scale/velocity-std (2,), fit on train
    rows only. `norm_latent` broadcasts the 4 VAE channels over the 3 stacked frames."""

    def __init__(self) -> None:
        self.lat_mean = torch.zeros(4)
        self.lat_std = torch.ones(4)
        self.vel_std = torch.ones(2)

    def fit(self, latents: np.ndarray, agent_states: np.ndarray) -> None:
        """latents (N,3,4,32,32) any float dtype; agent_states (N,4) [x,y,vx,vy]."""
        lat = np.asarray(latents, np.float32)
        # (N,3,4,32,32) -> channel axis last -> (-1, 4)
        c = np.moveaxis(lat, 2, -1).reshape(-1, 4)
        self.lat_mean = torch.as_tensor(c.mean(0), dtype=torch.float32)
        self.lat_std = torch.as_tensor(c.std(0) + 1e-6, dtype=torch.float32)
        vel = np.asarray(agent_states, np.float64)[:, 2:4].astype(np.float32)
        self.vel_std = torch.as_tensor(vel.std(0) + 1e-6, dtype=torch.float32)

    def norm_latent(self, latent_stack: torch.Tensor) -> torch.Tensor:
        """(...,3,4,32,32) -> same shape, per-channel normalized."""
        mean = self.lat_mean.to(latent_stack.device, latent_stack.dtype).view(1, 1, 4, 1, 1)
        std = self.lat_std.to(latent_stack.device, latent_stack.dtype).view(1, 1, 4, 1, 1)
        return (latent_stack - mean) / std

    def norm_agent_state(self, agent_state: torch.Tensor) -> torch.Tensor:
        """(...,4) [x,y,vx,vy] -> (...,4) [x/512, y/512, vx/std, vy/std]."""
        pos = agent_state[..., 0:2] / POS_SCALE
        vel_std = self.vel_std.to(agent_state.device, agent_state.dtype)
        vel = agent_state[..., 2:4] / vel_std
        return torch.cat([pos, vel], dim=-1)

    def oracle_feat(self, agent_state: torch.Tensor, cog_state: torch.Tensor) -> torch.Tensor:
        """agent_state (...,4) [x,y,vx,vy], cog_state (...,5) [agent_x,agent_y,cog_x,cog_y,theta]
        -> OracleNet input (...,8) = [agent_state_norm(4), cog_x/512, cog_y/512, sin th, cos th]."""
        agent_norm = self.norm_agent_state(agent_state)
        cx = cog_state[..., 2:3] / POS_SCALE
        cy = cog_state[..., 3:4] / POS_SCALE
        th = cog_state[..., 4:5]
        return torch.cat([agent_norm, cx, cy, torch.sin(th), torch.cos(th)], dim=-1).to(torch.float32)

    def state_dict(self) -> dict:
        return {"lat_mean": self.lat_mean, "lat_std": self.lat_std, "vel_std": self.vel_std}

    def load_state_dict(self, d: dict) -> None:
        self.lat_mean = torch.as_tensor(d["lat_mean"], dtype=torch.float32)
        self.lat_std = torch.as_tensor(d["lat_std"], dtype=torch.float32)
        self.vel_std = torch.as_tensor(d["vel_std"], dtype=torch.float32)


class TrunkHead(nn.Module):
    """Shared MLP trunk (in_dim -> hidden -> hidden, SiLU) + the three heads (V, RM, pose). Used by
    both ValueNet (in_dim = feat + 4) and OracleNet (in_dim = 8, "same trunk/heads minus the conv
    encoder")."""

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.v_head = nn.Linear(hidden, 1)
        self.rm_head = nn.Linear(hidden, 1)
        self.pose_head = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.silu(self.fc1(x))
        h = F.silu(self.fc2(h))
        return self.v_head(h).squeeze(-1), self.rm_head(h).squeeze(-1), self.pose_head(h)


class ValueNet(nn.Module):
    def __init__(self, conv: tuple[int, int, int] = (32, 64, 64), feat: int = 128, hidden: int = 256):
        super().__init__()
        self.conv, self.feat, self.hidden = tuple(conv), feat, hidden
        c1, c2, c3 = conv
        self.enc1 = nn.Conv2d(12, c1, 3, padding=1)
        self.enc2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.enc_fc = nn.Linear(c3, feat)
        self.trunk = TrunkHead(feat + 4, hidden)

    def encode(self, latent: torch.Tensor) -> torch.Tensor:
        b = latent.shape[0]
        x = latent.reshape(b, 12, latent.shape[-2], latent.shape[-1])
        x = F.silu(self.enc1(x))
        x = F.silu(self.enc2(x))
        x = F.silu(self.enc3(x))
        x = x.mean(dim=(2, 3))
        return self.enc_fc(x)

    def forward(
        self, latent_norm: torch.Tensor, agent_state_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(latent_norm)
        return self.trunk(torch.cat([z, agent_state_norm], dim=-1))

    @property
    def kwargs(self) -> dict:
        return {"conv": self.conv, "feat": self.feat, "hidden": self.hidden}


class OracleNet(nn.Module):
    """State-oracle baseline: the true state replaces the latent encoder entirely."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.hidden = hidden
        self.trunk = TrunkHead(8, hidden)

    def forward(self, oracle_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.trunk(oracle_feat)

    @property
    def kwargs(self) -> dict:
        return {"hidden": self.hidden}


def _random_pair(k: int, rng: np.random.Generator | None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.choice(k, size=min(2, k), replace=False)


class ValueEnsemble:
    """K ValueNet members + a shared Normalizer. `score` is the only inference entry point the
    later search should use."""

    net_cls = ValueNet

    def __init__(self, members: list[ValueNet], normalizer: Normalizer):
        assert members, "empty ensemble"
        self.members = members
        self.normalizer = normalizer

    def to(self, device) -> ValueEnsemble:
        for m in self.members:
            m.to(device)
        return self

    def eval(self) -> ValueEnsemble:
        for m in self.members:
            m.eval()
        return self

    @torch.no_grad()
    def _forward_all(self, latent_stack: torch.Tensor, agent_state: torch.Tensor):
        lat_n = self.normalizer.norm_latent(latent_stack)
        st_n = self.normalizer.norm_agent_state(agent_state)
        vs, rs, ps = [], [], []
        for m in self.members:
            v, r, p = m(lat_n, st_n)
            vs.append(v)
            rs.append(r)
            ps.append(p)
        return torch.stack(vs), torch.stack(rs), torch.stack(ps)

    @torch.no_grad()
    def score(
        self, latent_stack: torch.Tensor, agent_state: torch.Tensor, rng: np.random.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        """latent_stack (B,3,4,32,32), agent_state (B,4) -> dict of (B,) tensors:
        v_mean2/rm_mean2 (mean of 2 randomly chosen members' sigmoid output, the search-time
        convention), v_mean_all/rm_mean_all (mean over all K, deterministic, used for every gate
        metric), v_std/rm_std (std over all K, the ensemble-disagreement signal)."""
        v_logit, rm_logit, _ = self._forward_all(latent_stack, agent_state)
        v_sig, rm_sig = torch.sigmoid(v_logit), torch.sigmoid(rm_logit)
        idx2 = _random_pair(len(self.members), rng)
        return {
            "v_mean2": v_sig[idx2].mean(0),
            "v_mean_all": v_sig.mean(0),
            "v_std": v_sig.std(0),
            "rm_mean2": rm_sig[idx2].mean(0),
            "rm_mean_all": rm_sig.mean(0),
            "rm_std": rm_sig.std(0),
        }

    def save(self, path) -> None:
        torch.save(
            {
                "net_type": "ValueNet",
                "model_kwargs": self.members[0].kwargs,
                "normalizer": self.normalizer.state_dict(),
                "members": [m.state_dict() for m in self.members],
            },
            path,
        )

    @classmethod
    def load(cls, path, device: str = "cpu") -> ValueEnsemble:
        d = _torch_load(path, map_location=device)
        assert d["net_type"] == "ValueNet", d["net_type"]
        members = []
        for sd in d["members"]:
            m = ValueNet(**d["model_kwargs"])
            m.load_state_dict(sd)
            m.to(device).eval()
            members.append(m)
        norm = Normalizer()
        norm.load_state_dict(d["normalizer"])
        return cls(members, norm)


class OracleEnsemble:
    """K OracleNet members + a shared Normalizer (same normalizer stats as the latent ensemble's
    would-be Normalizer -- fit independently here on the same train split). Mirrors ValueEnsemble's
    API for the state-oracle baseline; not needed by the nanowm-side search."""

    net_cls = OracleNet

    def __init__(self, members: list[OracleNet], normalizer: Normalizer):
        assert members, "empty ensemble"
        self.members = members
        self.normalizer = normalizer

    def to(self, device) -> OracleEnsemble:
        for m in self.members:
            m.to(device)
        return self

    def eval(self) -> OracleEnsemble:
        for m in self.members:
            m.eval()
        return self

    @torch.no_grad()
    def _forward_all(self, agent_state: torch.Tensor, cog_state: torch.Tensor):
        feat = self.normalizer.oracle_feat(agent_state, cog_state)
        vs, rs, ps = [], [], []
        for m in self.members:
            v, r, p = m(feat)
            vs.append(v)
            rs.append(r)
            ps.append(p)
        return torch.stack(vs), torch.stack(rs), torch.stack(ps)

    @torch.no_grad()
    def score(
        self,
        agent_state: torch.Tensor,
        cog_state: torch.Tensor,
        rng: np.random.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        v_logit, rm_logit, _ = self._forward_all(agent_state, cog_state)
        v_sig, rm_sig = torch.sigmoid(v_logit), torch.sigmoid(rm_logit)
        idx2 = _random_pair(len(self.members), rng)
        return {
            "v_mean2": v_sig[idx2].mean(0),
            "v_mean_all": v_sig.mean(0),
            "v_std": v_sig.std(0),
            "rm_mean2": rm_sig[idx2].mean(0),
            "rm_mean_all": rm_sig.mean(0),
            "rm_std": rm_sig.std(0),
        }

    def save(self, path) -> None:
        torch.save(
            {
                "net_type": "OracleNet",
                "model_kwargs": self.members[0].kwargs,
                "normalizer": self.normalizer.state_dict(),
                "members": [m.state_dict() for m in self.members],
            },
            path,
        )

    @classmethod
    def load(cls, path, device: str = "cpu") -> OracleEnsemble:
        d = _torch_load(path, map_location=device)
        assert d["net_type"] == "OracleNet", d["net_type"]
        members = []
        for sd in d["members"]:
            m = OracleNet(**d["model_kwargs"])
            m.load_state_dict(sd)
            m.to(device).eval()
            members.append(m)
        norm = Normalizer()
        norm.load_state_dict(d["normalizer"])
        return cls(members, norm)
