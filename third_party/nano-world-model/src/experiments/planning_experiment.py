"""Planning experiment with diffusion-based world models (MPC via CEM)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import imageio
from omegaconf import OmegaConf

from .base import BaseExperiment
from diffusion import create_diffusion
from latent_codecs import build_latent_codec
from models import get_models
from planning import (
    CEMPlanner,
    DiffusionWorldModel,
    Preprocessor,
    create_objective_fn,
)
from planning.envs import create_planning_env


@dataclass
class PlanningGoal:
    init_state: np.ndarray
    goal_state: np.ndarray
    env_info: Optional[Dict[str, Any]]
    goal_meta: Dict[str, Any]
    init_obs: Optional[Dict[str, Any]] = None
    goal_obs: Optional[Dict[str, Any]] = None


class PlanningExperiment(BaseExperiment):
    """MPC planning experiment using trained world models."""

    def planning(self):
        cfg = self.cfg
        plan_cfg = cfg.planning

        if not hasattr(cfg, "ckpt_path") or cfg.ckpt_path is None:
            raise ValueError("Set ckpt_path via CLI")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Load model + latent codec + diffusion from checkpoint ---
        model, latent_codec, diffusion, train_cfg = self._load_from_checkpoint(
            cfg.ckpt_path, device
        )

        # Override sampling steps for planning (faster than training default)
        wm_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        if plan_cfg.get("num_sampling_steps") is not None:
            wm_cfg.model.num_sampling_steps = plan_cfg.num_sampling_steps

        world_model = DiffusionWorldModel(model, latent_codec, diffusion, wm_cfg)

        # --- Preprocessor (action normalization stats from dataset) ---
        preprocessor = self._build_preprocessor(train_cfg, device)

        # --- Environment ---
        env_name = plan_cfg.env_name
        print(f"Creating env: {env_name}")
        env = create_planning_env(env_name, device="cpu")

        # --- Goals ---
        n_evals = plan_cfg.n_evals
        goal_source = plan_cfg.get("goal_source", "random_state")
        goal_H = plan_cfg.get("goal_H", plan_cfg.horizon)
        frame_interval = train_cfg.dataset.frame_interval
        print(f"Sampling {n_evals} (init, goal) pairs (goal_source={goal_source}, goal_H={goal_H})")
        if goal_source == "dset":
            goals = self._sample_dset_goals(
                env=env,
                train_cfg=train_cfg,
                n_evals=n_evals,
                goal_H=goal_H,
                frame_interval=frame_interval,
                seed=plan_cfg.seed,
            )
        else:
            goals = []
            for i in range(n_evals):
                init_state, goal_state = env.sample_random_init_goal_states(seed=plan_cfg.seed + i)
                goals.append(
                    PlanningGoal(
                        init_state=np.asarray(init_state),
                        goal_state=np.asarray(goal_state),
                        env_info=None,
                        goal_meta={
                            "source": "random_state",
                            "episode": int(i),
                            "seed": int(plan_cfg.seed + i),
                        },
                    )
                )

        out_dir = Path(plan_cfg.get("output_dir", "planning_results"))
        out_dir.mkdir(parents=True, exist_ok=True)
        targets_path = out_dir / "planning_targets.json"
        with open(targets_path, "w") as f:
            json.dump(self._serialize_goals(goals), f, indent=2)
        print(f"Planning targets saved to {targets_path}")

        # --- Planner ---
        action_dim = train_cfg.dataset.spec.action_dim
        action_dim_total = action_dim * frame_interval

        objective_fn = create_objective_fn(
            alpha=plan_cfg.objective.alpha,
            base=plan_cfg.objective.base,
            mode=plan_cfg.objective.mode,
            visual_metric=plan_cfg.objective.get("visual_metric", "mse"),
            token_dim=plan_cfg.objective.get("token_dim", None),
        )

        planner = CEMPlanner(
            world_model=world_model,
            objective_fn=objective_fn,
            action_dim=action_dim_total,
            horizon=plan_cfg.horizon,
            num_samples=plan_cfg.cem.num_samples,
            topk=plan_cfg.cem.topk,
            opt_steps=plan_cfg.cem.opt_steps,
            var_scale=plan_cfg.cem.var_scale,
            eval_every=plan_cfg.cem.eval_every,
            sigma_min=plan_cfg.cem.get("sigma_min", 1e-3),
            action_low=plan_cfg.cem.get("action_low", None),
            action_high=plan_cfg.cem.get("action_high", None),
            rollout_batch_size=plan_cfg.cem.get("rollout_batch_size", None),
            name="CEM",
            device=str(device),
        )

        # --- Run MPC ---
        image_size = train_cfg.get("image_size", train_cfg.model.get("image_size", 256))
        if not isinstance(image_size, int):
            image_size = image_size[0]

        results = self._run_mpc(
            env=env,
            planner=planner,
            preprocessor=preprocessor,
            goals=goals,
            plan_cfg=plan_cfg,
            image_size=image_size,
            frame_interval=frame_interval,
            device=device,
        )

        # --- Save ---
        results_path = out_dir / "planning_results.json"
        with open(results_path, "w") as f:
            json.dump(self._json_safe(results), f, indent=2)

        print(f"\nResults saved to {results_path}")
        print(f"Success rate: {results['success_rate']:.2%}")
        print(f"Mean state dist: {results['state_dist_mean']:.4f}")
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_from_checkpoint(self, ckpt_path, device):
        ckpt_path_str = str(ckpt_path)

        # HuggingFace model ID (e.g. "knightnemo/nanowm-b2-dino-wm-point-maze-30k")
        if "/" in ckpt_path_str and not Path(ckpt_path_str).exists():
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(ckpt_path_str, "config.yaml")
            model_path = hf_hub_download(ckpt_path_str, "model.safetensors")
            train_cfg = OmegaConf.load(config_path)
            print(f"Loaded config from HF: {ckpt_path_str}")

            model = get_models(train_cfg).to(device)
            from safetensors.torch import load_file
            sd = load_file(model_path, device=str(device))
            model.load_state_dict(sd)
            model.eval()
            print(f"Model loaded from HF: {ckpt_path_str}")
        else:
            # Local checkpoint — walk up from ckpt to find config.yaml
            ckpt_path = Path(ckpt_path_str).expanduser().resolve()
            train_config_path = None
            for ancestor in [ckpt_path.parent] + list(ckpt_path.parents):
                candidate = ancestor / "config.yaml"
                if candidate.exists():
                    train_config_path = candidate
                    break
            if train_config_path is None:
                raise FileNotFoundError(f"Training config not found near: {ckpt_path}")

            train_cfg = OmegaConf.load(train_config_path)
            print(f"Loaded training config: {train_config_path}")

            model = get_models(train_cfg).to(device)
            if ckpt_path.suffix == ".safetensors":
                from safetensors.torch import load_file
                sd = load_file(str(ckpt_path), device=str(device))
                model.load_state_dict(sd)
            else:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                if "state_dict" in ckpt:
                    sd = {}
                    for k, v in ckpt["state_dict"].items():
                        k = k.removeprefix("model.")
                        k = k.removeprefix("_orig_mod.")
                        if k.startswith("vae."):
                            continue
                        sd[k] = v
                    model.load_state_dict(sd)
                elif "model" in ckpt:
                    model.load_state_dict(ckpt["model"])
                else:
                    model.load_state_dict(ckpt)
            model.eval()
            print(f"Model loaded from {ckpt_path}")

        latent_codec = build_latent_codec(train_cfg).to(device).eval()

        diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule=train_cfg.experiment.diffusion.noise_schedule,
            pred_name=train_cfg.experiment.diffusion.pred_name,
            diffusion_steps=train_cfg.experiment.diffusion.diffusion_steps,
            snr_gamma=train_cfg.experiment.diffusion.snr_gamma,
            zero_terminal_snr=train_cfg.experiment.diffusion.zero_terminal_snr,
        )
        return model, latent_codec, diffusion, train_cfg

    def _build_preprocessor(self, train_cfg, device):
        image_size = train_cfg.get("image_size", train_cfg.model.get("image_size", 256))
        if not isinstance(image_size, int):
            image_size = image_size[0]

        action_mean = action_std = None
        normalize_action = train_cfg.dataset.loader.get("normalize_action", False)
        if normalize_action:
            action_mean, action_std = self._load_action_stats(train_cfg)

        return Preprocessor(
            image_size=image_size,
            normalize=True,
            action_mean=action_mean,
            action_std=action_std,
            device=str(device),
        )

    def _load_action_stats(self, train_cfg):
        """Load action normalization stats from dataset cache.

        Datasets disagree on the loader key:
        - point_maze / wall: `data_path` (single dir, has wm_stats_cache/)
        - pusht: `data_path_train` (split dirs; train has wm_stats_cache/)
        """
        import glob, os
        loader = train_cfg.dataset.loader
        candidates = []
        for key in ("data_path", "data_path_train"):
            try:
                candidates.append(loader[key])
            except Exception:
                pass
        if not candidates:
            print("Warning: no data_path / data_path_train in loader; skipping action stats")
            return None, None

        dataset_dir = os.environ.get("DATASET_DIR", "")
        for data_path in candidates:
            if not Path(data_path).is_absolute() and dataset_dir:
                data_path = str(Path(dataset_dir) / data_path.lstrip("./"))
            cache_dir = Path(data_path) / "wm_stats_cache"
            if not cache_dir.exists():
                continue
            files = sorted(glob.glob(str(cache_dir / "action_train_*.pt")))
            if not files:
                continue
            stats = torch.load(files[0], map_location="cpu", weights_only=True)
            mean = stats["mean"]
            std = stats["std"]
            frame_interval = train_cfg.dataset.get("frame_interval", 1)
            if frame_interval > 1:
                mean = mean.repeat(frame_interval)
                std = std.repeat(frame_interval)
            print(f"Loaded action stats from {cache_dir}: mean={mean}, std={std}")
            return mean, std
        print("Warning: no action stats found, actions will not be denormalized")
        return None, None

    def _sample_dset_goals(self, env, train_cfg, n_evals, goal_H, frame_interval, seed):
        """Sample (init, goal) pairs by replaying ground-truth actions for goal_H * frame_interval
        env steps. Mirrors DINO-WM's `goal_source='dset'` mode: guarantees the goal is reachable
        from the init state in `goal_H` planner steps, so success rate measures world-model
        accuracy rather than goal feasibility.

        Layouts handled:
          - pusht: states.pth + velocities.pth (concatenated for 7D state) + rel_actions/abs_actions
            divided by action_scale, sampled from val/ split.
          - point_maze: states.pth + actions.pth + seq_lengths.pth from a single dir.
          - wall: same as point_maze, plus per-trajectory door/wall locations passed via
            env.update_env() before each replay so the environment matches the dataset trajectory.
        """
        import os, random, pickle as _pickle
        loader = train_cfg.dataset.loader
        # Resolve data dir (try data_path_val first for split layouts; fall back to data_path).
        # OmegaConf returns None for keys explicitly set to null, so filter both KeyError and None.
        val_path = None
        for key in ("data_path_val", "data_path"):
            p = loader.get(key) if hasattr(loader, "get") else None
            if p is None:
                continue
            if not Path(p).is_absolute():
                ds_dir = os.environ.get("DATASET_DIR", "")
                if ds_dir:
                    p = str(Path(ds_dir) / p.lstrip("./"))
            if Path(p).exists():
                val_path = Path(p)
                if key == "data_path":
                    sub = val_path / "val"
                    if sub.exists():
                        val_path = sub
                break
        if val_path is None:
            raise RuntimeError("goal_source='dset' but could not resolve val data path")

        states = torch.load(val_path / "states.pth").float()
        vel_path = val_path / "velocities.pth"
        if vel_path.exists() and bool(loader.get("with_velocity") or False):
            velocities = torch.load(vel_path).float()
            if velocities.shape[:2] == states.shape[:2]:
                states = torch.cat([states, velocities], dim=-1)
        use_relative = bool(loader.get("use_relative_actions") or False)
        action_scale = float(loader.get("action_scale") or 1.0)
        actions_file = val_path / ("rel_actions.pth" if use_relative else "abs_actions.pth")
        if not actions_file.exists():
            actions_file = val_path / "actions.pth"
        actions = torch.load(actions_file).float()
        if action_scale != 1.0:
            actions = actions / action_scale

        # Wall env varies door/wall location per trajectory; replay must use the matching layout.
        door_locations = None
        wall_locations = None
        for f, key in (("door_locations.pth", "door"), ("wall_locations.pth", "wall")):
            p = val_path / f
            if p.exists():
                t = torch.load(p)
                if key == "door":
                    door_locations = t
                else:
                    wall_locations = t

        shapes = None
        shapes_path = val_path / "shapes.pkl"
        if shapes_path.exists():
            with open(shapes_path, "rb") as f:
                shapes = _pickle.load(f)

        # Sequence lengths (default to full traj length when not provided).
        seq_path_pkl = val_path / "seq_lengths.pkl"
        seq_path_pth = val_path / "seq_lengths.pth"
        if seq_path_pth.exists():
            seq_lengths = torch.load(seq_path_pth)
            if isinstance(seq_lengths, torch.Tensor):
                seq_lengths = seq_lengths.tolist()
        elif seq_path_pkl.exists():
            with open(seq_path_pkl, "rb") as f:
                seq_lengths = _pickle.load(f)
        else:
            seq_lengths = [states.shape[1]] * states.shape[0]
        seq_lengths = list(seq_lengths)

        traj_len = frame_interval * goal_H + 1
        valid_traj = [i for i, L in enumerate(seq_lengths) if L >= traj_len]
        if not valid_traj:
            raise RuntimeError(f"No val trajectory with length ≥ {traj_len}")

        rng = random.Random(seed)
        goals = []
        for ep_idx in range(n_evals):
            traj_id = rng.choice(valid_traj)
            offset = rng.randint(0, seq_lengths[traj_id] - traj_len)
            init_state = states[traj_id, offset].numpy()
            act_seq = actions[traj_id, offset : offset + frame_interval * goal_H].numpy()

            env_info = None
            if door_locations is not None and wall_locations is not None and hasattr(env, "update_env"):
                env_info = {
                    "fix_door_location": door_locations[traj_id, 0],
                    "fix_wall_location": wall_locations[traj_id, 0],
                }
                env.update_env(env_info)
            elif shapes is not None and hasattr(env, "update_env"):
                env_info = {"shape": shapes[traj_id]}
                env.update_env(env_info)

            ep_seed = seed + ep_idx
            init_obs, _ = env.prepare(ep_seed, init_state)
            cur_state = init_state
            for a in act_seq:
                goal_obs, _, _, info = env.step(a)
                cur_state = info.get("state", cur_state)
            goal_state = np.asarray(cur_state)
            goals.append(
                PlanningGoal(
                    init_state=np.asarray(init_state),
                    goal_state=goal_state,
                    env_info=env_info,
                    goal_meta={
                        "source": "dset",
                        "episode": int(ep_idx),
                        "seed": int(ep_seed),
                        "val_path": str(val_path),
                        "traj_id": int(traj_id),
                        "offset": int(offset),
                        "traj_len": int(traj_len),
                        "frame_interval": int(frame_interval),
                        "goal_H": int(goal_H),
                    },
                    init_obs=self._copy_obs_dict(init_obs),
                    goal_obs=self._copy_obs_dict(goal_obs),
                )
            )
        return goals

    def _serialize_goals(self, goals):
        records = []
        for ep_idx, goal in enumerate(goals):
            goal = self._coerce_goal(goal)
            records.append({
                "episode": int(ep_idx),
                "init_state": self._json_safe(goal.init_state),
                "goal_state": self._json_safe(goal.goal_state),
                "env_info": self._json_safe(goal.env_info),
                "goal_meta": self._json_safe(goal.goal_meta),
            })
        return records

    @staticmethod
    def _copy_obs_dict(obs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if obs is None:
            return None
        copied = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                copied[key] = value.detach().cpu().clone()
            elif isinstance(value, np.ndarray):
                copied[key] = value.copy()
            else:
                copied[key] = value
        return copied

    @staticmethod
    def _coerce_goal(goal) -> PlanningGoal:
        """Accept old tuple goals while using an explicit type internally."""
        if isinstance(goal, PlanningGoal):
            return goal
        if len(goal) == 5:
            init_state, goal_state, env_info, goal_meta, goal_obs = goal
            return PlanningGoal(
                init_state=np.asarray(init_state),
                goal_state=np.asarray(goal_state),
                env_info=env_info,
                goal_meta=goal_meta,
                goal_obs=goal_obs,
            )
        if len(goal) == 4:
            init_state, goal_state, env_info, goal_meta = goal
        else:
            init_state, goal_state, env_info = goal
            goal_meta = {}
        return PlanningGoal(
            init_state=np.asarray(init_state),
            goal_state=np.asarray(goal_state),
            env_info=env_info,
            goal_meta=goal_meta,
        )

    @staticmethod
    def _json_safe(obj):
        if obj is None:
            return None
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return PlanningExperiment._json_safe(obj.detach().cpu().numpy())
        if isinstance(obj, np.ndarray):
            return PlanningExperiment._json_safe(obj.tolist())
        if isinstance(obj, np.generic):
            return PlanningExperiment._json_safe(obj.item())
        if isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        if isinstance(obj, (str, int, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): PlanningExperiment._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [PlanningExperiment._json_safe(v) for v in obj]
        return str(obj)

    @staticmethod
    def _metric_values(all_metrics, key):
        vals = []
        for m in all_metrics:
            value = m.get(key)
            if value is None:
                continue
            value = float(value)
            if np.isfinite(value):
                vals.append(value)
        return np.asarray(vals, dtype=np.float64)

    @staticmethod
    def _metric_summary(name, values):
        if values.size == 0:
            return {
                f"{name}_mean": None,
                f"{name}_median": None,
                f"{name}_std": None,
                f"{name}_stderr": None,
                f"{name}_min": None,
                f"{name}_max": None,
            }
        return {
            f"{name}_mean": float(np.mean(values)),
            f"{name}_median": float(np.median(values)),
            f"{name}_std": float(np.std(values)),
            f"{name}_stderr": float(np.std(values) / np.sqrt(values.size)),
            f"{name}_min": float(np.min(values)),
            f"{name}_max": float(np.max(values)),
        }

    def _obs_to_tensor(self, obs: Dict[str, Any], image_size, device):
        """Convert an env observation to a [1, 1, 3, H, W] tensor in [-1, 1]."""
        img = obs["visual"]  # numpy (H, W, C) uint8 or torch
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        from torchvision.transforms.functional import resize
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
        t = resize(t, [image_size, image_size], antialias=True)
        return t.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,3,H,W]

    def _render_state(self, env, seed, state, image_size, device):
        """Render a state by resetting the env. Use only outside live rollouts."""
        obs, _ = env.prepare(seed, state)
        return self._obs_to_tensor(obs, image_size, device)

    def _obs_to_frame(self, obs_dict, image_size):
        """Convert env obs dict to uint8 numpy (H, W, 3) for video saving."""
        img = obs_dict["visual"]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.shape[0] == 3:  # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        from PIL import Image
        img = np.array(Image.fromarray(img).resize((image_size, image_size)))
        return img

    def _tensor_obs_to_frame(self, obs_tensor):
        """Convert [1, 1, 3, H, W] in [-1, 1] to uint8 HWC."""
        img = obs_tensor[0, 0].detach().cpu().permute(1, 2, 0).numpy()
        img = ((np.clip(img, -1, 1) + 1.0) / 2.0 * 255.0).astype(np.uint8)
        return img

    def _run_mpc(
        self, env, planner, preprocessor, goals, plan_cfg,
        image_size, frame_interval, device,
    ):
        replan_every = plan_cfg.get("replan_every", 1)
        max_steps = plan_cfg.get("max_episode_steps", 50)
        n_plot = plan_cfg.get("n_plot_samples", 5)
        out_dir = Path(plan_cfg.get("output_dir", "planning_results"))
        out_dir.mkdir(parents=True, exist_ok=True)

        all_metrics = []

        for ep_idx, goal_item in enumerate(goals):
            goal = self._coerce_goal(goal_item)
            init_state = goal.init_state
            goal_state = goal.goal_state
            env_info = goal.env_info
            goal_meta = goal.goal_meta
            seed = plan_cfg.seed + ep_idx
            print(f"\n[Episode {ep_idx+1}/{len(goals)}]")

            # Per-trajectory env layout (e.g. wall door/wall positions). Must be applied before
            # both the goal render and the planning rollout so they observe the same world.
            if env_info is not None and hasattr(env, "update_env"):
                env.update_env(env_info)

            # DINO-WM uses the goal observation from the replayed trajectory. If it is not
            # available (e.g. random_state goals), render it by resetting outside the live rollout.
            if goal.goal_obs is not None:
                obs_g_tensor = self._obs_to_tensor(goal.goal_obs, image_size, device)
            else:
                obs_g_tensor = self._render_state(env, seed + 10000, goal_state, image_size, device)
            goal_frame = self._tensor_obs_to_frame(obs_g_tensor)

            # Reset env to init state (re-apply env_info — _render_state above may have reset it)
            if env_info is not None and hasattr(env, "update_env"):
                env.update_env(env_info)
            obs_dict, cur_state = env.prepare(seed, init_state)
            obs_0_source = goal.init_obs if goal.init_obs is not None else obs_dict
            obs_0_tensor = self._obs_to_tensor(obs_0_source, image_size, device)

            frames = [self._obs_to_frame(obs_0_source, image_size)]
            prev_actions = None
            done = False
            success_step = None

            def _to_np(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
                return np.asarray(x)

            for step in range(max_steps):
                if step % replan_every == 0:
                    warm_start = None
                    if prev_actions is not None and prev_actions.shape[1] > replan_every:
                        warm_start = prev_actions[:, replan_every:]

                    obs_0_dict = {"visual": obs_0_tensor}
                    obs_g_dict = {"visual": obs_g_tensor}
                    with torch.no_grad():
                        actions, info = planner.plan(obs_0_dict, obs_g_dict, actions=warm_start)
                    prev_actions = actions

                # Execute one action in env (denormalize from model space to env space)
                act_idx = step % replan_every
                act = actions[0, act_idx]
                act = preprocessor.denormalize_action(act).cpu().numpy()

                # Undo frame_interval packing: split into sub-actions
                if frame_interval > 1:
                    act_per_step = act.reshape(frame_interval, -1)
                else:
                    act_per_step = act[np.newaxis]

                for sub_act in act_per_step:
                    obs_dict, _, done_flag, info_dict = env.step(sub_act)
                    frames.append(self._obs_to_frame(obs_dict, image_size))
                    cur_state = info_dict.get("state", cur_state)
                    if done_flag:
                        done = True
                        break

                if done:
                    break

                # Keep the live env continuous. Do not call _render_state() here because it
                # resets PushT and drops simulator state between action chunks.
                obs_0_tensor = self._obs_to_tensor(obs_dict, image_size, device)

                # DINO-WM's MPC evaluator checks success after each executed chunk.
                # Stop the episode once the goal is reached instead of judging only
                # the state after the full max_episode_steps budget.
                is_checkpoint = ((step + 1) % replan_every == 0) or (step + 1 == max_steps)
                if is_checkpoint:
                    checkpoint_eval = env.eval_state(_to_np(goal_state), _to_np(cur_state))
                    if bool(checkpoint_eval["success"]):
                        success_step = step + 1
                        break

            # Evaluate (some envs mix np / torch, normalize to numpy)
            gs_np = _to_np(goal_state)
            cs_np = _to_np(cur_state)
            eval_result = env.eval_state(gs_np, cs_np)
            success = float(eval_result["success"])
            state_dist = float(eval_result["state_dist"])
            xy_dist = float(np.linalg.norm(gs_np[:2] - cs_np[:2])) if gs_np.ndim == 1 and gs_np.shape[0] >= 2 else float("nan")
            # pusht success uses first-4 dims (agent + T-block xy); print it for diagnosis.
            pos4_dist = float(np.linalg.norm(gs_np[:4] - cs_np[:4])) if gs_np.ndim == 1 and gs_np.shape[0] >= 4 else float("nan")
            if gs_np.ndim == 1 and gs_np.shape[0] >= 5:
                angle_diff = float(abs(gs_np[4] - cs_np[4]))
                angle_dist = float(min(angle_diff, 2 * np.pi - angle_diff))
            else:
                angle_dist = float("nan")
            print(f"  success={bool(success)}, state_dist={state_dist:.4f}, xy_dist={xy_dist:.4f}, pos4_dist={pos4_dist:.4f}, angle_dist={angle_dist:.4f}, steps={len(frames)-1}")

            metrics = {
                "episode": int(ep_idx),
                "seed": int(seed),
                "success": success,
                "state_dist": state_dist,
                "xy_dist": xy_dist,
                "pos4_dist": pos4_dist,
                "angle_dist": angle_dist,
                "n_steps": len(frames) - 1,
                "success_step": success_step,
                "goal_meta": self._json_safe(goal_meta),
            }
            all_metrics.append(metrics)

            # Save video
            if ep_idx < n_plot:
                vid_path = out_dir / f"episode_{ep_idx:03d}.mp4"
                frames_with_goal = [
                    np.concatenate([frame, goal_frame], axis=1)
                    for frame in frames
                ]
                imageio.mimwrite(str(vid_path), frames_with_goal, fps=10, quality=8)
                print(f"  Video saved: {vid_path}")

        success_values = self._metric_values(all_metrics, "success")
        success_rate = float(np.mean(success_values)) if success_values.size else 0.0
        success_count = int(np.sum(success_values)) if success_values.size else 0

        results = {
            "success_rate": success_rate,
            "success_count": success_count,
            "n_episodes": len(all_metrics),
            "planning_config": self._json_safe(OmegaConf.to_container(plan_cfg, resolve=True)),
            "per_episode": all_metrics,
        }

        for key in ("state_dist", "xy_dist", "pos4_dist", "angle_dist", "n_steps"):
            results.update(self._metric_summary(key, self._metric_values(all_metrics, key)))
        return results
