# NanoWM imagined-rollout augmentation for PushT diffusion policy

Research code for a prototype, kept in-tree under `examples/`. Generated data
(`imagined/`, `cl/`, `diag/`, `logs/`, `outputs/`) is git-ignored -- everything here is
reproducible from the scripts below.

## Layout and the two-venv split

NanoWM has its own uv venv (Python 3.11, torch 2.7.1 / diffusers 0.29.2), which cannot be
merged with the lerobot venv (Python >= 3.12).
The pipeline therefore crosses venvs through `.npz` files on disk, and the directory
layout follows that boundary:

| path                    | venv             | what                                                                                     |
| ----------------------- | ---------------- | ---------------------------------------------------------------------------------------- |
| `imagined_rollout.py`   | **nanowm uv**    | both NanoWM generation modes, sharing one `load_nanowm`: `closed-loop` (the persistent worker, spawned by `eval/closed_loop_rollout.py`) and `open-loop` (stage 1 -> `.npz` shards) |
| `datagen/`              | **lerobot uv**   | stage 2 -- `.npz` / archives -> `LeRobotDataset` (plus the two pure-numpy modules below) |
| `eval/`                 | **lerobot uv**   | training + eval drivers (Python)                                                          |
| `scripts/`              | **lerobot uv**   | every shell recipe, run from the repo root                                                |

`datagen/action_convert.py` and `datagen/coverage_from_frame.py` are the exception: they
are pure numpy (+ cv2) and are imported from **both** venvs, so they must never import
`lerobot`. That is also why the package is called `datagen` and not `utils` -- `utils`
would collide with nano-world-model's own top-level `src/utils`, a hard dependency of
`imagined_rollout`.

Prerequisites (pinned revisions and sha256 in `configs/nanowm_pusht_common.yaml`):

    git clone https://github.com/simchowitzlabpublic/nano-world-model ~/repos/nano-world-model
    (cd ~/repos/nano-world-model && uv sync)                 # -> .venv, the nanowm venv
    hf download knightnemo/nanowm-b2-dino-wm-pusht-100k --local-dir ~/repos/nano-world-model/ckpt
    hf download stabilityai/sd-vae-ft-mse --local-dir ~/models/sd-vae-ft-mse

`imagined_rollout.py` loads `ckpt/model.safetensors` (falling back to a legacy `ckpt/model.pt`)
and points `VAE_MODEL_PATH` at `~/models/sd-vae-ft-mse` when that directory exists. The
DINO-WM archive is expected at `$DATASET_DIR` (default `~/Downloads/bmw48-osfstorage-datasets-archive`).

## Goal

Use the **SD-VAE** NanoWM (`knightnemo/nanowm-b2-dino-wm-pusht-100k`) as a
pixel-space data-augmentation source for an **image-only** LeRobot diffusion
policy on PushT, then measure `a` vs `a+aug`.

Everything lives in one visual distribution by training on the exact PushT data
NanoWM used: DINO-WM `pusht_noise` (osf.io/bmw48).

## Key alignment facts (Task #0)

|                        | value                                                                                                                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Env                    | both nano-world-model `PushTEnv` and `gym_pusht` are forks of `diffusion_policy` PushTEnv — same colors (RoyalBlue agent, LightSlateGray T, white bg, LightGreen goal), 512² pygame raster |
| `pusht_noise` obses    | 224×224 uint8, 1 frame / control step, ~109 frames/episode, 18685 train episodes                                                                                                           |
| state                  | 5-dim `[agent_x, agent_y, block_x, block_y, block_angle]` (+2 vel) — **contains block pose → not reconstructable from actions → image-only only**                                          |
| action                 | `abs_actions` = absolute agent target xy (same semantics as `lerobot/pusht` action)                                                                                                        |
| NanoWM SD-VAE pipeline | resize 224→256 stretch, pixels `[-1,1]`, `frame_interval=5` (1 NanoWM step = 5 sim steps → 10-dim action/step), `num_frames=4`, `n_context_frames=1`                                       |
| eval                   | `lerobot-eval env.type=pusht obs_type=pixels observation_width=observation_height=224`                                                                                                     |

### Frame-rate / temporal-window decision

`frame_interval` (NanoWM, fixed = 5) and `n_obs_steps` (policy, our choice) both
control temporal spacing but in different systems; they collide when mixing datasets:

- NanoWM only _generates_ frames 5 sim-steps apart → imagined episodes are fps 2.
- Real `pusht_noise` mp4s are dense (fps 10) — no gap to "fill"; NanoWM is a forward
  world model (branch counterfactual futures from a start frame), **not** an interpolator.
- If real is fps 10 and imagined is fps 2, the same `n_obs_steps` / `horizon` covers a
  5× different real-time window in each → corrupts training.

**Fix: unify fps.** Sub-sample the real dataset to fps 2 (`--frame-interval 5`, done in
the converter). Then use the _same_ policy hyperparams for `a` and `a+aug`:

| param            | value             | why                                                           |
| ---------------- | ----------------- | ------------------------------------------------------------- |
| `n_obs_steps`    | 1                 | no inter-obs spacing to be inconsistent                       |
| `horizon`        | 16 or 32 (not 64) | robustness to small imagined-obs error; PushT doesn't need 64 |
| `n_action_steps` | horizon / 2       | standard                                                      |

Shrinking `horizon` is a robustness hedge, **not** required to "match NanoWM": a training
sample only consumes `n_obs_steps` image frames — the `horizon` action labels are
ground-truth from the (perturbed-expert) action sequence and never rendered.

## Diffusion-policy patch (`observation.state` optional)

`src/lerobot/policies/diffusion/{modeling,configuration}_diffusion.py` — small
backward-compatible change so the policy runs image-only (`robot_state_feature is None`):

- `reset()` only creates the `OBS_STATE` queue when `robot_state_feature` is set.
- `DiffusionModel.__init__` — state term contributes 0 to `global_cond_dim` when absent.
- new `_batch_bs_nobs()` helper reads `(batch, n_obs_steps)` from state → images →
  env-state, whichever is present; used in `_prepare_global_conditioning`,
  `generate_actions`, `compute_loss`.
- `compute_loss` no longer asserts `OBS_STATE` in the batch when it isn't a feature.

With `observation.state` present, behaviour is byte-identical (all `-k diffusion`
tests pass -- 16 passed / 41 skipped on a CUDA box). Verified image-only: `make_policy` → `forward`/`backward` (UNet + ResNet
get grads) → `select_action` all work; a 3-step `lerobot-train` run completes.

## Scripts

### `datagen/run_converter.py pusht-noise`

`pusht_noise/{train,val}` → `LeRobotDataset` (mirrors `lerobot/pusht` schema at 224²).
All dataset converters share `datagen/dataset_converter.py` (`LeRobotDatasetConverter`)
and are dispatched by `datagen/run_converter.py <name>` (`--list` for the names).

**Frame rate is sub-sampled to match NanoWM.** NanoWM's PushT checkpoint runs at
`frame_interval=5` (1 NanoWM step = 5 sim-steps), so its _generated_ frames are 5
sim-steps apart. `--frame-interval 5` (default) sub-samples the real mp4s with the
same stride → **fps 2**, so real and imagined episodes share one frame rate and
`n_obs_steps` / `horizon` mean the same real-time window in both. `--frame-interval 1`
keeps native fps 10.

```bash
# smoke test
uv run python examples/nanowm_pusht_aug/datagen/run_converter.py pusht-noise \
    --splits val --max-episodes 20 --overwrite

# full, fps-2 matched to NanoWM (repo_id -> local/pusht_noise_224_fi5)
uv run python examples/nanowm_pusht_aug/datagen/run_converter.py pusht-noise \
    --splits train val --frame-interval 5 --overwrite
```

Output feature schema:

| key                    | shape             | note                                                                    |
| ---------------------- | ----------------- | ----------------------------------------------------------------------- |
| `observation.image`    | video (224,224,3) | decoded to CHW `[0,1]` on load                                          |
| `observation.state`    | (2,)              | agent xy — `lerobot/pusht` compatible                                   |
| `action`               | (2,)              | absolute target xy at the **end** of each 5-step window (`--action-at`) |
| `observation.pose`     | (5,)              | full env state, ignored by image-only policy                            |
| `observation.velocity` | (2,)              | agent velocity, ignored by image-only policy                            |

`--minimal` drops `pose`/`velocity`; **`--drop-state`** keeps only `observation.image`

- `action` (repo_id gets `_imgonly`) — the dataset for Task #1.
  `--action-at {end,start,next_agent_pos}` selects how the coarse action is picked from
  the 5 sim-step targets in each window (`end` = demonstrator's target by window end;
  `next_agent_pos` = most eval-consistent).

Coarse-frame agent motion is ~30–120 px per step (0.5 s), well within a held-target
PD controller's reach.

### `datagen/pack_closed_loop.py` -- closed-loop imagined data

Packs accepted closed-loop rollouts (`eval/closed_loop_rollout.py`) as fps-10 episodes that
aggregate with `local/pusht_imgonly`. Each (imagined frame, action chunk) pair becomes its
own short episode, so the sampler reads exactly one fresh image per pair.

**Defaults: `--ep-len 10`, `drop_n_last_frames=9`, `--pad-mode hold`.**

`EpisodeAwareSampler` gives an episode of length `L` exactly `L − drop` sample starts, so one
sample per imagined episode requires

```
ep_len = drop_n_last_frames + 1
```

`drop` is global and also trims the real dataset, so `ep_len` trades controllable label slots
against real samples:

| ep_len | drop  | slots written by packer | auto-padded (copy of last row) | real samples | vs drop=8 |
| ------ | ----- | ----------------------- | ------------------------------ | ------------ | --------- |
| 9      | 8     | L0–L8                   | 7 (L9–L15)                     | 24,002       | —         |
| **10** | **9** | **L0–L9**               | **6 (L10–L15)**                | **23,796**   | **−0.9%** |
| 16     | 15    | L0–L15                  | 0                              | 22,560       | −6.0%     |

`ep_len=16` is not usable: `drop=15` strips the 1,442 highest-coverage real frames (mean
coverage 0.608 vs 0.258 kept), which took Control from 70.0% to 59.5%.

`hold` fills slots that have no genuine executed action with the agent's own position (the
only target that makes the PD controller stop; `zero` is a target at the top-left corner and
`copy` keeps pushing). The auto-pad of L10–L15 then copies L9, which is already the hold
value, so all 16 slots are consistent.

**`drop` must match on both arms.** `scripts/finetune_cl.sh <arm> [seed] [steps] [drop]` defaults
to `drop=9`; Control has to be trained at the same value as the augmented arm. Datasets packed
with `--ep-len 9` (`run_dose464.sh`, `run_dose500.sh`) pass `drop=8` explicitly.

```bash
uv run --extra training python examples/nanowm_pusht_aug/datagen/pack_closed_loop.py \
    --accepted examples/nanowm_pusht_aug/cl/pool500 --repo-id local/pusht_cl_allpre_h10 \
    --truncate-cov 0.95 --frame-select all --max-pairs 464 --overwrite
# aggregate with local/pusht_imgonly (pin stats to the real set), then:
bash examples/nanowm_pusht_aug/scripts/finetune_cl.sh allpre_h 1000 20000   # drop=9
bash examples/nanowm_pusht_aug/scripts/finetune_cl.sh a        1000 20000   # Control, drop=9
```

## Next steps

1. **T1** — image-only diffusion policy baseline (`a`). Build the dataset then run
   `train_baseline.sh` (>=3 seeds), eval each on gym_pusht at 224:
   ```bash
   uv run python examples/nanowm_pusht_aug/datagen/run_converter.py pusht-noise \
       --splits train val --frame-interval 5 --drop-state --overwrite
   bash examples/nanowm_pusht_aug/scripts/train_baseline.sh 100000
   ```
2. **T3** — policy sim-rollout in nano-world-model's `PushTEnv` (render_size 224) to
   collect context frames + visited states.
3. **T4** — NanoWM branch rollouts from visited states, perturbed expert actions,
   VAE-decode → `LeRobotDataset` of imagined episodes at fps 2, same schema.
4. **T6/T7** — `MultiLeRobotDataset` mix, sweep ratio, compare `a` vs `a+aug`
   (both with photometric aug on to suppress the domain shortcut).
