# 提案：在 NanoWM 裡用 PPO 微調 PushT diffusion policy（參考 DiWA）

狀態：**草稿**，尚未實作、尚未預先註冊。參考來源：`third_party/diwa`（DiWA, CoRL 2025）與其上游 `third_party/dppo`（DPPO）。

## 0. 一句話

把 SAILOR 的「想像 rollout → 挑選 → 當 BC 資料」換成 DiWA 的做法：**NanoWM 當凍結的模擬器，diffusion policy 直接在 NanoWM 裡用 DPPO（denoising-chain PPO）微調**，reward 用想像影格的 coverage，最後只在 gym_pusht 上評估。先用同一套 PPO 程式在真 gym_pusht 跑一個對照組（PPO-Sim），確認 PPO 本身能把 base 69.4% 往上推，再換成 NanoWM。

## 1. DiWA 的設計拆解

### 1.1 World model（`diwa/wm/`）

| 元件 | DiWA 的做法 | 程式 |
| --- | --- | --- |
| WM 本體 | DreamerV2 RSSM（LUMOS），latent = `[h, z]` 共 2048 維；一步想像 = 一次 `wm.dream(action, (h,z))`，不需要解碼 | `wm/wrapper/base.py:wm_step` |
| 多步 | 一個 action chunk（`act_steps=4`）逐步 dream，chunk 結束才解碼一次影像（只用來錄影） | `wm/wrapper/visionwm_cls.py:multi_step` |
| policy 輸入 | **policy 直接吃 WM latent**（`obs_dim=2048`），不是影像。所以想像中不需要 decode | config `obs_dim: 2048` |
| reward | 在 latent 上訓練的 contrastive 二分類器（`model/rewcls/contrastive.py`），`argmax` 當 0/1 reward；reward>0 即 terminated | `visionwm_cls.py`、agent 的 `get_terminated_from_reward` |
| 初始狀態 | 從資料集的真實初始觀測編碼成 latent（`get_zero_wm_features`） | `wm/encoder/visionwm.py` |
| 真實評估 | 真實環境中每步用 RSSM posterior（`get_hist_wm_features`，帶 action 與 reset）把真實觀測過濾成 latent 再餵 policy | `env/runner/visionwm.py` |

### 1.2 PPO（`diwa/model/diffusion/diffusion_{vpg,ppo}.py`，源自 DPPO）

- **Denoising MDP**：把 diffusion 的去噪鏈當成內層 MDP。只對最後 `ft_denoising_steps` 步（20 步裡的 10 步）算 Gaussian logprob 並更新；`actor`（凍結的 base）+ `actor_ft`（可訓練的拷貝）。
- **探索**：`min_sampling_denoising_std=0.1`（去噪時的 std 下限）、`randn_clip_value=3`；logprob 計算用 `min_logprob_denoising_std=0.1`。
- **Loss**：PPO clip，clip 係數沿去噪步從 `clip_ploss_coef_base=1e-3` 指數增加到 `1e-2`（越接近乾淨動作越寬）；advantage 乘上 `gamma_denoising^k`（0.99）；`reward_horizon=act_steps`（只對真正執行的動作算 logprob）。
- **BC 正則**：可選，用 base policy 在同一 obs 上的去噪鏈，最大化其在 `actor_ft` 下的 logprob（README 範例 `bc_loss_coeff=0.025`）。
- **其他**：GAE（λ=0.95、γ=0.999）、running reward scaling、critic warmup 2 itr、`target_kl=1`、actor lr 1e-5 / critic lr 1e-3、`update_epochs=10`。

### 1.3 訓練迴圈（`agent/finetune/train_mb_ppo_diffusion_agent_visionwm.py`）

每個 itr：50 個想像 env 同步 rollout `n_steps=36` 個 chunk → 算 value / logprob → GAE → 10 個 epoch 的 minibatch 更新；每 `val_freq=25` itr 在真實 CALVIN/LIBERO 評估。WM 全程凍結，訓練時**零真實互動**（`offline_method: true`）。

### 1.4 不照搬的地方

- **截斷時的 bootstrap**：DiWA 只用 `terminated` 決定 `nonterminal`，env 因 `max_episode_steps` 截斷並 reset 後，GAE 會用「新 episode 初始狀態」的 value 去 bootstrap 舊 episode。本提案的想像 rollout 都是短的截斷 branch，必須用截斷當下狀態的 `V(s_T)` bootstrap。
- **授權**：DiWA 是 GPL-3.0，DPPO 是 MIT，lerobot 是 Apache-2.0。**移植程式碼一律從 `third_party/dppo` 取**（`diffusion_vpg.py` / `diffusion_ppo.py` 與 DiWA 版幾乎相同），DiWA 只參考設計。

## 2. DiWA 與 PushT + NanoWM 的對照

| 面向 | DiWA | PushT + NanoWM（本專案） | 影響 |
| --- | --- | --- | --- |
| WM 類型 | RSSM，一步 = 一次 GRU，非常快 | DFoT 影片 diffusion，SD-VAE latent `4×32×32`，3 個 context 幀、每幀 30 個 sampling step | **單步慢 3–4 個數量級**。T4 實測：2000 branch × 8 round ≈ 2 h 17 m（wm_batch 24、和 collect 並行）≈ **2 幀/秒/GPU**（含 policy 取樣） |
| 時間尺度 | 1 WM step = 1 控制步 | 1 NanoWM 幀 = 5 sim steps（`frame_interval=5`）；一個決策（`n_action_steps=10`）= 2 幀 | 一個決策要 2 次 DFoT |
| policy 輸入 | WM latent，不需 decode | lerobot 影像 policy（ResNet18 + UNet1D，96 px → crop 84），image-only | 每幀都要 VAE decode → resize 96。真實評估反而簡單：直接 `lerobot-eval`，不需要 WM filter |
| reward | 學出來的 latent 分類器，稀疏 0/1 | `coverage_from_frame`（像素分割 + 校正，worker 每 round 已算好 `cov`），稠密；成功門檻 0.95 | 不用訓練 reward model；ValueNet 的 RM head（latent，MAE 0.034）可當替代 |
| 可信 horizon | 整個 episode（72 步） | PushT 300 步 = 60 幀；SAILOR v1 量到 AUROC 隨想像長度下降（r=8 明顯變差） | 改成**從中途 anchor 分支的短 rollout + critic bootstrap**（MBPO 式），不是從初始狀態跑完整 episode |
| 動作信任域 | 無特別限制 | `inject_noise` 的 fidelity sweep：3σ 一個 round 就壞，>~1σ 不可信（σ≈20.8 px） | 探索雜訊與 policy 漂移都必須留在 NanoWM 訓練分布內 |
| 初始狀態 | 資料集初始觀測 | NanoWM 需要 3 個相隔 5 步的真實 context 幀 | 沿用 `closed_loop_rollout.py` 的 lead-in（真 gym 跑出 3 幀 + 動作）與 `make_reset_anchors.py` |
| policy 大小 | DiffusionMLP 3×512，20 步 | UNet1D down_dims `[512,1024,2048]`（~250M）+ ResNet18，DDPM 100 步 | 需要縮短去噪鏈（§3.2） |

## 3. 提案

### 3.1 整體架構

```
            ┌──────────────── 每個 PPO itr ────────────────┐
anchor pool │ 抽 B 個 anchor（真 gym 的 base-policy 狀態，含 3 個 context 幀）│
(離線、固定) │        │                                                    │
            │        ▼                                                    │
            │  NanoWMVecEnv.reset(anchors)  ── 或 ──  SimVecEnv.reset(anchors)│  ← 同一介面，換 env 即換 arm
            │        │ obs96, agent_xy                                    │
            │        ▼                                                    │
            │  DPPO actor：DDIM K 步、記錄去噪鏈 chain                       │
            │        │ chunk (16×2)，執行 10 步                            │
            │        ▼                                                    │
            │  env.step(chunk) → 2 幀 → reward(cov)、terminated(cov≥0.95)    │
            │        … 共 D 個決策（截斷）…                                  │
            │        ▼                                                    │
            │  critic V(s)，GAE（截斷處 bootstrap V(s_D)）                   │
            │  PPO 更新 actor_ft（+ BC 正則），critic                        │
            └──────────────────────────────────────────────────────────┘
每 N itr：匯出 lerobot checkpoint → gym_pusht 評估（固定 seeds，可與 base 成對比較）
```

### 3.2 Policy 端：把 lerobot DiffusionPolicy 包成 DPPO actor

- **去噪鏈**：沿用 DPPO 影像版的做法（`cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp_img.yaml`：`denoising_steps=100`、`use_ddim=True`、`ddim_steps = ft_denoising_steps = 5`）。本案預設 **DDIM K=10 步、全部 K 步都微調**，訓練時 eta=1（隨機、可算 logprob），評估時 eta=0。
  - 好處：匯出時只要把 `actor_ft` 的 UNet 權重寫回 lerobot checkpoint，並把 config 改成 `noise_scheduler_type: DDIM`、`num_inference_steps: 10`，就能直接用現有 `eval_baseline.py` / `lerobot-eval`。
  - **前提**（P0 要量）：base policy 用 DDIM-10 的成功率不能明顯低於 DDPM-100 的 69.4%。若掉太多，退回「DDPM 100 步、只微調最後 10 步」，此時需要自訂 eval wrapper。
- **凍結 vision encoder**：ResNet18 + spatial softmax 凍結且固定 `eval()`。原因：(1) `crop_is_random=True` 在 train mode 會隨機 crop，rollout 與更新時的 logprob 會對不上；(2) 想像影格與真實影格有分布差異（SAILOR v1 微調對照中加入想像 pair 讓 policy 退步），不讓 encoder 去適應想像影格。只訓練 UNet（`actor_ft`）。
- **reward_horizon = 10**：DiWA 只對執行的動作算 logprob。NanoWM 的 4 幀 window 雖然帶了第 4 組 action（lookahead），但 `NanoWM.forward` 會把 action 往後移一格（第 f 幀拿 `a_{f-1}`，最後一組被丟掉），temporal attention 又是 causal，所以生成幀只看得到本 round 執行的 5 步。在 `--replan-every-rounds 2` 下，一個決策的兩幀分別依賴 `chunk[0:5]` 與 `chunk[5:10]`，reward 只依賴 `chunk[0:10]`。
- **BC 正則**：預設開，係數 0.025（DiWA README）。可加一個「真實資料 DDPM loss」的變體當 ablation（直接把 `local/pusht_imgonly` 的 BC loss 混進來）。

### 3.3 Env 端：NanoWMVecEnv

- **沿用現有 worker 協定**（`imagined_rollout.py closed-loop` + `eval/closed_loop_rollout.py` 的 alignment contract），因為 NanoWM venv（Py3.11、torch 2.7.1、diffusers 0.29.2）與 lerobot venv（Py≥3.12、diffusers 0.38）無法合併，而這條路徑已驗證過對齊（open-loop tracking corr 0.958）。
  - 改動：worker 跨 itr 常駐（新增 `reset_<itr>.npz` 訊息，不必每 itr 重載模型）；work dir 放 `/dev/shm`；worker 另外回傳 latent（給 latent RM 用）。
  - 多 GPU：每張 GPU 一個 worker，各自負責 B/N 個 branch；driver（policy + PPO）在 GPU0。
  - P0 另做一個 spike：只 import NanoWM 推論需要的模組（`models`、`diffusion`、`latent_codecs`、`sampling_utils`）到 lerobot venv。若可行，改成同一行程、GPU tensor 直接傳，省掉 IPC 與 uint8 round-trip。
- **episode 定義**：每個 branch 從 anchor 開始跑 **D=4 個決策（8 round = 40 sim steps）**後截斷；`cov ≥ 0.95` 時 terminated（之後該 branch 遮罩掉）。所有 branch 同步、等長，worker 不需要中途 reset。
- **reward**（預設）：`r_t = (cov_{t+1} − cov_t) + 1[cov_{t+1} ≥ 0.95]`，cov 用 worker 已算的 `coverage_cal`（取該決策第 2 幀）。potential-based shaping，搭配截斷 bootstrap 不改變最佳策略。ablation：只用 success、或改用 ValueNet RM head（latent）。
- **動作信任域**：env 端照 `inject_noise` 的 cap 規則把相對動作限制在 `max_z=3σ` 以內並 clip 到 `AGENT_BOX`（屬於 env dynamics，不影響 PPO 正確性）；另外記錄每個 itr 的「相對動作 / σ」分布，超過 1σ 的比例當作 WM 被利用的警訊。
- **anchors**：base policy 在真 gym_pusht 上跑出的狀態，`t ∈ [10, 250]`（SAILOR v1 T2：89.2% 失敗 episode 的決定時刻在 t>100，v1 只取到 t≤100 是已知缺口）。排除 `wall_artifact` 與已成功的狀態。預設固定一個 2000 個 anchor 的 pool（離線，符合 DiWA 精神）；「每 K itr 用目前 policy 在真 sim 補 anchor」列為 ablation，並記錄用掉的真實步數。

### 3.4 Critic

- 輸入：凍結 policy encoder 的特徵（`_prepare_global_conditioning` 的輸出，detach）+ agent xy（PD replay 精確已知）。MLP 3×256、Mish、residual（DiWA `CriticObs` 規格）。
- 暖啟動選項：SAILOR v1 的 ValueNet（SD-VAE latent + agent state）。它在 stage-1 reset 上 AUROC 0.663，雖然組內排序不行，但當 critic 初值仍可能比隨機好。列為 ablation，不當預設，避免把 v1 的問題帶進來。
- `n_critic_warmup_itr`：DiWA 用 2；本案 critic 從零開始，建議 5。

### 3.5 實驗設計

| arm | env | 用途 |
| --- | --- | --- |
| base | — | 69.4%（1000 ep，seed 100000，na=10） |
| base-DDIM10 | — | 確認匯出格式本身不掉分 |
| **PPO-Sim** | 真 gym_pusht（同 anchors、同 D、同 reward 定義，reward 用真實 coverage） | 上限 / 對照組：PPO 實作與超參數是否有效 |
| **PPO-WM** | NanoWM | 主角（DiWA 式） |
| PPO-WM-RM | NanoWM，reward 用 latent RM | reward 來源 ablation |

- 主要指標：真 gym_pusht 成功率，1000 個成對 episode（`configs/nanowm_pusht_common.yaml` 的 eval 協定），每個 arm ≥3 個 PPO seed；報 arm − base 的成對 bootstrap CI 與 McNemar。
- **WM 被利用的偵測**：每 N itr 同時記錄想像成功率與真實成功率；兩者的差距持續擴大即視為 reward hacking，以真實評估最好的 checkpoint 為準（並報告這個挑選用了多少次真實評估）。
- 解讀：PPO-Sim 沒進步 → 問題在 PPO/policy 端，不在 WM；PPO-Sim 進步但 PPO-WM 沒有 → 差距來自 WM 誤差，接著看 reward 來源與 horizon D 的 ablation。

### 3.6 起始超參數

| 參數 | 值 | 來源 |
| --- | --- | --- |
| B（branch / itr） | 256 | 每 itr 256 × 4 = 1024 個決策；DiWA 是 50 × 36 = 1800、DPPO 影像版 50 × 300 = 15000。受 WM 速度限制 |
| D（決策 / branch） | 4（8 round） | WM fidelity |
| γ / λ | 0.99 / 0.95（每決策） | D 很短，0.999 沒意義 |
| K（DDIM 步數，全部微調） | 10 | DPPO img 用 5；本案 UNet 是原生 DDPM-100 訓練 |
| min sampling / logprob std | 0.1 / 0.1 | DiWA/DPPO；normalized action 的 0.1 ≈ 25.6 px，約 1.2σ，需在 P0 看實際動作分布 |
| clip_ploss_coef / base / rate | 0.01 / 0.001 / 3 | DiWA |
| gamma_denoising | 0.99 | DiWA |
| actor lr / critic lr | 1e-5 / 1e-3 | DiWA（與 `finetune.lr` 相同量級） |
| update_epochs / minibatch / target_kl | 10 / 256 / 1 | DiWA/DPPO |
| BC coeff | 0.025 | DiWA README |
| itr 數 | 150 | DPPO img 151 itr |

### 3.7 成本估計（以 T4 的 2 幀/秒/GPU 粗估，P0 會重量）

- 每 itr 想像幀數：256 × 8 = 2048 幀。
- 現況速度：1 GPU ≈ 17 分/itr；8 GPU ≈ 2.1 分/itr → 150 itr ≈ 5.3 h。
- 可能的加速（P0 量測）：`wm_batch` 24 → 64+（96 GB 卡）、`num_sampling_steps` 30 → 15（需配 fidelity 檢查）、bf16、policy 用 DDIM-10 取代 DDPM-100。若達 8 幀/秒/GPU，150 itr ≈ 1.3 h（8 GPU）。
- PPO-Sim 的 rollout 幾乎免費（gym_pusht 很快），瓶頸只在 policy 前向。

## 4. 階段與 gate

| 階段 | 內容 | 產出 / gate |
| --- | --- | --- |
| **P0** 基礎檢查（~1 天） | (a) base 用 DDIM-10 / eta=0 的成功率（1000 ep）；(b) NanoWM 吞吐 sweep：`wm_batch × num_sampling_steps`，以及 15 步時的 fidelity（`eval/wm_fidelity_ood.py`）；(c) single-venv import spike | (a) 掉 >3 pp 就改用 DDPM-ft-last-10；(b) 定下 sampling steps 與 B |
| **P1** DPPO 移植 + 單元測試 | `rl/dppo_actor.py`（由 DPPO 移植）、匯出工具 | 測試：同一 chain 的 logprob 在 rollout 與更新時一致；`actor_ft == actor` 時匯出模型與 base-DDIM10 成功率相同（±CI）；匯出 → 載入 round-trip |
| **P2** PPO-Sim | 真 gym_pusht、同 anchors / D / reward | gate：3 seeds 平均比 base-DDIM10 高，且成對 CI 不含 0。沒過就先調 PPO，不進 P3 |
| **P3** PPO-WM | NanoWMVecEnv | 主要結果：PPO-WM − base 與 PPO-WM − PPO-Sim |
| **P4** ablation | reward 來源（pixel cov vs latent RM）、D ∈ {2, 4, 6}、BC coeff、critic 暖啟動、anchor 補充 | 只在 P3 有訊號時做 |
| **P5**（選做） | DiWA 原味：在 SD-VAE latent 上重新預訓練 latent policy（policy 不需要 decode，評估時每步 VAE encode 真實影格） | 只有在 decode 成為瓶頸、或影像 encoder 的分布差異證實是主因時才做 |

## 5. 程式配置（預計）

```
examples/nanowm_pusht_aug/rl/
  dppo_actor.py      # 包 lerobot DiffusionPolicy：actor / actor_ft、DDIM 取樣 + chain、get_logprobs、PPO loss（移植自 third_party/dppo，MIT）
  wm_vec_env.py      # NanoWMVecEnv：reset(anchors) / step(chunk) → obs96, agent_xy, reward, terminated, info{cov, rel_sigma}
  sim_vec_env.py     # SimVecEnv：gym_pusht + nwaug.pusht_state.place_body，介面相同
  critic.py
  train_dppo.py      # 主迴圈；config: configs/dppo_nanowm_pusht.yaml（extends nanowm_pusht_common.yaml）
  export.py          # actor_ft → lerobot pretrained_model（DDIM、num_inference_steps=K）
examples/nanowm_pusht_aug/tests/test_dppo_*.py
```

`imagined_rollout.py` 的 worker 只做小改動（常駐 reset 訊息、回傳 latent），不要分叉出第二個 worker。

## 6. 主要風險

| 風險 | 說明 | 對策 |
| --- | --- | --- |
| WM 被利用（reward hacking） | NanoWM 會幻覺 block 姿態（SAILOR v1：想像 rollout 上的 RM AUROC 比真實低 0.071，CI 不含 0）；PPO 會主動找出這種漏洞 | 短 D、動作信任域、BC 正則、想像 vs 真實成功率差距監控、以真實評估選 checkpoint |
| 速度 | 2 幀/秒/GPU 下資料量只有 DiWA 的一小部分 | P0 加速；8 GPU；短 D + critic bootstrap |
| 回報變異大 | SAILOR v1 表 D：同一 branch 重跑接續，成敗一致率只有 0.599 | 靠 critic + GAE + 大 B 降變異；不依賴單次 outcome 當標籤 |
| 想像影格分布差異 | v1 微調對照中加入想像 pair 讓 policy 退步 | 凍結 encoder；PPO-Sim 對照把 WM 誤差和 PPO 問題分開 |
| DDIM 匯出掉分 | lerobot base 是 DDPM-100 訓練 | P0(a) 先量，必要時改 DDPM-ft-last-10 |

## 7. 定稿前要決定的事

1. 是否接受「從中途 anchor 分支的短 rollout」，而不是 DiWA 的「從初始狀態跑完整 episode」。本提案認為以 NanoWM 的 fidelity 只能這樣做。
2. anchor pool 是否完全離線（DiWA 精神），或允許每 K itr 用目前 policy 在真 sim 補 anchor（真實互動數要另外報告）。
3. 預設 reward：`Δcov + success bonus` 或只用 success。
4. P2 的 PPO-Sim gate 沒過時，是否仍要跑 P3。
