# SAILOR v1 — 落地前程式碼重讀（T1）

- 分支：`feat/sailor-v1-value`
- HEAD：`f9eeb5d60d57b14690d4f7014c25771761404000`（`f9eeb5d6`，T0 pre-registration commit；工作樹乾淨）
- 日期：2026-10-01
- 執行者：Sonnet（唯讀；未改任何其他檔、未 commit、未跑 python / 訓練 / 收資料）
- 行號一律以 `rg -n` / `cat -n` 對 HEAD 取得（不是抄 plan）。`git_dirty` 檢查見 §2-18(i)。

## 讀過的檔案

`docs/decisions/sailor_v1.md`、`~/.cursor/plans/sailor_v1_value_model_plan_091ed530.plan.md`（§2、§3）。
`examples/nanowm_pusht_aug/` 下：`critic/{sim_rollout,collect_value_states,label_imagined_branches,train_value_model,value_model,value_gate_report}.py`、
`eval/closed_loop_rollout.py`（全檔）、`eval/make_topband_seeds.py`、`eval/ood_failure_map.py`（`block_wall_gap` / `sample_topband_starts`）、
`nwaug/pusht_state.py`、`imagined_rollout.py`（`encode` 與 `closed_loop`）、`attribution/config_io.py`、`attribution/policy_io.py`（`load_policy`）、
`scripts/run_value_model.sh`、`datagen/provenance.py`、`tests/{conftest,test_pusht_state,test_config_io,test_closed_loop_wait,test_imagined_rollout_helpers,test_cross_venv_imports}.py`。
repo 根：`configs/value_model_topband.yaml`、`configs/nanowm_pusht_common.yaml`。
外部：`.venv/lib/python3.12/site-packages/gym_pusht/envs/pusht.py`、`gym_pusht/__init__.py`（確認 `terminated`、`max_episode_steps=300`、reset 分佈）。

---

## 0. 結論速覽

- plan §2 的 9 項已知問題：**8 項確認、1 項部分錯誤**（P5：`place_seed` 其實已經在呼叫 `nwaug.pusht_state.place_body`，`FI/LEAD` 行號也差 2 行）。
- 新發現會「讓 T3 直接當掉」的有 2 個：`make_starts` 在 `band.n=0` 時 `np.concatenate` 維度不合（§2-18a）、`n=0` 時 config 仍需要 `band.seed/min_wall_gap`。
- plan / pre-registration 與現行程式不一致的地方：
  1. `Normalizer` 目前是在 **fit + es** 上擬合（`train_value_model.py:532,557`），不是 pre-reg §4 寫的「只在 fit split」。
  2. plan §2 說「`sim_rollout` 要把 coverage / CoG 網格延到 300」——可行，但**更好的做法是記錄 per-step coverage**（`info["coverage"]` 每步本來就算好了），否則 `T_succ` 不在 5 步網格上的 `cov_end` 拿不到（P1 / P16）。
  3. V head 改 soft `v_target` 之後，G1 的 ECE、G4 的 `|V−y|`、shuffled control、`--overfit-test` 的 `bce<0.05` 門檻都會變得**語意不對**（N14）。
  4. 停損用的 test anchors 與 training anchors 共用 `seeds_stage2.npz` / `stage2/labels.npz` 的硬編碼路徑（N18c/d）：若不改，test anchors 會被 `train_value_model.py` 讀進訓練。
- 本機 checkout 沒有 `outputs/`（`ckpt.base` = `outputs/train/pusht_imgonly/obs1_s100000/...` 不存在），T3 smoke / T4 需要使用者先提供 base ckpt 位置（N18j）。

---

## 1. 已知問題確認（plan §2）

### P1 `rollout_batch` 的網格只記到 `frame_cutoff_step=110`（**確認**）

- 現況：
  - `critic/sim_rollout.py:60` `frame_cutoff_step: int = 110`
  - `:154-156` `grid_steps = list(range(0, frame_cutoff_step + 1, frame_grid_stride))`，`grid_idx_of_step` 只含這些步
  - `:158-162` `grid_cog / grid_agent_pos / grid_agent_vel / grid_coverage / grid_frames` 全部以同一個 `G=len(grid_steps)` 配置
  - `:236-241` 只有 `(t+1) in grid_idx_of_step` 才記錄 `info["coverage"]`
  - `:232,242` 224px env 只在 `(t+1) <= frame_cutoff_step` 才 step / 抓影格
  - stage 1 呼叫點 `collect_value_states.py:231` `frame_cutoff_step=110`；stage 2 接續 `label_imagined_branches.py:227-240` 沒傳 `frame_cutoff_step` → 預設 110，但接續長度 `episode_steps=B=200`，所以 **接續的 coverage 也只記到 continuation 的第 110 步**。
  - 另外：episode 成功後 `done[i]=True`（`:247-248`），`:226-228` 會 `continue`，所以成功之後的網格項目維持 `np.zeros`（cog/agent/coverage 都是 0，不是最後值）。
- 影響：t=100 的 query 需要 `coverage(t+min(T,200))`，最遠到 step 300，現在拿不到。且只延長 5 步網格並不夠：`T_succ` 通常不是 5 的倍數，`T≤200` 時 `Φ_end = coverage(T_succ)` 需要成功那一步的 coverage（> 0.95），網格上沒有。
- 處理（T3，`critic/sim_rollout.py::rollout_batch`）：
  1. 把「影格網格」與「追蹤網格」拆開：`frame_cutoff_step`（=100，只管 224px env、`grid_frames`、`grid_steps`，保持 `lockstep224` 的語意）；新增 `cov_per_step`（`(nb, episode_steps+1)` float32，index `s` = 已執行 s 步後的 `info["coverage"]`，index 0 = `_get_coverage()`）。成功後 forward-fill 成最後一個 coverage（或直接填成功那一步的 coverage），不要留 0。記憶體微不足道（2000×301×4B ≈ 2.4MB）。
  2. `grid_cog / grid_agent_*` 仍只需要到 t≤100（query 與 oracle 特徵只用到 t≤100），**不需要**延到 300；如 Opus 仍想保留 pre-reg §2 表格「CoG 網格到 300」，成本也很小（`(nb,61,5)` f64），但 `_load_stage1` / encode 不能因此多出影格。
  3. 回傳新增 `cov_per_step`；保留舊 key（`grid_*`、`success_step`）不改語意，讓 `label_imagined_branches.py` 與 T2 腳本不壞。
  4. `prepared_envs` 路徑同一套程式碼（沒有分支），`cov_per_step[:,0]` = 接續起點的 coverage，自動涵蓋 stage 2。
  - 測試：`tests/test_v_target.py` 之外，加一個小測試（需 gym_pusht，`importorskip`）：`cov_per_step` 長度、`success_step` 對應位置 > 0.95、成功後 forward-fill。

### P2 stage 2 每 round（5 步）重取樣 chunk，只執行 `chunks[:, :fi]`（**確認**）

- 現況（`eval/closed_loop_rollout.py`）：
  - `:309` `for r in range(args.h_max):`
  - `:311-313` `chunks = sample_chunks(policy, pre, post, cur96[::k] if r == 0 else cur96, k if r == 0 else 1, dev)` — **每個 round 都重新取樣**
  - `:314-317` `if args.inject_sigma > 0: chunks = inject_noise(chunks, pos, vel, ...)`（每個 round 都加，沒有「只在 query 前」的閘門）
  - `:322-323` `past = stream[b, k0 : k0 + (H - 1) * fi + fi]`；`look = chunks[b, : need - len(past)]`
  - `:343-344` `ex = chunks[:, :fi]`; `exec_stream[:, r*fi:(r+1)*fi] = ex`
- 影響：與 stage 1 / eval 的 `n_action_steps=10` 節拍不一致（每 5 步重規劃，比真實 policy 更「反應式」）。stage 2 的 r=2 query 也不再是「執行完一個候選 chunk 之後」的狀態。
- 處理（T3，`closed_loop_rollout.py`）：新增 `--replan-every-rounds R`（預設 1，行為與現在逐位元相同）。細節見 N15。

### P3 stage 1 只擾動 `perturb_p == t` 那一個 chunk；query 保留規則 `t >= p+10`（**確認**）

- 現況：
  - `critic/sim_rollout.py:198-199` `if t % na == 0: active = [i for i in range(nb) if (not done[i]) and sigma[i] > 0 and perturb_p[i] == t]`
  - `critic/collect_value_states.py:128` `if sigma[e] > 0 and t < perturb_p[e] + 10: continue`
  - `:78-88` `draw_perturbations`：`sigma = rng.choice(sigmas, size=n, p=probs)`，`p_choices = np.arange(0, 101, 10)`（`:86`）
  - config 是 per-episode `sigmas: [0.0, 0.5, 1.0]`、`noise_hold: 5`（`configs/value_model_topband.yaml:18`）
- 影響：與 pre-reg §2 的「t < p 皆加噪聲、只留 t ≥ p」相反方向（舊版是「噪聲之後 10 步丟掉」）。
- 處理（T3）：
  - `rollout_batch`：新增 `perturb_mode: "point" | "prefix"`（預設 `"point"`，保留 T2 與舊 config）。`prefix`：`active = [i ... if sigma[i] > 0 and t < perturb_p[i]]`。其餘（`pending[i] = (t, noised[0])`、`:222-224` 覆蓋 `act[i]`、`:252-254` 清除 pending）不變。`perturb_p ≥ episode_steps` 就等於「整條加噪聲」，所以 T2(a) 的備案**不需要另一套程式**。
  - `collect_value_states.py`：`build_queries` 的條件改 `t < perturb_p[e]`（即保留 `t >= p`）。建議抽成純函式 `keep_query(t, p, sigma)` 放到新的 numpy-only 模組 `critic/labels.py`，供 `tests/test_prefix_perturb.py` 直接測而不必 import `sim_rollout`（見 §3）。
  - config：`perturb: {sigmas: [0.1], probs: [1.0], noise_hold: 1, mode: prefix, max_z: 3.0, seed: <新>}`。`draw_perturbations` 會照常消耗 `rng`（`rng.choice` with `p`）；`p_choices` 目前寫死在 `:86`，與 pre-reg 的 {0,…,100} 一致，可保留。
  - 副作用提醒：`p` 均勻 over 11 個值時，query t 的保留機率 = (t/10+1)/11（t=10: 2/11 … t=100: 11/11），平均每 episode 約 5.9 個 query、2000 episodes ≈ 11.8k 列（尚未扣掉成功後與 wall artifact），**晚期 t 較多**。這是 pre-reg 設計的直接結果，T4 後在 class balance 報告中要按 t 分層看 `y` 比例。

### P4 stage 2 接續只回傳 `y`（**確認**）

- 現況：`critic/label_imagined_branches.py:243-246`
  ```
  s = int(res["success_step"][j])
  y[i] = s != -1 and s <= B
  wall_gap[i] = min(wall_gap[i], float(res["wall_gap_min"][j]))
  ```
  輸出欄位列表 `:183-197`；儲存 `:298-318`；只有 `y / pre_success / sim_cov`。`rollout_batch(episode_steps=B)` 讓接續恰好跑 B 步，所以 `s <= B` 永遠成立（冗餘）。
- 影響：缺 `success_step`（T）與 `cov_end`，無法套 `v_target`。
- 處理（T3）：`res["success_step"][j]` 已是「接續起點起算」的 T；`cov_end = res["cov_per_step"][j, min(T, B)]`（T=-1 → `B`）。對 replay 期間就成功的 row（`terminated=True`，`:222` `alive` 排除它、不進接續）：`T=0`、`v_target=1`、`cov_end = sim_cov`。新增欄位 `success_step`（= T；-1 表示接續內未成功）、`cov_end`、`v_target`（呼叫 `critic/labels.py::v_target(T, cov_end, B, gamma, beta=0)`）。命名見 §4 決策 D3。

### P5 `label_imagined_branches.py` 寫死 `FI/LEAD`、自己實作 body-origin 擺放（**部分錯誤**）

- 現況：
  - `FI = 5`、`LEAD = 10`、`TOL = 1e-6` 在 `critic/label_imagined_branches.py:72-74`（plan 寫 :70-71，差 2 行）。
  - `place_seed` 在 `:81-84`：
    ```
    def place_seed(env, seed_index: int, st: np.ndarray) -> None:
        env.reset(seed=9000 + seed_index)
        place_body(env.unwrapped, st)
    ```
    `place_body` 是 `from nwaug.pusht_state import load_seeds, place_body, read_body`（`:67`）。**它已經用 `nwaug.pusht_state.place_body`，沒有自己的 body-origin 實作**；本地只剩 `reset(seed=9000+idx)` 包裝（與 driver `closed_loop_rollout.py:228-229` 同樣的兩行重複）。plan 的這個假設是錯的。
- 影響：
  - `FI/LEAD` 不是 band 專屬；它們等於 NanoWM 幾何常數（`--fi 5`、`(H-1)*fi=10`，`closed_loop_rollout.py:147,149,189`），sailor_v1 沒改這兩個值，所以**硬編碼對本計劃無害**，只是脆弱。
  - `replay_row`（`:87-124`）用 `FI` 算 `r*FI` 步 replay，這個 5 與 `--replan-every-rounds` 無關（round 仍是 5 步）。
- 處理（T3，可選、低風險）：(a) driver 在 branch npz 多存 `fi`、`hist`、`replan_every_rounds` 三個 int；label 腳本 `assert` 它們等於 `FI / H / 預期值` 再往下跑；(b) 把 `place_seed` 收成 `nwaug/pusht_state.py::place_seed(env, idx, st)`，driver 與 label 共用（需同步改 `tests/test_pusht_state.py` 的 `legacy_place_seed` 對照，若要做請單獨 commit）。不做也不影響正確性，建議只做 (a)。

### P6 anchor 產生器只會產生上牆帶（**確認**）

- 現況：`eval/make_topband_seeds.py:83` `cog = sample_topband_starts(spec["n"], spec["seed"], min_gap)`；`:75` `jobs = [(args.section, node, node["min_wall_gap"], args.out_name)]`（`min_wall_gap` 為必要 key）；`:92` `assert ((cog[:, 3] >= band[0]) & (cog[:, 3] < band[1])).all()`（上牆帶 y 檢查）。`cog_to_body`（`:41-46`）與 0.5 px round-trip（`:80-91`）可重用。
- 影響：不能直接 `--section stage2.anchors` 產 reset anchors：(1) 來源是 rollout 不是取樣；(2) `:92` 的 band 斷言會失敗；(3) reset rollout 需要 policy/torch，`make_topband_seeds.py` 目前是純 numpy+gym。
- 處理（T3）：新腳本 `critic/make_reset_anchors.py`（在 `critic/`，可 import `sim_rollout` / `load_policy`）：
  1. 對 `i in range(n_src)`：`gym.make("gym_pusht/PushT-v0").reset(seed=base+i)` → `sim_state` 取得 CoG start（與 `collect_value_states.make_starts:66-71` 同一招）。
  2. `rollout_batch(..., lockstep224=False, sigma/perturb_p 依決策 D2, seed=cfg.seed_namespace.sailor_v1_stage2_anchor, batch_start=...)`，取 `grid_cog[:, t//5]`。
  3. 每個來源 episode **只抽一個** anchor（t 從 {10,…,100} 中抽，排除 `success_step != -1 and success_step <= t`、`wall_artifact`）；避免同一來源 episode 的多個 anchor 被分到 val / fit 兩邊（group leakage；`seed_id` 是 anchor 列號）。
  4. `cog_to_body`（`from make_topband_seeds import cog_to_body`）+ `place_body` 讀回的 0.5 px round-trip 檢查照搬；不要套 topband y 斷言。`save_seeds(out / "seeds_stage2.npz", body=body, cog=cog, t=t_anchor, src_episode=ep)`（`save_seeds` 接受 `**extra`）。
  5. `--smoke`：輸出到 `out_dir/smoke/`（見 N18c，否則 smoke 會覆寫正式的 `seeds_stage2.npz`）。
  - 已知限制：seeds 檔只存 5 維 `[agent x, y, block x, y, angle]`，anchor 的 **agent 速度丟失**；driver 以 `reset(seed=9000+i)` 之後 `place_body`（沒有 `zero_agent_vel`）起算，agent 速度≈0。所以 stage 2 的 r=0 row 的 agent 速度分佈與 stage 1 同 t 的 row 不同（見 §4 R4）。

### P7 `BUCKETS`、G1、`real_vs_imagined_block` 假設 band 存在（**確認**）

- 現況：
  - `critic/train_value_model.py:69-71` `BUCKETS = ("band","reset","stage2")`、`BUCKET_OFFSET = {"band": 0, "reset": 1, "stage2": 2}`
  - `critic/value_gate_report.py:193-194` `# ---- G1: stage-1 band rows ----` / `m1 = gm & (d["stage"] == 1) & (d["source"] == "band")`
  - `:378-421` `real_vs_imagined_block`：`m_real_band`（`:383`）是基準，`bootstrap_diff_ci_unpaired` 的 `imagined − real_band`（`:393-421`）
  - `:454-455` 可靠度圖：`m1 if m1.any() else gm`（band 空時自動退回全部 gate rows — 會默默改變圖的意義）
  - `:504-510` markdown 文字寫死 "stage-1 band"
- 影響：`source=="band"` 為空時，G1 的 `roc_auc` 回傳 NaN、`g1_pass = False`（`np.isfinite(auc1)` 為 False），**沒有例外但 G1 永遠顯示 FAIL / NaN**；`real_vs_imagined` 的 unpaired diff 全 NaN。
- 處理（T5）：G1 改為 `source == "reset"` 的 stage-1 rows；`real_vs_imagined_block` 以 `real_reset` 為基準；`BUCKETS` 維持三個名字但在 sailor config 下 `band` 自然為空（`compute_split` / `class_balance` 已能處理，見 N12）。markdown / 圖題同步改。建議新增 `gate.g1_source: reset` 之類的 config key，讓舊 config 行為不變。

### P8 `sim_cov` 只是診斷；loader 讀 `cov`（**確認**）

- 現況：`critic/label_imagined_branches.py:38`（docstring「diagnostic」）、`:121` `"sim_cov": float(u._get_coverage())`、`:255` `out["cov"].append(float(br["coverage"][r - 1]) if r >= 1 else rp[i]["cov_anchor"])`、`:257` `out["sim_cov"].append(...)`、`:304-305` 儲存；`critic/train_value_model.py:179` `"cov": d["cov"].astype(np.float32)`（stage 2）；`:157` `cols["cov"].append(d["q_coverage"]...)`（stage 1）；RM loss `:302-304`。
- 影響：stage 2 的 RM 標籤現在是 NanoWM 想像影格的 coverage（`coverage_cal`），不是真實 sim coverage。
- 處理（T3，label 腳本）：輸出 `cov = sim_cov`、`cov_source = "sim"`，另存 `cov_imagined`（r≥1：`br["coverage"][r-1]`；r=0：與 sim 相同或 NaN）。`train_value_model.py::_load_stage2` 不用改。**注意** label 腳本 `:279-295` 的「stored cov vs worker `frames_r.npz` cov」交叉檢查目前用 `out["cov"]`（`:281`），改完 `cov` 之後要改成比較 `cov_imagined`，否則這個 sanity check 會變成比較 sim vs imagined 而誤報差異。

### P9 `episode_steps=310` 與 gym 300 上限（**確認**，並有一處 docstring 是錯的）

- 現況：
  - `configs/value_model_topband.yaml:16` `episode_steps: 310`
  - `critic/sim_rollout.py:112` `max_ep_steps = episode_steps + 10`；env 以 `max_episode_steps=max_ep_steps` 建立（`:125,137`），覆蓋 gym 註冊值 300（`gym_pusht/__init__.py:6`）。所以 310 步是真的可達的：成功發生在 step 301–310 的 episode 會被算成成功，而 eval protocol（`configs/nanowm_pusht_common.yaml:24` `episode_length: 300`；`eval/band_eval.py:39` `--max-steps 300`）不會。
  - `critic/collect_value_states.py:6` docstring：「Episodes run the full 310 steps regardless of success」——**不對**：gym_pusht `step` 在成功時 `terminated = is_success`（`gym_pusht/envs/pusht.py:258`），`rollout_batch` 在 `term` 時 `done[i]=True`（`sim_rollout.py:247-248`）且不再 step。成功後的 query 本來就因 `s < t` 被丟掉（`collect_value_states.py:126-127`）。
- 影響：新 config `stage1.episode_steps: 300` 即可；`+10` margin 仍然有用（避免 TimeLimit 在 loop 內截斷），loop 由 `for t in range(episode_steps)`（`:192`）控制，所以實際只跑 300 步。stage 1 的 query t ≤ 100 → `t+200 ≤ 300` 剛好完整。
- 處理（T3）：config 設 300；`collect_value_states.py` docstring 更正；`sim_rollout.py` 的 +10 margin 保持（加註解說明）。

---

## 2. 新發現 / 待確認項目（plan 10–18）

### N10 `imagined_rollout.py encode` 與 `frames_224` 網格長度（**不依賴 23；可以把影格網格縮到 100（21 格）**）

- 現況：
  - `imagined_rollout.py:465-503` `encode`：`frames = d[args.key]`（`:482`），只檢查 `frames.shape[-3:] == (224,224,3)`（`:483`），`lead_shape = frames.shape[:-3]`（`:485`），`flat = frames.reshape((-1,)+frames.shape[-3:])[:, None]`（`:486`），輸出 `lat.reshape(lead_shape + (Cz, Sz, Sz))`（`:493`）。**對 episode 數 E 與每 episode 影格數 G 沒有任何假設**。
  - `latents_*.npz` 版面：`latents (E, G, 4, 32, 32) f16`、`seed`、`vae_precision`、`scaling_factor`、`img_size`（`:496-502`）。
  - `critic/train_value_model.py:135` 只是註解 `# (E,23,4,32,32) f16`；`:147-151` 用 `gsteps = q_fs // 5`（`:147`，並 assert 在 5 步網格上 `:148`）做 `stack[i] = lat[ep_local[int(q_ep[i])], gsteps[i]]`，`gsteps[i]` 是 (3,) 索引。只要 `max(q_frame_steps)//5 < G` 就正確。
- 結論：新 `QUERY_TS = 10..100`：最大 `gsteps = 20`，`G=21`（影格 0..100 步，stride 5）就夠；**把影格網格截在 100 並把 coverage 延到 300 不會弄壞 `encode` 與 `_load_stage1`**。只有一個前提：`sim_rollout.rollout_batch` 必須把「影格網格」與「coverage 追蹤」拆開（P1 處理 1）；如果只是把 `frame_cutoff_step` 拉到 300，`grid_frames` 會變 `(nb,61,224,224,3)`（每 100-episode shard ≈ 0.92 GB uncompressed，encode 時 `encode_context_frames` 的 float32 + resize 暫存約 3×），而且會編碼 40 格沒人用的影格。
- 處理（T3）：`collect_value_states.py:231` `frame_cutoff_step=100`，shard 內 `frames_224` 為 `(E,21,224,224,3)`；`train_value_model.py:135` 註解改為 `(E,21,...)`，並在 `_load_stage1` 加 `assert lat.shape[1] > gsteps.max()` 與 `assert lat.shape[0] == len(episodes)`（見 N18b 的 stale-latents 風險）。

### N11 `value_model.Normalizer` 的擬合範圍（**與 pre-reg 不一致：目前是 fit + es，不是只 fit**）

- 現況：
  - `Normalizer.fit(latents, agent_states)` 定義於 `critic/value_model.py:64-72`：per-channel `lat_mean/lat_std`（跨 3 幀、32×32）與 `vel_std`（agent 速度 std）。位置項固定 `/512`（`POS_SCALE`），沒有擬合。
  - 呼叫點 `critic/train_value_model.py:556-557`：`normalizer.fit(rows["latent_stack"][train_idx], rows["agent_state"][train_idx])`；`:532` `train_idx = np.nonzero(~val_mask)[0]  # fit + es combined -- normalizer stats only`。
  - ValueEnsemble 與 OracleEnsemble **共用同一個 `normalizer` 物件**（`:630-631`；OracleEnsemble 的 docstring 寫「fit independently」，實際不是；Oracle 只用到 `vel_std`）。
  - 另：擬合範圍含 stage 1 與 stage 2 的全部列（含 stage-2 `wall_artifact` 列）。
- 影響：val 沒有漏（`~val_mask`），但 es 進了統計量。pre-reg §4 明文「Normalizer 只在 fit split 擬合」。影響量極小（統計量），但預先註冊文件要嘛照做、要嘛改文字。
- 處理（T5，`train_value_model.py:556-557`）：改成 `fit_idx`（若 Opus 要求精確遵守 pre-reg；`fit_idx` 是 bootstrap 前的 fit 列）。`Normalizer` 類本身不用改。或在 pre-reg 修訂紀錄註明「train = fit+es，val 排除」。見 §4 決策 D6。

### N12 `compute_split` 在 band bucket 為空時的行為（**正常；沒有除以零或空陣列問題**）

- 現況：`critic/train_value_model.py:220-232`：`for bucket, off in BUCKET_OFFSET.items(): bmask = rows["source"] == bucket; if not bmask.any(): info[bucket] = {...全 0 / []...}; continue`。非空 bucket：`n_val = max(1, round(len(groups)*val_frac)) if len(groups) > 1 else 0`（`:237`）、`n_es = max(1, round(len(train_groups)*es_frac)) if len(train_groups) > 1 else 0`（`:244`）——單一 group 時不會出現空 val / es 造成的除法問題。`composite_key`（`:262-265`）只對實際出現的 `source` 取 `BUCKET_OFFSET`，band 為空不影響。`class_balance`（`:100-105`）對空 bucket 回傳 `nan`（`float(y[m].mean()) if m.any() else float("nan")`）。`load_table` 的 `set(unique source) <= set(BUCKETS)`（`:203`）允許子集。
- 預期 bucket 大小：reset 2000 groups → val 400 / es 160 / fit 1440；stage2 250 anchors → val 50 / es 20 / fit 180。**stage-2 val 只有 50 個 anchor**（r≥2 的列約 50×32≈1600，但 group 只有 50），表 A 的 b/d 格 CI 會寬；G6 用 test anchors n=100 也是同量級。
- `GROUP_MOD = 10_000_000`（`:71`）對 episode ≤ 2000、seed_id ≤ 250 綽綽有餘。
- 處理：`compute_split` 不用改。**真正會在 band=0 時出問題的是上游與下游**：`collect_value_states.make_starts`（N18a）與 `value_gate_report` 的 G1（P7）。T3 應新增一個小單元測試：給 `rows` 只含 `reset` / `stage2` 兩種 source，檢查 `compute_split` 回傳的 `info["band"]["n_groups"] == 0` 且 val/es/fit 互斥。

### N13 `config_io.load_config` 與新 config（**不需要新 key；`~` 會展開；會把設定拷貝到 `out_dir`**）

- 現況（`attribution/config_io.py`）：
  - `extends:` 解析 `:34-39`（相對於該 config 自身目錄、偵測循環、遞迴 `_merge`：dict 遞迴覆蓋，list / scalar 取代）。新 `configs/sailor_v1_reset.yaml` 只要寫 `extends: nanowm_pusht_common.yaml`，**不需要新 key**。
  - `EXPAND_KEYS = ("ckpt","pool_index","oracle_npz","out_dir","nanowm_python","dataset_dir","lerobot_home")`（`:17`），只展開**頂層**且型別是 `str` 的值（`:40-42`）。`out_dir: ~/datastorage/nanowm_pusht_aug/value_model/sailor_v1` 會被 `expanduser()`。注意：`ckpt` 是 dict（`ckpt.base` 等），所以 `ckpt` 底下的 `~` 不會被展開（現有值都是相對路徑，無影響）；**新增任何巢狀路徑 key（例如 `test_anchors.dir`）不會被展開**，請全部由頂層 `out_dir` 推導。
  - `:44-46` `out.mkdir(parents=True, exist_ok=True)`（即使 `out_dir_copy=False` 也會 mkdir，shell 的 `y()` 也會建立目錄）；`:47-58` 拷貝解析後的 config 到 `out/config.yaml`，若內容改變則把舊檔改名為 `config.<UTC ts>.yaml`；`:59-70` 對每次 `load_config(..., out_dir_copy=True)` 追加一行 `config_provenance.jsonl`（`provenance_entry(None, config=..., config_sha1=...)`：含 argv、git sha / dirty / branch）。
  - `--smoke` 的 `out_dir` 仍是根目錄，所以 smoke 也會寫根目錄的 `config.yaml` / `config_provenance.jsonl`。
- 對本計劃的好處：T5 在 `gate:` 填門檻後 config 內容改變 → 舊版自動變成 `config.<ts>.yaml`，是「門檻在看 val 之前寫入」的稽核證據；建議 T5 commit 後、第一次跑 `train` 之前先跑一次任意腳本讓快照落地。
- 必須在 sailor config 重寫的 key（因為 `extends` 只繼承 common，不繼承 topband）：`ckpt_key`、`label{B,na,wall_pen_px}`、`stack`、`stage1{reset{n,seed}, band{n:0,seed,min_wall_gap}（見 N18a）, episode_steps, perturb{...}, batch, shard_episodes}`、`stage2{anchors{n,seed}, k, h_max, inject_sigma, query_rounds, replan_every_rounds}`、`encode{batch,seed}`、`model`、`train`、`gate`、`smoke`。shell 會用 `y encode.seed` 等讀值，少一個就 `KeyError`。`stage1.query_every`（topband yaml:17）在全 repo 沒有任何讀取點（`rg query_every` 無結果）——是死 key，`QUERY_TS` 寫死在 `collect_value_states.py:54`；新 config 可直接不寫，或讓 `QUERY_TS` 從 config 讀。
- 另一個陷阱：common 裡也有 `train:`（`nanowm_pusht_common.yaml:74-80`：`steps/batch_size/horizon/n_action_steps/n_obs_steps/save_freq`）與 `eval:`；`_merge` 會把 sailor 的 `train: {lr, batch, ...}` 遞迴併進去，`cfg["train"]`（`train_value_model.py:526`）因此多出無害的額外 key。**新 config 不要用 `batch_size` / `horizon` / `n_action_steps` 當 train 下的新 key**，避免語意混淆。

### N14 `train_value_model.compute_loss` 的 V loss（**BCE-with-logits 可直接吃 soft target；`y` 必須保留**）

- 現況：`critic/train_value_model.py:288-316`
  ```
  y_np = rows["y"][idx] if y_override is None else y_override[idx]      # :300
  y = torch.as_tensor(y_np.astype(np.float32), device=device)
  bce = F.binary_cross_entropy_with_logits(v_logit, y)                 # :303
  rm_mse = F.mse_loss(torch.sigmoid(rm_logit), cov)
  loss = bce + w_reward * rm_mse + w_pose * pose_mse                   # :311
  ```
  `F.binary_cross_entropy_with_logits` 接受 [0,1] 內的 float target；`v_target ∈ {0} ∪ [γ^200≈0.367, 1]` 合法。**不需要改網路結構**（head 已是 1 個 logit，`value_model.py:114`）。
- `y` 的其他用途（改 loss 目標後仍要保留二元 `y` 欄位）：
  - `train_value_model.py:450` 每個 member 的 val AUROC；`:605-625` shuffled control（`y_override`）；`:625` `roc_auc(rows["y"][val_idx], shuffled_val_v)`；`:656` `val_pred["y"]`；`:674-675` metric-code sanity；`:692-695` `class_balance`。
  - `value_gate_report.py`：G1–G3 AUROC（`:194-247`）、`ece`（`:36`，以二元 y 為標籤）、G4 `absresid = |v_mean_all − y|`（`:261`）、`_rvi_cell`（`:350`）。
- 會變語意 / 會壞的地方（必須在 T5 處理）：
  1. **ECE（G1）**：V 訓練目標變成 `E[γ^T·1[T≤B]]`，本質 < `P(T≤B)`；拿 V 對二元 y 算 ECE 會系統性偏差、永遠難過門檻。建議 ECE 對 `v_target` 算（bin 內 `mean(v_target)` vs `mean(V)`），`y` 只用於 AUROC。見 §4 決策 D4。
  2. **G4**：`|V − y|` 改為 `|V − v_target|`（`value_gate_report.py:261`）。
  3. **shuffled control**（`:605-622`）：需 shuffle `v_target`（fit / es 兩池各自獨立，照現有寫法）；`compute_loss` 的 `y_override` 改名 / 改語意為 `target_override`。
  4. **`--overfit-test`**（`:482` `if parts["bce"] < 0.05`）：soft target 的 BCE 下界是 target 的熵，不會低於 0.05 → 會永遠 FAIL。overfit test 應該繼續以二元 `y` 當目標（或改成與目標熵比較）。
  5. **val_predictions.npz**（`:648-667`）新增欄位：`v_target`、`T`（或 `success_step`）、`cov_end`、`t`（stage 1 的 query 時間；現在 `rows` 沒有 `t`，只有 `frame_steps[:,2]`）、`anchor_id`（= stage 2 `seed_id`）、`cand`。`_load_stage2` 已有 `seed_id/r/sigma`，但**沒有載入 `cand`**（label npz 有，`label_imagined_branches.py:312`）；G6 需要它。另外 OracleNet 的 RM 輸出目前被丟掉（`:645` `ov_logit, _, _ = oe._forward_all(...)`），表 A 的 OracleNet「RM MAE」需要 `oracle_rm_mean_all`，還有 `oracle_v_std`。
  6. `_load_stage1`（`:109-166`）需載入新欄位 `q_T_succ / q_cov_end / q_v_target / q_perturb_p`；`_load_stage2` 載入 `v_target / success_step / cov_end / cand`；`load_table` 的 `np.concatenate` 要在兩邊欄位集合一致（`:192-194` `rows = {k: np.concatenate([s1[k], s2[k]]) for k in s1}` 以 s1 的 key 為準，少一邊會 KeyError）。

### N15 `closed_loop_rollout.py`：round 迴圈、NanoWM 視窗、`inject_noise` 一致性、`--inject-sigma` 閘門

#### 15.1 視窗與 `stream` bookkeeping（現況）

- `stream`：`:259` `stream = np.repeat(lead_acts, k, axis=0)`（`(bs, lead=10, 2)`），每 round 在 `:354` append 5 個已執行的目標；`apos / avel`（`:262-263`、`:355-356`）是對應的 PD 軌跡；`pos / vel`（`:260-261`）是當前 agent 狀態；`agent_state_hist[:, r+1]`（`:353`）= round r 結束後的 `[pos,vel]`。
- round r 的視窗（`:318-328`）：`need = nf*fi = 20`；`k0 = r*fi`；`past = stream[b, k0 : k0+10]`（10 個已執行動作 = 視窗的 group 0、1）；`look = chunks[b, : need - len(past)]`（10 個 lookahead = group 2、3）；`rel = abs_targets_to_rel(seg, apos[b,k0], avel[b,k0])`；`wins[b] = rel_to_nanowm(rel[:need])`。Group 2 = 這個 round 會執行的 `chunk[0:5]`，group 3 = 未執行的 `chunk[5:10]`（lookahead）。
- 執行（`:342-356`）：`ex = chunks[:, :fi]`、`exec_stream[:, r*fi:(r+1)*fi] = ex`，PD 推進 5 步。
- `sample_chunks`（`:96-108`）直接呼叫 `policy.diffusion.conditional_sample`，回傳**整個 horizon**（不切到 n_action_steps）——所以 `chunks.shape[1] = horizon`。

#### 15.2 `--replan-every-rounds R` 需要的改動（R=2）

只在 `r % R == 0` 取樣（並加噪聲）；其餘 round 沿用同一 chunk，以 `off = (r % R) * fi` 為偏移：

```python
if r % R == 0:
    cur_chunks = sample_chunks(policy, pre, post, cur96[::k] if r == 0 else cur96, k if r == 0 else 1, dev)
    if args.inject_sigma > 0:
        cur_chunks = inject_noise(cur_chunks, pos, vel, args.inject_sigma, args.max_z, args.noise_hold, noise_rng)
off = (r % R) * fi
look = cur_chunks[b, off : off + need - len(past)]     # R=1 -> off=0, 與現在逐位元相同
ex = cur_chunks[:, off : off + fi]                     # 取代 :343 的 chunks[:, :fi]
```

- **未執行的後半（`chunk[5:10]`）在下一個 round 被執行**：奇數 round 的 `ex = cur_chunks[:, 5:10]`，`exec_stream` 對應位置寫入同一份（已加噪聲）的 chunk；label 腳本 `replay_row`（`:106-107`）重播 `exec_abs_targets[: r*5]` 所以自動一致。
- **奇數 round 的 lookahead（group 3）需要 chunk 的 [10:15]**：`off + 10 = 5 + 10 = 15`，也就是 `cur_chunks[b, 5:15]`，需 `horizon ≥ 15`。`configs/nanowm_pusht_common.yaml:77` 的 `train.horizon: 16`（from-scratch 設定）足夠，但 **ckpt 的真實 horizon 我在本機讀不到**（N18j），請在 driver 加 `assert cur_chunks.shape[1] >= off + need - len(past)`。這個 lookahead 在真實 policy 中是「下一個 chunk 的前 5 步」（要到下個 frame 生成後才會被取樣），此處改用 policy 對 horizon 尾端的預測來近似，是設計上的一個近似（見 §4 D5）；group 2（這個 round 會真的執行的 5 步）與 R=1 時完全相同。
- `cur96`（`:357`）每 round 仍要更新（偶數 round 取樣用）；奇數 round 用不到，但不影響。
- `h_max` 須為 R 的倍數（8 與 R=2 OK），否則最後一個 chunk 不完整。
- 測試：抽出純函式 `chunk_slices(r, R, fi, need, past_len) -> (sample_now: bool, off: int, look: slice, exec: slice)`，`tests/test_replan_cadence.py` 直接測：R=1 → 與舊行為相同；R=2 → 偶數 round 取樣、奇數 round `off=5`、`exec` 連續覆蓋 `[0:10]` 不重複不遺漏。

#### 15.3 與 `inject_noise` 的交互（noise 一致性）

- `inject_noise`（`:116-137`）：對**整個 horizon**（hz=16）一次抽 `z = rng.standard_normal((bs, ceil(hz/hold), 2))`（`:129`），沿 chunk 用 `ac.pd_step` 追蹤 agent（`:131-136`），輸出 `out[b,t] = clip(pp + rel, AGENT_BOX)`。`noise_hold=1` → 每一步獨立 z（16 個）。
- 因為 noise 是在**取樣那一刻對整條 chunk（含尾端）一次做完**，R=2 時只要在 `r % R == 0` 才呼叫（上面的程式就是這樣），奇數 round 取用的 `cur_chunks[5:15]` 已經是加過噪聲、且以正確的 PD 軌跡為基準的版本（agent 在奇數 round 的真實狀態 = 取樣時追蹤的軌跡，因為 driver 的 PD 與 sim 在 1e-6 內一致，見 label 腳本 asserts `label_imagined_branches.py:96,98,115`）。**一致**。
- **不可以**在奇數 round 再對 `cur_chunks[5:]` 呼叫一次 `inject_noise`（雙重噪聲，且 `rel` 基準錯）。
- 另一個細節：`inject_noise` 的 `max_z` cap（`:134` `rel *= min(1.0, max_z / ...)`）與 `AGENT_BOX` clip 是作用在**整個相對動作**（含乾淨部分），不只噪聲項；所以 `inject_sigma>0` 的分支即使噪聲很小也可能削掉 policy 原本 >3σ（≈62 px）的大動作，而 `inject_sigma=0` 的分支不會呼叫 `inject_noise`（`:314`）。σ∈{0,0.1} 的對比因此混入「是否被 cap / clip」。T2(c)（0.1σ 的實際像素偏移）請順便量「未加噪聲時 cap 會削掉多少比例的動作」。

#### 15.4 `--inject-sigma` 目前的閘門與 `query_rounds` 的意義

- 閘門：**沒有閘門**。`args.inject_sigma > 0` 時每個 round（0…h_max−1）都加噪聲（`:314`）。「只在 query 之前」是靠 label 腳本天然成立的：query r 的標籤 = replay `exec_abs_targets[: r*5]`（都是 round < r 的動作）再接乾淨 policy；round ≥ r 的噪聲不影響該 row 的標籤，也不會進入該 row 的 latent stack（`latents[r:r+3]`）。R=2 之後，r∈{0,2,4,6,8} 的 query 都落在 chunk 邊界，所以 replay 結束時接續的新 policy `policy.reset()` 之後第一步會重新規劃，與 stage 1 的 chunk 邊界語意一致。
- `--noise-hold` 預設 **5**（`:169`）；`scripts/run_value_model.sh:62-66` 的 `imagine` **沒有傳 `--noise-hold`**，所以 pre-reg 的 `noise_hold=1` 必須由 T3 的 sh 新增 `--noise-hold "$(y stage2.noise_hold)"`（見 §3）。
- `query_rounds`（`label_imagined_branches.py:137`）：round index r，代表「想像 rollout 推進 r 個 round（= 5r 個 sim step）之後」的狀態；row 的 latent stack = `br["latents"][r : r+3]`（`:252`，最新一格 = 時間 5r 的影格；latents index 0–2 是真實 context、3+ 是生成），`frame_steps = [5r−10, 5r−5, 5r]`。`r=0` 每個 seed 只輸出一次（`:170-171`）：排序後第一個 branch（`sigma0.0` 目錄先於 `sigma0.1`，`:143` `sorted(root.glob("sigma*/branches"))`），所以 r=0 那列的 `sigma` 欄位一律是 0.0。
- r=2 的 latent stack = `[真實 anchor 影格, 想像 1, 想像 2]`；r≥4 才是全想像。G2 定義的 r≥2 因此包含「含一格真實影格」的 r=2。

### N16 `sim_rollout.rollout_batch`：成功偵測、`success_step`、任意步的 coverage

- 成功偵測：`critic/sim_rollout.py:229` `o, _r, term, trunc, info = envs96[i].step(act[i])`；`:247-250` `if term or trunc: done[i] = True; if bool(info.get("is_success", False)): success_step[i] = t + 1`。gym_pusht 的 `terminated = is_success = coverage > success_threshold`（`pusht.py:258`），`truncated` 由 TimeLimit（`max_ep_steps`）產生。所以 `success_step` = **首次成功的步（1-based，已執行步數）= termination step**，兩者相同；`trunc` 只在 step 超過 `max_ep_steps`（= `episode_steps+10`）時才可能為真，loop 內不會。`success_step = -1` 表示未成功。
- stage 1 的 T：`T = success_step − t`；`success_step == t` 表示 query 狀態本身就是終止狀態（`pre_success`，`collect_value_states.py:131-132`，`y=True`）。`T_succ`（pre-reg 用語）= `success_step`（絕對步）。
- 任意步 coverage：**現在只有 5 步網格**（`grid_coverage`，`:161,241`）。`info["coverage"]` 每一步其實都算了（`pusht.py:256,263`），`sim_rollout` 只是沒存。新增 `cov_per_step` 即可（P1 處理 1）。`v_target` 所需的 `cov_end`：
  - `T ≤ B`（成功）：`cov_per_step[t + T] = cov_per_step[success_step]`（> 0.95）
  - 否則（`T > B` 或未成功）：`cov_per_step[t + B]`（≤ 300）
  - **注意「T > B 但最終有成功」的 episode**（success 在 t+200 到 300 之間）：標籤視為失敗（`y=0`、`v_target=0`），`cov_end` 必須取 t+200 處（episode 當時還沒結束，有記錄），不是成功時的 coverage。
- stage 2（`prepared_envs`）：`success_step` 是「接續起點起算」，因此直接就是 T；`cov_per_step[:,0]` = replay 後 sim coverage。

### N17 seed 與 namespace（730000 / 740000 / 750000 可以直接接上；但有三處需要顯式傳入）

- stage 1 起點：`collect_value_states.py:66-71` `reset_env = gym.make("gym_pusht/PushT-v0")`; `reset_env.reset(seed=s1["reset"]["seed"] + i)`; `reset_starts.append(sim_state(reset_env.unwrapped))`。gym_pusht reset（`pusht.py:268-286`）：`state = [integers(50,450), integers(50,450), integers(100,400), integers(100,400), uniform(-π,π)]`，`np_random` 由 `seed` 決定。→ `reset(seed=730000+i)` 直接可用。之後以 CoG 讀回 + `set_state` 放進 96px / 224px env（`sim_rollout.py:147-152`，兩者都先 `reset(seed=0)`，env 的隨機性在 step 中不再使用，所以 env 端不受 seed 影響）。
- policy 雜訊：`sim_rollout.py:184` `torch.manual_seed(seed + batch_start)`（**全域** seed，整個 batch 共用一個 stream）。stage 1 傳 `seed=0, batch_start=shard0+b0`（`collect_value_states.py:222-223`）。→ T3 要把 `seed` 改成 namespace 起點：stage 1 `seed=730000`、anchors 來源 rollout `seed=740000`、test anchors 來源 `seed=750000`。三個 namespace 相隔 10000，`batch_start ≤ 2000` 不會撞。結果依賴 batch 組成（batch=50，`value_model_topband.yaml` `batch: 50`），所以 `batch` 要固定寫進 config 並視為實驗參數。
- 動作噪聲：`noise_rng = np.random.default_rng(s1["perturb"]["seed"] + 1)`（`collect_value_states.py:190`），在 batch 順序中推進；`draw_perturbations`（`:78-88`）用 `perturb.seed` 抽 `sigma` 與 `perturb_p`。topband config 用 `seed: 0`（yaml:18）。→ sailor config 請用明確新 seed（例如 `perturb.seed: 730000`，`noise_rng` 自動是 730001；避免與 T2 的 760000 重疊）。
- driver（stage 2）：`closed_loop_rollout.py` **完全沒有 `torch.manual_seed`**（`rg manual_seed` 只在 `imagined_rollout.py:308,473`）：policy 的 diffusion 取樣、lead-in 的 `policy.select_action`、NanoWM worker 的 `dfot_sample` 都是不可重現的（branch npz 有存動作與 latents，所以標籤可重現，但 branch 生成本身不能重跑）。`--noise-seed` 預設 0（`:170`），anchors 與 test anchors 兩次 run 會用**同一串 z**。→ T3 新增 `--torch-seed`（預設 None = 現行行為），`run_value_model.sh` 傳 `namespace + 分支索引`；`--noise-seed` 同樣以 namespace 區分（σ 的兩次 run 不要相同）。
- driver 的 `reset(seed=9000 + i)`（`:228`）：`i` 是 seeds 檔的列號，train anchors（0..249）與 test anchors（0..99）會重複使用 9000..9099；`reset` 後 `place_body` 會覆蓋整個 state，僅殘留 agent 速度（reset 內部一次 `space.step`），所以不影響正確性；`closed_loop_leadin: 9000` 保持原樣即可。
- **nothing reads `seed_namespace`**：`rg seed_namespace` 只有 yaml 註解。sailor config 必須把 730000/740000/750000 抄成實際數值；建議加一個小測試（讀 common.yaml 與 sailor yaml，斷言相等）防止漂移。

### N18 其他會影響目標設計的項目

**(a) `band.n = 0` 會讓 `make_starts` 當掉；且 config 仍需 band 的 `seed` / `min_wall_gap`**（**新發現，阻斷性**）
- `critic/collect_value_states.py:65` `band = sample_topband_starts(n_band, seed=s1["band"]["seed"], min_wall_gap=s1["band"]["min_wall_gap"])`：就算 `n_band=0` 也會取 `s1["band"]["seed"]` 與 `["min_wall_gap"]`——plan §3 建議的 `band: {n: 0}` 會 `KeyError`。
- `sample_topband_starts(0, ...)`（`eval/ood_failure_map.py:129-149`）`while len(kept) < n` 不進迴圈，`np.asarray(kept[:0])` 形狀是 `(0,)`（一維）；`:73` `np.concatenate([band, reset_starts], axis=0)` 與 `(2000,5)` 維度數不同 → `ValueError`（以讀碼判斷；我依規定沒有跑 python，T3 請用單元測試證實）。
- 處理（T3）：`make_starts` 對 `n_band == 0` 短路：`band = np.zeros((0,5))`，並且只在 `n_band > 0` 時才讀 `s1["band"][...]`；`smoke` 分支的 `ratio_band`（`:166`）在 `band.n + reset.n = 0` 時會除以零，不會發生在本計劃但順手加 guard。

**(b) `encode` stage 會默默沿用過期的 latents**
- `scripts/run_value_model.sh:43-50`：`[ -f "$lat" ] && { echo "skip ... (done)"; continue; }`。若在同一個 `out_dir` 重新 `collect`（shard 內容改變），舊的 `latents_*.npz` 會被沿用，`_load_stage1` 只檢查檔案存在（`train_value_model.py:131-132`）。T3 請 (1) `collect` 重寫 shard 時刪掉同名 latents；(2) `_load_stage1` 加 shape 斷言（N10）；(3) 可選：`encode` 輸出同時存 shard 的 `frames_224` 形狀 / 檔案 sha1，載入時核對。

**(c) 路徑硬編碼（test anchors 與 smoke 都會撞）**
- `label_imagined_branches.py:136` `root = out_dir / ("smoke/stage2" if smoke else "stage2")`、`:148` `seeds_path = out_dir / "seeds_stage2.npz"`、`:297` `dst = root / "labels.npz"`；`train_value_model.py:510-512` 固定讀 `stage1`、`stage2/labels.npz`；`run_value_model.sh:32-34` `S1/S2`、`:39-40` `seeds` stage 不看 `$SMOKE`、`:61` `--seeds "$OUT/seeds_stage2.npz"`。
- 影響：(1) test anchors（750000，G6 專用）若也放在 `stage2/`，會被 `train_value_model.py` 讀進訓練，**直接破壞 G6 的獨立性**；(2) smoke 的 anchors 腳本（新）若沿用 `seeds_stage2.npz` 會覆寫正式檔。
- 處理（T3）：label / imagine 改用 `--tag`（或 `--stage2-dir`）：`stage2`（預設）與 `test_stage2`；seeds 檔名 `seeds_stage2.npz` / `seeds_test.npz`；`--smoke` 一律放 `out_dir/smoke/…`；`train_value_model.py` 只讀 `stage2/`（不 glob），並在 `sailor_v1_reset.yaml` 註明。`run_value_model.sh` 新增 stage：`anchors`（產生 train anchors，取代 `seeds`）、`test_anchors`、`test_imagine`、`test_label`，且每個 stage 傳 `--seed-base`。

**(d) label 腳本的 seeds 一致性檢查是「檔案存在才做」**
- `label_imagined_branches.py:148-149,165-166`：`seeds_body = load_seeds(...) if seeds_path.exists() else None`，不存在就靜默略過 `seed_state == seeds row` 斷言。新流程請改為必須存在（`assert`），避免用錯 seeds 檔。

**(e) 雜訊閘門與 `noise_hold`**：見 N15.3 / N15.4（sh 沒傳 `--noise-hold`，預設 5）。

**(f) driver 的 lead-in 使用 ckpt 預設的 `n_action_steps`（很可能是 8），不是 10**
- `closed_loop_rollout.py:230` `policy.reset()`、`:247` `policy.select_action(...)`：沒有設 `policy.config.n_action_steps`，用 ckpt 內的值；repo 的 from-scratch 設定是 8（`configs/nanowm_pusht_common.yaml:78`）。`rollout_batch` 則明確把它設成 `na=10`（`sim_rollout.py:185`）。所以 stage 2 的 10 步 lead-in（= 2 個 NanoWM round 的 context）跟 stage 1 / label 接續的 chunk 節拍（10）不同（8+2 vs 10）。lead-in 動作有存（`lead_acts`）且 label 會原樣重播，所以**不影響標籤正確性**，只是 anchor 之後 10 步的行為分佈與 stage 1 略有差異。若要對齊：driver 加 `--lead-na`（預設 None = 現行行為），sailor 流程傳 10。列為 §4 D5 的附帶決定。

**(g) `inject_noise` 的 cap / box clip**：見 N15.3。

**(h) V 目標改 soft label 的連帶改動**：見 N14（ECE、G4、shuffled、overfit-test、val_predictions 欄位）。另外：G3（within-anchor）分組鍵是 `(seed_id, r, sigma)`（`value_gate_report.py:235-238`），每組只有 k=4 列；G6 的候選池是 (k=4 × σ∈{0,0.1}) = 8 個，需以 `(seed_id, r=2)` 為鍵、跨 σ 合併（用 `cand` + `sigma` 區分）。表 D 雜訊天花板需要「同一 val 狀態重跑兩次乾淨接續」：`rollout_batch(seed=0, batch_start=c0)`（`label_imagined_branches.py:235-236`）固定 seed；需新增 `--cont-seed`（重跑用不同 seed）。stage 1 列無法精確重播（`set_state` 會歸零 agent 速度；noise_rng 不可回溯），所以表 D 建議只用 stage-2 val rows（精確 replay）。

**(i) provenance**
- `datagen/provenance.py::provenance_entry(args, argv=None, **extra)`（stdlib only，兩個 venv 都能 import）；`git_info` 用 `git status --porcelain --untracked-files=no`（`:31-33`），**新增但未追蹤的檔案不會讓 `git_dirty=True`**。
- 現況：只有 `load_config` 會自動寫 `config_provenance.jsonl`（argv、git sha/dirty/branch、config sha1）；沒有任何 critic 腳本把 provenance 嵌進 artifact（`pack_closed_loop.py:413-425` 才有）。
- 建議（T3 / T4）：`collect`、`make_reset_anchors`、`label` 三個腳本在輸出的 npz 加一個純字串欄位 `provenance_json = json.dumps(provenance_entry(args, stage=..., config_sha1=...))`；T4 full run 前先 commit（否則 sha 不對應程式），完成後把 `git_sha` 寫回 pre-reg §8 的 phase 表。`collect_value_states.py` 目前的 `sys.path` 沒有 `examples/nanowm_pusht_aug`（`:46-48`），要 `sys.path.insert(0, str(_CRITIC.parent))` 才能 `from datagen.provenance import provenance_entry`（`label_imagined_branches.py:62-65` 已有）。

**(j) 本機找不到 base ckpt / 無法驗證 horizon**
- `configs/nanowm_pusht_common.yaml` `ckpt.base: outputs/train/pusht_imgonly/obs1_s100000/checkpoints/last/pretrained_model`（相對路徑，`cd` 到 repo root 後解析）；本 checkout **沒有 `outputs/` 目錄**，`~/datastorage/` 只有 `nanowm_pusht_aug/`（裡面有 `value_model/sailor_v1/t2_checks`），其他 worktree 也沒有。NanoWM ckpt（`third_party/nano-world-model/ckpt/`）、SD-VAE（`~/models/sd-vae-ft-mse`）、NanoWM venv 都存在。
- 影響：無法讀 ckpt 的 `horizon` / `n_action_steps` / 統計量，N15.2 的 `horizon ≥ 15` 與 N18(f) 的 8 vs 10 只能依 config 推測；T3 smoke 與 T2 都需要這個 ckpt。請使用者提供實際路徑（或在 sailor config 覆寫 `ckpt.base`）。

**(k) 測試慣例（T3 新測試要遵守）**
- `tests/conftest.py` 只做 `sys.path` 注入：`PROJ`、`PROJ/eval`、`PROJ/attribution`、`PROJ/critic`（所以測試用 `from closed_loop_rollout import ...`、`from config_io import ...`、`from nwaug.pusht_state import ...`、`from critic.value_model import ...`）。提供常數 `PROJ / REPO / NANOWM_PY`。
- 需要 gym_pusht 的測試用 `pytest.importorskip("gym_pusht")`（`test_pusht_state.py:8`、`test_closed_loop_wait.py:12`）。`closed_loop_rollout` 與 `sim_rollout` 的 import 會拉進 `lerobot`（`closed_loop_rollout.py:50-52`）與 `gym_pusht`，所以要在 repo 的 `.venv`（`uv run pytest examples/nanowm_pusht_aug/tests`）跑；不需要 GPU / ckpt / 資料集。
- `test_cross_venv_imports.py` 保證 `critic.value_model`、`config_io`、`datagen.*` 在沒有 lerobot / gym_pusht 的 NanoWM venv 也能 import（`CROSS_VENV_MODULES`）；`value_model.py` 不得新增 lerobot 相依。
- 因此 `v_target`、`keep_query`、`chunk_slices` 請寫成**純 numpy 函式**放在新的 `critic/labels.py`（與 `critic/__init__.py` 並存），測試 import 時不需要 gym / lerobot（`test_v_target.py` 可以完全離線）。

---

## 3. 對 T3 的實作建議清單（依順序；每項附測試）

| # | 檔案 | 位置 / 函式 | 改動 | 測試 |
| --- | --- | --- | --- | --- |
| 1 | `critic/labels.py`（新） | `v_target(T, cov_end, B, gamma, beta=0.0)`、`keep_query(t, p, sigma)`、`cov_end_index(t, T, B)` | numpy-only 純函式。`v_target = gamma**T * (T<=B) + beta * gamma**min(T,B) * cov_end`；`T<0/∞` 視為未成功 | `tests/test_v_target.py`：T=50,B=200 → 0.995^50≈0.7783；T=∞,Φ_end=0.6 → β=0:0、β=0.1: 0.1·0.995^200·0.6≈0.0220；`pre_success`(T=0) → 1；T=201 → 0 |
| 2 | `critic/sim_rollout.py` | `rollout_batch` | (a) 新增 `cov_per_step (nb, episode_steps+1)`，成功後 forward-fill；(b) 新增 `perturb_mode="point"\|"prefix"`（預設 `point`）；`prefix` → `t < perturb_p[i]`；(c) 影格網格與追蹤網格解耦（`frame_cutoff_step=100` 只管 224px 與 `grid_frames`）；(d) 保留舊回傳 key | `tests/test_prefix_perturb.py`（純函式 `keep_query` + 用 fake 最小 policy 或直接驗 `active` 選擇函式；若要端到端需 gym_pusht，`importorskip`）；`cov_per_step` 小測試（N16） |
| 3 | `critic/collect_value_states.py` | `QUERY_TS`（`:54`）、`make_starts`（`:57-75`）、`draw_perturbations`（`:78-88`）、`build_queries`（`:91-152`）、`main`（`:155-`） | `QUERY_TS = range(10,101,10)`；`n_band==0` 短路（N18a）；`perturb.mode/noise_hold` 由 config；保留 `t>=p`；`rollout_batch(seed=730000, frame_cutoff_step=100, perturb_mode=...)`；`build_queries` 新增並輸出 `q_T_succ`、`q_cov_end`、`q_v_target`、`q_perturb_p`；shard 內加 `provenance_json`；`episode_steps=300`；docstring 更正 | 同 #2；`build_queries` 對假資料的輸出（含 T>200 但最終成功 → y=0、cov_end 取 t+200） |
| 4 | `critic/make_reset_anchors.py`（新） | `main` | reset rollout（seed 740000 / 750000 namespace）→ 每來源 episode 抽一個 anchor（t∈{10..100}、排除已成功 / wall artifact）→ `cog_to_body` + 0.5 px round-trip → `save_seeds(..., t=..., src_episode=...)`；`--smoke` 寫 `out_dir/smoke/` | 純函式 `pick_anchor(success_step, wall, rng)` 的單測；round-trip 用 `test_pusht_state.py` 同款 gym fixture |
| 5 | `eval/closed_loop_rollout.py` | `main` round 迴圈（`:309-357`）、argparse（`:140-180`） | `--replan-every-rounds`（預設 1）、`--torch-seed`（預設 None）、`--lead-na`（預設 None）；`chunk_slices` 純函式；branch npz 多存 `fi`、`hist`、`replan_every_rounds`；`assert horizon ≥ off+need−len(past)`；預設行為不變 | `tests/test_replan_cadence.py`（R=1 與舊行為等價；R=2 的偶奇 round 行為；`exec` 覆蓋） |
| 6 | `critic/label_imagined_branches.py` | `main`（`:127-`）、`:136,148,297` 路徑、`:243-246` 接續、`:255-257` cov、`:279-295` 交叉檢查、`:298-318` 輸出 | `--tag`/`--stage2-dir`、必存在的 seeds 斷言；接續輸出 `success_step`、`cov_end`（`cov_per_step`）、`v_target`；`cov=sim_cov`、新增 `cov_imagined`（交叉檢查改用它）；讀 branch 的 `fi/hist/replan_every_rounds` 並 assert；新增 `--cont-seed`；輸出 `provenance_json` | `test_v_target.py`（已覆蓋公式）；label 的欄位對齊用一個小 fake-branch 整合測試（可選，需 gym_pusht） |
| 7 | `scripts/run_value_model.sh` | stage 清單（`:1-16`）、`collect/seeds/encode/imagine/label`（`:38-70`） | 新 stage：`anchors`（取代 `seeds`）、`test_anchors`、`test_imagine`、`test_label`；`imagine` 傳 `--replan-every-rounds $(y stage2.replan_every_rounds) --noise-hold $(y stage2.noise_hold) --torch-seed ... --noise-seed ...`；`encode` 對過期 latents 的防護；smoke 下 anchors 不得覆寫正式檔 | 無單元測試；用 `--smoke` 跑通全流程（T3 驗收） |
| 8 | `configs/sailor_v1_reset.yaml`（新） | — | extends common；`out_dir: ~/datastorage/nanowm_pusht_aug/value_model/sailor_v1`；key 清單見 N13；seeds 抄 730000/740000/750000；`gate:` 先留空但讓 report 腳本在缺少門檻時 fail closed | `tests/test_sailor_config.py`（載入 config、檢查必要 key、seed 與 common 的 `seed_namespace` 相等、`out_dir` 已展開 `~`、`episode_steps==300`、`band.n==0`） |
| 9 | `critic/train_value_model.py`（T5） | `_load_stage1/_load_stage2/load_table`、`compute_loss`、shuffled control、`run_overfit_test`、`val_pred`、`normalizer.fit` | 載入 `v_target / T / cov_end / t / cand / perturb_p`；V loss 目標換 `v_target`，`y` 保留供 AUROC；shuffled 改 shuffle `v_target`；overfit test 用二元 y；`val_predictions` 加欄位與 `oracle_rm_mean_all`；`normalizer.fit` 改 `fit_idx`（D6） | `tests/test_compute_split.py`（band 空 bucket）、`tests/test_soft_target_loss.py`（soft target BCE 等於手算；shuffle 不洩漏） |
| 10 | `critic/value_gate_report.py`（T5） | G1（`:193-208`）、G4（`:261`）、ECE（`:36`）、`real_vs_imagined_block`（`:378-440`）、`write_plots`（`:453-455`）、`to_markdown` | G1 改 reset rows；ECE 對 `v_target`（D4）；G4 用 `|V−v_target|`；新增 2×2 表、G6（`(seed_id, r=2)` 跨 σ 的 8 候選）、表 D；`gate` 缺門檻即報錯 | `tests/test_gate_report.py`（合成 `val_predictions`：G1 選 reset、2×2 表與手算一致） |

---

## 4. 風險 / 需 Opus 決定的事項

### 需要決定（D）

- **D1（T2 結論之後）成功後的 query 與 `T_succ` 命名**：建議 stage 1 存 `q_T_succ`（絕對步，未成功 = -1）、stage 2 存 `success_step`（接續起點起算）；訓練端統一產生 `T`（= 到成功的步數，`-1`=無）。若 Opus 要求兩階段欄位同名，T3 需調整 loader（N14 第 6 點）。
- **D2 anchors 來源 rollout 是否加 prefix 噪聲**：pre-reg §2 只說「另一批 reset rollout」。建議與 stage 1 相同（`σ=0.1`、隨機 p、同規則），使 anchor 的狀態分佈與 stage-1 val 列可比；若是乾淨 rollout，stage 2 與 stage 1 的狀態分佈會多一個差異來源（2×2 表的 c−d 會混入）。需 Opus 定案並寫入修訂紀錄。
- **D3 `cov_end` / `T` 的定義邊界**：`T>200 但最終成功` 的 episode 一律視為失敗、`cov_end` 取 t+200（N16）。請確認這是預期（與 pre-reg §3 的 `T_end=min(T,B)` 一致，我的讀法是一致的）。
- **D4 ECE 與 G4 在 soft label 下的定義**：建議 ECE 對 `v_target`、G4 對 `|V−v_target|`，AUROC / within-anchor AUROC 仍對二元 `y`。若堅持「ECE 對二元 y」，G1 的 E1 門檻需要對「V 系統性偏低」有寬容度。必須在 T5 門檻填入前定案。
- **D5 replan=2 奇數 round 的 lookahead 與 lead-in 節拍**：(a) 奇數 round 的 group 3 用 `cur_chunks[5:15]`（policy 對 horizon 尾端的預測）而不是「下一個 chunk 的前 5 步」（後者在該 round 尚不存在）；(b) lead-in 是否要改為 `n_action_steps=10`（`--lead-na`）。兩者都只影響 NanoWM 的條件輸入 / anchor 後 10 步分佈，不影響標籤正確性。建議 (a) 採用、(b) 採用並註記。
- **D6 `Normalizer` 擬合集合**：改成 `fit_idx`（符合 pre-reg）或修改 pre-reg 文字為「fit + es，val 排除」。建議改程式。
- **D7 CoG 網格是否仍延到 300**：我的建議是只加 `cov_per_step`（到 300），`grid_cog/agent` 保持到 100（N10 / P1）；若 Opus 想保留 pre-reg §2 字面「CoG 網格 每 5 步到 300」，成本低，但沒有下游使用者。
- **D8 t 分佈偏斜**：`p` 均勻 → 晚期 t 的 query 較多（P3），且晚期 query 更容易已成功而被丟掉；是否需要對 t 做重新加權或在報告中分層，請 Opus 決定（我建議只在報告中按 t 分層）。

### 風險（R，依影響排序）

1. **R1 test anchors 與訓練資料混在一起**（N18c/d）：若 label / train 沿用 `stage2/` 與 `seeds_stage2.npz`，G6 的獨立性當場失效。必須在 T3 先改路徑參數化，並在 `train_value_model.py` 加 `assert` 只讀 `stage2/`。
2. **R2 `band.n=0` 與 config 必要 key**（N18a）：`np.concatenate` 維度錯誤與 `KeyError`，T3 第一個 smoke 就會撞到；已列入 #3。
3. **R3 soft label 下 gate 語意漂移**（N14 / D4）：ECE、G4、shuffled control、overfit-test 若不同步改，G1 會因為「V 比 P(success) 低」而系統性失敗，或 shuffled control 沒有真的洗掉訓練目標而洩漏。
4. **R4 stage 1 / stage 2 的狀態分佈差異**：(a) anchors 只存 5 維、agent 速度丟失，r=0 的 stage-2 列速度≈0 而 stage-1 同 t 有真實速度，會影響 OracleNet / ValueNet 的 `agent_state` 特徵；(b) driver lead-in 用 na=8（N18f）；(c) `inject_noise` 的 cap / clip 只在 σ>0 分支出現（N15.3）；(d) 想像 latents 是 NanoWM 生成，真實 latents 是 VAE `posterior.sample()`（這是 2×2 表要量的東西，不是 bug）。這些會讓 (d−b)−(c−a)「WM 幻覺」的拆解混入其他因素，報告中應列為 caveat。
5. **R5 stage-2 val / test 規模小**：val 只有 50 個 anchor（N12）、test 100 個，G3 / G6 的 CI 會很寬；門檻（T5）與 T7 的停損判斷要預期這一點；表 D 的雜訊天花板（20–30 個狀態）同樣很粗。
6. **R6 base ckpt 不在本機**（N18j）：T2 / T3 smoke / T4 都被擋住；horizon 與 n_action_steps 無法實證（N15.2 的 `horizon≥15` 只是推測）。
7. **R7 stale latents**（N18b）與**不可重現的 stage-2 生成**（N17：driver 沒有 `torch.manual_seed`）：前者可能默默餵舊資料，後者讓 branch 無法重跑（標籤仍可由存檔的動作重播）。T3 加 shape 斷言與 `--torch-seed`。
8. **R8 provenance 僅在 config 層級**（N18i）：`git_dirty` 忽略未追蹤檔案；T4 full run 之前務必 commit，並把 `provenance_json` 嵌進 npz，否則事後無法證明 artifact 對應哪一版程式。

---

## 5. Opus 審核（T1）

- 審核者：Opus（唯讀；未改任何程式檔、未 commit、未跑 python / 訓練 / 收資料）
- HEAD：`f9eeb5d60d57b14690d4f7014c25771761404000`，`git status --porcelain` 只有一個未追蹤項（本文件），與 §開頭的自述一致。
- **審核期間新增的事實（推翻 N18j / R6）**：`ckpt.base` 現在存在（`outputs/train/pusht_imgonly/obs1_s100000/checkpoints/last/pretrained_model/config.json` 為 symlink 到 `~/datastorage/nanowm_pusht_aug/ckpts/pusht_imgonly_obs1_s100000/`）。實際值：`horizon: 16`、`n_action_steps: 8`、`n_obs_steps: 1`、`drop_n_last_frames: 8`、`crop_shape: [84,84]`。因此 N15.2 的「`horizon >= 15`」**成立但只多 1 步**，N18f 的「很可能是 8」**確認為 8**。R6 作廢，T2 / T3 smoke 不再被 ckpt 擋住。

### 5.1 抽查結果

行號整體評語：**抽查的 7 項全部對得上，行號誤差都在 0–2 行以內**，沒有一項是抄 plan 而來（plan 對 P5 的 `:70-71`、對 P1 的 `:154` 等處確實與實碼不同，本文件都已自行更正）。逐項如下。

**P1（`frame_cutoff_step` / 網格）：確認。** `critic/sim_rollout.py:60` `frame_cutoff_step: int = 110`；`:154-156` `grid_steps = list(range(0, frame_cutoff_step + 1, frame_grid_stride))` + `grid_idx_of_step`；`:158-162` 五個 `grid_*` 陣列共用 `G`；`:236-241` 只有 `(t+1) in grid_idx_of_step` 才寫 `info["coverage"]`；`:232,242` 224px env 只在 `(t+1) <= frame_cutoff_step` 才 step / 抓影格；`:226-228` `done[i]` 直接 `continue`（成功後網格保持 `np.zeros`，不是最後值）；`:247-250` `term or trunc → done[i]=True` 且 `success_step[i]=t+1`。呼叫點 `critic/collect_value_states.py:231` `frame_cutoff_step=110` 確認；`critic/label_imagined_branches.py:227-240` 確認沒傳 `frame_cutoff_step`。
「改用 per-step `cov_per_step` 而不是把 5 步網格延到 300」的提案 **正確且應採納**（= D7）。補充三點理由與一個實作陷阱：
1. `build_queries` 只用 `gi = t // 5`（`collect_value_states.py:130`）取 `grid_cog / grid_agent_* / grid_coverage`，`QUERY_TS` 最大 100 → `gi<=20`；把 `frame_cutoff_step` 降到 100 後 `G=21`，`gi` 仍在界內，`_load_stage1` 的 `gsteps` 最大也是 20（`train_value_model.py:147`）。所以「影格與 cog 網格截在 100、coverage 延到 300」是自洽的。
2. 把 `frame_cutoff_step` 拉到 300 會讓 `grid_frames` 變 `(nb,61,224,224,3)`，且 `:232,242` 的 lockstep assert 會一路跑到 300 步 —— 成本與風險都沒必要。
3. **實作陷阱**：`cov_per_step` 的 forward-fill 不能只處理「成功」，還要處理 `trunc`（本計劃 `episode_steps=300 < max_ep_steps=310` 所以 loop 內不會發生，但 `prepared_envs` 路徑的 env TimeLimit 由呼叫方給，要防守）。且 `done.all()` 會 `break`（`:255-256`），break 之後的 index 必須已經被 fill，不能留 0。

**P3（`perturb_p == t` 與 query 保留規則）：確認，逐字對得上。** `critic/sim_rollout.py:198-199`：`if t % na == 0:` / `active = [i for i in range(nb) if (not done[i]) and sigma[i] > 0 and perturb_p[i] == t]`。`critic/collect_value_states.py:128` `if sigma[e] > 0 and t < perturb_p[e] + 10: continue`。`draw_perturbations` 在 `:78-88`、`p_choices = np.arange(0, 101, 10)` 在 `:86`。
另外我驗過 `prefix` 模式改完之後 `pending` 的記帳仍正確：每個 `t % na == 0` 邊界會覆寫 `pending[i]`（`:220`），`:252-254` 在 `t == p0 + na - 1` 刪除，而新邊界恰好在 `p0 + na`，所以連續加噪聲的 chunk 不會互相干擾、也不會漏刪。`perturb_p` 都是 `na=10` 的倍數，所以「`t < p` 加噪聲」等於「邊界 0…p−10 加噪聲、從 p 開始的 chunk 乾淨」，query t=p 就是第一個乾淨 chunk 的起點 —— 與 pre-reg §2 完全一致。**但見 §5.3 O11：p=0 的 episode 在 prefix 模式下其實完全沒有噪聲。**

**N14（`compute_loss` 與 soft target）：確認，且「`--overfit-test` 會永遠 FAIL」這個判斷是對的。** `critic/train_value_model.py:288-316` 確認（`:300` `y_np`、`:303` `binary_cross_entropy_with_logits`、`:304` `rm_mse`、`:311` `loss`）。`y` 的下游確認：`:450`（per-member val AUROC）、`:605-625`（shuffled control，`:625` 對真實 y 算 AUROC）、`:656`（`val_pred["y"]`）、`:674-675`（metric-code sanity）、`:692-695`（class_balance）；`value_gate_report.py:194-247`（G1–G3）、`:36`（`ece`，以二元 y 為標籤）、`:261`（G4 `absresid = |v_mean_all − y|`）、`:350`（`_rvi_cell`）。
`--overfit-test` 的門檻在 `:482` `if parts["bce"] < 0.05` 與 `:485` `passed = bool(final_bce < 0.05)`。**算式上證實**：soft target p 的 BCE 下界是 p 的二元熵 H(p)；`v_target` 的非零值域是 `[γ^200, 1] = [0.367, 1]`，例如 T=50 → p=0.7783 → H(p)=0.529 nats。只要有任何成功列，平均 BCE 的下界就遠大於 0.05，門檻必然 FAIL。處理方式（overfit-test 繼續用二元 `y`）正確。
補一個 N14 沒提到的：`run_overfit_test` 呼叫 `compute_loss` 時 `y_override=None`（`:477`），所以一旦 `compute_loss` 的預設目標換成 `v_target`，overfit-test 會自動吃到 soft target —— 修法必須是 **`run_overfit_test` 顯式傳 `target_override=rows["y"].astype(np.float32)`**，不是「不改」。

**N15.2 / N15.3（`--replan-every-rounds` 機制與噪聲一致性）：確認，而且我自己重推了一次索引。** `eval/closed_loop_rollout.py:309`（round 迴圈）、`:311-313`（每 round 都 `sample_chunks`）、`:314-317`（每 round 都 `inject_noise`，無閘門）、`:322-323`（`past` / `look`）、`:343-344`（`ex = chunks[:, :fi]`）確認。`sample_chunks` 在 `:96-109`（回傳整個 horizon）、`inject_noise` 在 `:116-138`（`:129` 一次抽整條 horizon 的 z、`:130-136` 沿 chunk 用 `ac.pd_step` 追蹤、`:134` cap、`:135` `AGENT_BOX` clip）。
索引重推：`stream` 在 round r 開頭長度恰為 `lead + r*fi = 10 + 5r`，`k0 = 5r`，所以 `:322` 那個**寬 15** 的切片 `stream[b, 5r : 5r+15]` 被陣列長度截成 **10** 個 —— 這是 `len(past)==10` 的唯一理由，本文件引用了程式註解但沒有說明機制（見 §5.4 M7 要求加 assert）。於是 `need - len(past) = 10`。R=2 的奇數 round：chunk 在 r−1 取樣，chunk index j ↔ stream index `5r+5+j`；窗口需要 stream index `5r+10 … 5r+19` → chunk index **5…14** → `cur_chunks[b, 5:15]`，**horizon >= 15 確認為必要條件**，而 ckpt 的 horizon=16 剛好滿足。`ex` 應為 stream index `5r+10 … 5r+14` → `cur_chunks[:, 5:10]` → 與 `off = (r % R) * fi = 5` 一致。R=1 時 `off=0`，`look`/`ex` 與現行逐位元相同 —— **提案的三行改動在索引上完全正確**。
噪聲一致性也確認：`inject_noise` 在取樣那一刻以 `pos, vel` 為起點、用 `ac.pd_step` 沿整條 chunk 追蹤（`:131-136`），而 driver 推進時用的是同一個 `ac.pd_step` 與同一批 `ex`（`:347-352`），所以奇數 round 取用的 `cur_chunks[5:15]` 的 PD 基準 = 該 round 的真實 agent 狀態。**一致，不需要（也不可以）在奇數 round 再呼叫一次 `inject_noise`。** `noise_hold=1` 時 `z` 有 16 個獨立抽樣（`:129`），足夠覆蓋 `[5:15]`。
N15.3 關於 `max_z` cap 與 `AGENT_BOX` clip 只在 `inject_sigma>0` 分支出現的觀察也是對的，而且**比本文件說的更嚴重**：見 §5.3 O1 的副產品 —— 若 σ=0 與 σ=0.1 共用 `--torch-seed` 與重播的 lead-in，σ=0.1 的候選就是 σ=0 同一支 base chunk 加噪聲，cap/clip 的影響變成成對可量的，T2(c) 的量測也才有意義。

**N17（seeding）：確認。** `critic/sim_rollout.py:184` `torch.manual_seed(seed + batch_start)`（全域 seed，整個 batch 共用一條 stream）；呼叫點 `collect_value_states.py:222-223` `seed=0, batch_start=shard0+b0`。`eval/closed_loop_rollout.py` **完全沒有 `manual_seed`**（`rg -n "manual_seed|default_rng|np.random"` 在該檔只命中 `:296` `noise_rng = np.random.default_rng(args.noise_seed)`）；`imagined_rollout.py` 有兩處（`:308`、`:473`）。`configs/nanowm_pusht_common.yaml:84-101` 的四個 `sailor_v1_*` namespace 確認已登記（730000 / 740000 / 750000 / 760000）且 `rg seed_namespace` 沒有任何程式讀取點。
本文件沒有推到底的後果就是 §5.3 的 **O1（阻斷 G6）**：driver 沒有 torch seed ⇒ 兩次 σ run 的 lead-in 動作不同 ⇒ anchor 狀態不同 ⇒ 「同一 anchor 的 8 個候選」不成立。另外接續也寫死 `seed=0`（`label_imagined_branches.py:235`），要換成登記過的 namespace（見 O9）。

**N18a（`band.n = 0`）：確認，兩個失效模式都真的存在。** (i) `critic/collect_value_states.py:65` 無條件讀 `s1["band"]["seed"]` 與 `s1["band"]["min_wall_gap"]` → plan §3 建議的 `band: {n: 0}` 會 `KeyError`。(ii) `eval/ood_failure_map.py:129-149` `sample_topband_starts`：`n=0` 時 `while len(kept) < n` 不進迴圈，`return np.asarray(kept[:n])` 對空 list 回傳 **shape `(0,)`、ndim 1**；`collect_value_states.py:73` `np.concatenate([band, reset_starts], axis=0)` 與 `(2000,5)` 的 ndim 不同 → `ValueError: all the input array dimensions ... must match`。我同樣沒跑 python，但這是 numpy 的定義行為，T3 的單元測試必須把它釘住。順帶：`:166` 的 `ratio_band` 在 `band.n=0 / reset.n=2000` 下是 0（不會除零），只有兩者皆 0 才會炸 —— guard 仍值得加。

**N18c（硬編碼 `stage2/` 與 `seeds_stage2.npz`）：確認，是本計劃最高風險項。** `critic/label_imagined_branches.py:136`（`root = out_dir / ("smoke/stage2" if smoke else "stage2")`）、`:148`（`seeds_path = out_dir / "seeds_stage2.npz"`）、`:297`（`dst = root / "labels.npz"`）；`critic/train_value_model.py:510-512` 固定 `root/"stage1"` 與 `root/"stage2"/"labels.npz"`；`scripts/run_value_model.sh:34`（`S1`/`S2`）、`:40-41`（`seeds` stage 完全不看 `$SMOKE`）、`:61`（`--seeds "$OUT/seeds_stage2.npz"`）。N18d 也確認：`:149` `seeds_body = load_seeds(...) if seeds_path.exists() else None`、`:165-166` 的 `seed_state == seeds row` 斷言會在檔案不存在時靜默跳過。
**但「只改路徑參數化」不足**：路徑 assert 只能證明「讀的是 `stage2/`」，不能證明 `stage2/labels.npz` 裡沒有 test anchors（例如 T7 手誤把 test 的 label 併進去、或 `--tag` 傳錯）。見 §5.4 M2：必須在 labels.npz 加 `anchor_set` 欄位並在 `_load_stage2` 斷言。

**順帶更正本文件的一處高估（R4(a)）**：「anchor 的 agent 速度丟失」**不成立**。seeds 檔只存 5 維沒錯，但 driver 在 `place_body` 之後會跑 `lead=10` 步真實 policy（`eval/closed_loop_rollout.py:229-252`），`agent_state_hist[0]`（`:307-308`）是那 10 步之後的真實 sim 速度，不是 0。真正的差異是兩件別的事，兩者都該寫進 caveat：(a) anchor 的速度歷史是「從靜止起算的 10 步暫態」，而 stage-1 在同一個 t 有 t 步歷史；(b) **imagined branch 實際的起點是「取樣到的 anchor 狀態再往後 10 步」，不是取樣狀態本身**（`make_reset_anchors` 在 t 抽狀態，driver 又跑 10 步），所以 stage-2 的「有效 t」是 t+10+5r。

### 5.2 決定 D1–D8

**D1（欄位命名）：採用本文件的建議，但兩個 stage 都必須額外輸出一個統一的 `T` 欄位。** stage 1 存 `q_T_succ`（絕對步，-1=未成功）、stage 2 存 `success_step`（接續起點起算）供稽核；同時兩邊都輸出 `T`（int64，到成功的步數，-1 = 觀測窗內未成功）、`cov_end`、`v_target`。理由：`load_table` 的合併是 key 驅動的（`train_value_model.py:194` 以 s1 的 key 為準），少一邊就 `KeyError`；多存一個統一欄位是消掉整類 bug 最便宜的辦法。

**D2（anchor 來源 rollout 是否加 prefix 噪聲）：加，與 stage 1 完全相同的規則（σ=0.1、`noise_hold=1`、`mode=prefix`、p ∈ {0,10,…,100} per source episode），且 anchor 的取樣時刻必須滿足 `t >= p`。** 理由：2×2 表的 c−d 與 (d−b)−(c−a) 都建立在「stage-1 val 列與 stage-2 列的狀態分佈只差在想像」這個前提上；若 anchor 來源是乾淨 rollout，stage-1 val 是「乾淨前綴 + 噪聲前綴」的混合、stage-2 是純乾淨，分佈差異就多出一個無法歸因的來源，而這正是本計劃要拆的那個量。成本為零：`perturb_mode="prefix"` 與 `draw_perturbations` 本來就要為 stage 1 做，`make_reset_anchors` 只是再呼叫一次同樣的 `rollout_batch`。額外的加分理由（見 O13）：anchor 被 `place_body` 重新擺放時噪聲歷史已經被抹掉，噪聲**只**透過「取樣到哪些狀態」起作用，不會汙染 anchor 之後的動力學 —— 所以這個選擇不會讓 stage 2 多出任何噪聲相關的 confound。簡潔性的代價（多一組 config key、T2(a) 若選備案時兩邊要一起改）可接受。

**D3（`cov_end` / `T` 的邊界）：確認本文件的讀法，並補一條記錄要求。** `T > B` 但最終成功 → `y=0`、`v_target=0`、`cov_end = cov_per_step[t+B]`（不是成功那一步的 coverage）；`T` 欄位仍存真值（>B），只有 `y` / `v_target` 把它當失敗，這樣 T2(b) 與結案報告才能算出「有多少列是被 B=200 的窗口切掉的成功」。完全未成功 → `T=-1`，`v_target` 視同 T=∞。與 §3 的 `T_end = min(T,B)` 一致。

**D4（soft label 下的 ECE 與 G4）：ECE 對 `v_target` 算，G4 用 `|V − v_target|`，AUROC / within-anchor AUROC 仍對二元 `y`。** 理由：BCE-with-logits 對 soft target p 的最小值點是 `E[p|z]`，所以 V 收斂到的量是 `E[v_target|z]`，本質低於 `P(y=1|z)`；對二元 y 算 ECE 會有一個與 γ、T 分佈有關的系統性偏差下界，門檻無法解讀。**T5 必須實作的確切定義**：

> 在 G1 的列集合（stage-1 `source=="reset"`、`gate_mask` 後）上，取 `p_i = v_mean_all_i`，用 **[0,1] 上 10 個等寬 bin**（沿用 `value_gate_report.py:36-49` 的 `ece` 骨架，`np.digitize` + `np.clip`），令 bin b 的列集合為 S_b、`n_b=|S_b|`、`N=Σn_b`，則
> `ECE_soft = Σ_b (n_b / N) * | mean_{i∈S_b}(v_target_i) − mean_{i∈S_b}(p_i) |`。
> G1 的 E1 門檻掛在 `ECE_soft`。同時**無門檻**回報 `ECE_bin`（同式但用二元 `y_i` 取代 `v_target_i`）以便與舊的 topband run 對照，並把 `gate_report.png` 的可靠度圖 y 軸換成 `mean v_target`（`value_gate_report.py:453-471`，同時把 `:455` 的 `m1 if m1.any() else gm` 退回行為拿掉 —— 靜默換資料的圖不可接受）。
> G4：`absresid = |v_mean_all − v_target|`（取代 `:261`），仍用 `bootstrap_group_diff_ci` 的 top/bottom quintile 比較；額外無門檻回報用二元 y 的版本。

**D5（奇數 round 的 lookahead 與 lead-in 節拍）：(a) 接受該近似；(b) lead-in 強制 `n_action_steps=10`。**
(a) **可接受。** 奇數 round 真正被執行的 5 步（window group 2）與 R=1 時逐位元相同，標籤來自 `exec_abs_targets` 的精確 sim 重播，所以近似**完全不影響標籤正確性**，只影響 NanoWM 的條件輸入。而且在 10 步重規劃節拍下，真實 policy 在那一刻本來就還沒有「下一個 chunk」，用自己 horizon 尾端的預測填 lookahead 比憑空再取樣一個 chunk 更貼近真實資訊集。條件：必須加 `assert cur_chunks.shape[1] >= off + need - len(past)`（實測 horizon=16 ≥ 15，只多 1 步，所以這個 assert 是必要的防呆），並在 branch npz 存 `horizon`。**caveat 入報告**：`cur_chunks[10:15]` 落在 policy 部署時從不執行（ckpt `n_action_steps=8`）、訓練時 `drop_n_last_frames=8` 的 horizon 尾段，是條件輸入品質最差的區段（見 O12）。
(b) **強制設 10。** 理由：ckpt 實測 `n_action_steps=8`，10 步 lead-in 會變成「一個 8 步 chunk + 另一個 chunk 的前 2 步」，anchor 因此落在 chunk 中間，而 stage 1 的每個 query 都在 chunk 邊界上（`sim_rollout.py:198` 的 `t % na == 0`）。設成 10 之後 anchor 恰好是一個 chunk 邊界，stage-1 / stage-2 的「query 在邊界」語意才一致，代價是一個 `--lead-na` 參數。`sample_chunks` 繞過 `n_action_steps`（`:96-109` 直接用 `conditional_sample`），所以不會干擾 round 取樣。driver 要 assert 設定後的 `policy.config.n_action_steps == 10`。

**D6（Normalizer 擬合集合）：改程式，只在 `fit_idx` 上擬合，並排除 stage-2 `wall_artifact` 列。** 理由：預先註冊文件是承諾，程式改一行（`train_value_model.py:556-557` 的 `train_idx` → `fit_idx`）就能照做，沒有理由改承諾。排除 `wall_artifact` 是為了一致性 —— 那些列已經被 loss 丟掉（`:353-355`），統計量卻把它們算進去沒有道理。同時修掉 `critic/value_model.py:256-257` 的錯誤 docstring（OracleEnsemble 聲稱「fit independently」，實際與 ValueEnsemble 共用同一個 `normalizer` 物件，`train_value_model.py:630-631`）。

**D7（CoG 網格是否延到 300）：不延。** 只新增 `cov_per_step`（`(nb, episode_steps+1)`，到 300，成功 / 終止後 forward-fill），`grid_cog / grid_agent_*` 與 `grid_frames` 一律截在 `frame_cutoff_step=100`。理由：100 步之後的 cog / agent 沒有任何下游消費者（query 特徵與 oracle 特徵都只用 `gi = t//5 <= 20`），延長只會讓 `grid_frames` 與 lockstep assert 一起膨脹。pre-reg §2 表格的「coverage / CoG 網格 每 5 步到 300」由 §9 修訂為「coverage **每步**到 300；CoG / agent 網格每 5 步到 100」。

**D8（t 分佈偏斜）：只在報告中分層，不重新加權。** 理由：重新加權會改掉估計量本身，讓 G1/G2 的門檻依賴一個在看到資料品質之前拍定的權重；而 unweighted 的列分佈就是模型實際被使用的分佈。要求：`val_predictions.npz` 必須帶 `t`（stage 1）與 `r`（stage 2），報告按 t ∈ {10–40, 50–70, 80–100} 三層各自回報 n / y 比例 / AUROC / `ECE_soft`，並附 t 的直方圖與每個 t 的 `y` 比例；`class_balance` 同時按 t 層輸出。若任一層的 n < 300 或 y 比例落在 <5% / >95%，在報告中標記該層不可解讀。

### 5.3 兩份文件都漏掉的事

**O1（阻斷 G6 與表 C；最嚴重）σ=0 與 σ=0.1 的 branch 不共享 anchor 狀態。** `scripts/run_value_model.sh:58-66` 對每個 σ 跑一次 `closed_loop_rollout.py`；driver 的 lead-in 用 `policy.select_action`（`:247`）——diffusion 取樣，吃全域 torch RNG——而 driver **沒有任何 `manual_seed`**（N17 已確認事實，但沒有推到這個後果）。兩次獨立 process 的 torch 預設 generator 由 OS entropy 播種，所以兩次 run 的 `lead_acts` 不同 ⇒ `anchor_state` 不同 ⇒ 表 C 的「每個 anchor 8 個候選（k=4 × σ∈{0,0.1}）」**在物理上不是同一個決策點的 8 個候選**，G6 的「挑一個候選」這件事沒有定義。標籤本身仍正確（label 用該 branch 自己的 `lead_acts` 重播，`label_imagined_branches.py:90-98`），所以這個錯誤不會被任何現有 assert 抓到 —— 它只會安靜地讓 G6 量錯東西。
修法（M1）：driver 輸出 `lead_in.npz`（per-seed `lead_acts / anchor_state / agent_state_hist[0] / ctx`）並新增 `--lead-in <npz>` 直接重播（不呼叫 policy），加上 `--torch-seed`；sailor 的 `imagine` stage 先跑 σ=0.0，再用 `--lead-in` 跑 σ=0.1。**副產品（很有價值）**：lead-in 重播不消耗 torch RNG，於是同一個 `--torch-seed` 下兩次 run 在 r=0 抽到的 4 支 base chunk **完全相同**，σ=0.1 的候選 j 就是候選 j 加噪聲 —— σ 對比變成成對比較，`inject_noise` 的 cap/clip 效應（N15.3）也才可量。這同時要求把表 C 的候選集定義寫清楚：8 個候選 = 4 支 base chunk × {乾淨, 加噪聲}，**不是** 8 次獨立取樣，所以「隨機挑」不等於「base policy 的成功率」（pre-reg §5 表 C 的括號註「= base 取樣」必須撤回）。label 腳本另加 assert：同一 `seed_index` 的所有 branch 的 `lead_acts` / `anchor_state` / `agent_state_hist[0]` 必須逐位元相同。

**O2（改變 2×2 表的解讀）b 格的 ValueNet 也拿到真實 agent 狀態。** `train_value_model.py:280-283` 的 `forward_net` 對 ValueNet 餵 `rows["latent_stack"]` **加上 `rows["agent_state"]`**；stage-2 的 `agent_state` 是 `agent_state_hist[r]`（`label_imagined_branches.py:253`），而該值與 sim 在 1e-6 內相等（`:113-115` 的 assert 保證）。也就是說 **b 與 d 兩格的 agent pos/vel 都是真值**，WM 的幻覺只能透過 block pose（latent 影像）進來。後果：(d−b)−(c−a) 只量到 **block-pose 的 WM 幻覺**，不是「WM 狀態誤差」。這對線上用法其實是忠實的（線上 agent 狀態永遠是已知的 PD 追蹤值），但 pre-reg §5 的分解文字必須改口，否則結案報告會宣稱一個沒有量到的量。

**O3（預先註冊的程序漏洞）表 D 用 val 列估天花板，而 T3 門檻又必須在看到 val 之前填入。** pre-reg §5 表 D 寫「20–30 個 **val** 狀態」，§5 表 B 寫「G3 的 T3 < 表 D 的雜訊天花板」，§5 標題又寫「門檻由 Opus 在 T5、**看到任何 val 之前**填入」—— 三者互相矛盾：要用 val 的標籤重複量測去決定 val 上的門檻。**修法（M3）：表 D 一律用 stage-2 的 `fit` + `es` split anchors**（兩者都永不進 gate metric，`compute_split` 已把它們分開，`train_value_model.py:208-260`），隨機抽 25 個 anchor（seed 寫進 config），val 上的天花板在 T6 只當一致性檢查回報、不得反過來調 T3。

**O4（pre-reg §3 的推導站不住）`score = V + β·RM` 不是 potential-based shaping 的 value。** 依 shaping 的 telescoping，狀態 s 的 shaped value = `E[γ^T·1[T<=B]] + β·E[γ^{T_end}Φ_end] − β·Φ(s)`。而本計劃訓練 V 用 β=0（owner 定案），搜尋時加 `+β·RM(z_end)`，RM 的目標是 **當下**的真實 coverage `Φ(s)`（§3、D8 的 P8 處理）。所以 score 用的是 `+β·Φ(s')`，符號與 shaping 推導**相反**，而且缺了 `E[γ^{T_end}Φ_end]` 那一項；更關鍵的是 8 個候選在 r=2 各自處於**不同**狀態，`Φ_t` 並不是可以「去掉的常數」。這不影響 owner 已定案的任何東西（score 的式子、β 候選、V 用 β=0 訓練都保留），但 §3 那句推導必須撤回：β·RM 是一個 **greedy coverage bonus**（偏好「已經把 coverage 推高」的候選），不是 shaped value。`cov_end` / `T_succ` 欄位仍要存，但用途改成 (i) 標籤診斷與表 D、(ii) 未來若要真的訓練 β-target 的 V 的原料 —— 不是「讓 β 離線套用到 score」。

**O5（G6 的估計量）`S_random` 與 gap closed 的定義現在會產生一個不穩定的檢定。** 兩份文件都只寫「隨機挑」與「gap closed 的 CI 不含 0」。必須釘死：(i) `S_random(i) = mean_j y_ij`（8 個候選的平均，= 隨機挑的**精確期望**），不是再抽一次候選 —— 這是純粹的變異數縮減，沒有代價；(ii) gap closed 的分母 `mean_i(S_oracle(i) − S_random(i))` 在 bootstrap 重抽中可能接近 0 或為 0，比值的 percentile CI 會被少數重抽支配。**gate 必須掛在分子** `mean_i Δ_β(i)`，`Δ_β(i) = S_β(i) − S_random(i)`，gap closed 只作為描述統計回報。

**O6（G6 的檢定力）n=100 的 test anchors 只夠偵測相當大的效果，而且真正有資訊的 anchor 可能遠少於 100。** `Δ_β(i) ∈ [−1,1]`；只有 `0 < Σ_j y_ij < 8` 的 anchor（8 個候選的結果有差異）才可能貢獻非零 Δ。若 SD(Δ)≈0.3，one-sided 95% 可偵測的 `mean Δ ≈ 0.05` 絕對成功率；在 `S_oracle − S_random ≈ 0.2` 的情形下相當於 gap closed ≈ 0.25。**保持 G6 誠實的最小分析（必須在 T5 前寫進 pre-reg）**：
> 分析單位 = test anchor i（i = 1…n_used）。`S_β(i) = y_{i,sel}`，`sel = argmax_j [v_mean_all_j + β·rm_mean_all_j]`；`S_random(i) = mean_j y_ij`；`S_oracle(i) = max_j y_ij`。主統計量 `mean_i Δ_β(i)`，用**對 anchor 的成對 bootstrap**（重抽 anchor with replacement，10000 次；每次重抽內同時算 `mean Δ`、`mean(S_oracle − S_random)` 與其比值）。**one-sided 95% 下界（5th percentile）> 0 才算 G6 通過** —— 方向在看資料之前就預先承諾為「優於隨機」，one-sided 是這個假設的誠實形式，也比 two-sided 有檢定力。gap closed 回報點估計與同一批重抽的 percentile interval，**不作為 gate**。同時必須回報 `n_used`、`n_informative`、`SD(Δ)`；若 `n_informative < 40`，報告必須寫「underpowered」而**不是**「離線不顯著」（後者會被 §6 的 T7 停損規則誤讀為負面證據）。val 上用完全相同的程序選 β（`argmax_β mean Δ_β`），三個 β 的 val 數字全部回報，只有勝出的那一個在 test 上算一次。

**O7（G6 與 `pre_success` 規則衝突）** 現行標籤規則是「pre_success 列從**所有** gate metric 排除」（`value_gate_report.py:176-179`）。但 G6 量的是「挑出來的候選最後有沒有成功」，一個在 r=2 之前 10 步內就已經成功的候選**是**一個成功的候選；把它丟掉會讓 `S_oracle` 偏低、系統性懲罰好候選，也會讓候選集大小逐 anchor 變動。必須明文：**G6 保留 `pre_success` 候選並記 y=1，只排除 `wall_artifact`；任一候選是 `wall_artifact` 或缺列 → 整個 anchor 退出 G6**，並回報退出數。這是對「pre_success 一律排除」規則的一個有意的、預先註冊的例外（理由：G6 量選擇成功率，不是判別力），必須寫進 §9。

**O8（表 A 的 bootstrap 型態）b 與 d 是同一批列，不能用 unpaired。** `value_gate_report.py:135-169` 的 `bootstrap_diff_ci_unpaired` 對兩側各自獨立重抽 group，適用於 stage1 vs stage2（不同 group 軸）。但 2×2 表的**欄內**比較（a vs c、b vs d）是同一批列、同一個 group 軸上的兩個分數（`v_mean_all` vs `oracle_v_mean_all`），unpaired 重抽會高估 CI、可能把真實存在的「編碼器損失」判成不顯著。必須：欄內 Value−Oracle 用**成對** grouped bootstrap（一次重抽同時算兩個 AUROC 再相減）；跨欄（stage1 vs stage2）維持獨立重抽；差中之差 (d−b)−(c−a) 用「兩欄各自獨立重抽、欄內成對」的混合式。

**O9（標籤噪聲依賴 batch 組成）** 接續的 policy 取樣用 `torch.manual_seed(seed + batch_start)`（`sim_rollout.py:184`）且整個 batch 共用一條 RNG stream，而 `label_imagined_branches.py:203` 的 batch 由 **CLI 預設 `--batch 100`**（`:131`）與 job 順序決定，`alive` 過濾（`:222`）還會平移組成。後果：(i) 一列的標籤依賴哪些列跟它同批 —— `--batch` 必須寫進 config 並當實驗參數記錄，train anchors 與 test anchors 必須用同一個值；(ii) 表 D 的兩次重跑除 `--cont-seed` 外必須組成完全相同（同 job 順序、同 `--batch`），否則量到的是 batch 效應而不是 policy 噪聲；(iii) 接續現在寫死 `seed=0`（`:235`），要換成登記過的 namespace（建議在 `configs/nanowm_pusht_common.yaml` 的 `seed_namespace` 新增 `sailor_v1_label_cont: 770000`，表 D 的重跑用 770001）。**建議但非強制**：把 job list 依 `(seed_id, sigma, cand, r)` 排序後才切 batch（現在是依 branch 檔名，`:143-144` 先 sigma 目錄再 seed，同一個 anchor 的 σ=0 與 σ=0.1 候選會差 ~250 列、落在不同 batch），這樣同一 anchor 的 8 個候選落在同一批，重跑時的配對結構穩定。

**O10（group leakage：查過了，設計上沒有洩漏，但沒有任何程式保證）** 針對「stage-2 的 label 列是否來自也出現在 stage-1 訓練列的狀態」：不會。stage-1 起點是 `env.reset(seed=730000+i)`（`collect_value_states.py:66-71`），anchor 來源是 `reset(seed=740000+i)`，gym_pusht 的 reset 分佈由 `np_random` 決定（`gym_pusht/envs/pusht.py:268-286`），兩個 namespace 不重疊 ⇒ 起始狀態幾乎必然互斥，anchor 又是來源 rollout 在 t∈{10..100} 的狀態，不可能等於任何 stage-1 列。`compute_split` 的 group 是 stage-1 `episode` / stage-2 `seed_id`，且 `make_reset_anchors` 每個來源 episode 只抽一個 anchor（本文件 P6 處理 3），所以 stage-2 內部也沒有跨 split 的同源 episode。**但這整條論證完全依賴 config 裡抄對了三個數字，而 `rg seed_namespace` 顯示沒有任何程式讀它。** 必須：(i) `tests/test_sailor_config.py` 斷言 `stage1.reset.seed == common.seed_namespace.sailor_v1_stage1_reset`、`stage2.anchors.seed == …sailor_v1_stage2_anchor`、`test_anchors.seed == …sailor_v1_test_anchor`，且三者兩兩相異；(ii) `seeds_stage2.npz` / `seeds_test.npz` 存 `src_seed` 與 `src_episode`，labels.npz 沿用，讓事後可以直接證明不重疊。

**O11（prefix 模式下 1/11 的「加噪聲」episode 其實是乾淨的）** `perturb_p` 從 `np.arange(0, 101, 10)` 均勻抽（`collect_value_states.py:86-87`），`p=0` 在 prefix 規則下代表「沒有任何 chunk 在 p 之前」⇒ 整條乾淨，但 shard 的 `sigma` 欄位仍是 0.1，`train_value_model.py:165` 會把它當成 σ=0.1 的列。結果：約 9% 的「加噪聲」列其實乾淨，class balance 與任何按 σ 分層的診斷都會被污染。修法：shard 與 row table 多存一個 `noised = (sigma > 0) & (perturb_p > 0)` 欄位，所有按噪聲分層的診斷改用它。

**O12（奇數 round 的條件輸入落在 policy 的 horizon 尾段）** 見 D5(a)：ckpt `horizon=16`、`n_action_steps=8`、`drop_n_last_frames=8`，所以 `cur_chunks[10:15]` 是部署時從不執行、訓練時最後被丟掉的區段。不影響標籤，但 NanoWM 在奇數 round 的條件輸入品質比偶數 round 差，`auroc_by_r`（`value_gate_report.py:312-321`）應該預期 r 為偶數（query round 都是偶數）之間也有系統性差異。列為報告 caveat。

**關於「stage-2 r=0 列算不算想像」（題目 (ii)）：不算，而且應該把它升格成 2×2 表的第三欄。** r=0 的 latent stack 是 `latents[0:3]`（`label_imagined_branches.py:252`），三格都是真實 context 影格的 VAE 編碼，`cov_source="sim"`（`:257`），只差在 VAE 的 `posterior.sample()` 噪聲。程式已經把它分開回報（`value_gate_report.py:223-228` `G2_r0_diagnostic`、`:386,391` `stage2_r0_diagnostic`），pre-reg §5 的 2×2 也已經寫「stage-2 val **r≥2**」—— 這部分沒有錯。漏掉的是它的診斷價值：把它當第三欄「stage-2 r=0（真實影格、stage-2 狀態分佈）」之後，`(r=0) − (stage-1)` 隔離出**狀態分佈位移**（anchor 的 10 步暫態、有效 t = t+10、O13 的速度歷史差異），`(r≥2) − (r=0)` 隔離出**想像**本身。沒有這一欄，c−d 與 (d−b)−(c−a) 就混在一起，O2 的 caveat 也無法量化。代價為零（列已經存在）。唯一的統計限制要寫明：r=0 每個 anchor 只有一列（`:170-171` 每個 seed 只輸出一次）且 σ 欄位一律 0.0，所以 r=0 欄的 n = anchor 數（val 約 50），CI 很寬。

### 5.4 T3 清單的強制變更

以下是 §3 清單通過審核的**必要**修改（編號承接 §3 的 #1–#10）。§3 原有的 10 項全部保留，方向正確。

- **M1（#5，阻斷 G6）** `eval/closed_loop_rollout.py`：除了 `--replan-every-rounds` / `--torch-seed` / `--lead-na` 之外，必須新增 **`--lead-in <npz>` 與 lead-in 的輸出**。driver 在 lead-in 之後把 `lead_acts / anchor_state / agent_state_hist[:,0] / ctx` 寫成 `lead_in.npz`；給了 `--lead-in` 就直接載入並跳過 policy lead-in（不消耗 torch RNG）。`run_value_model.sh` 的 `imagine` stage 先跑 σ=0.0 產生 `lead_in.npz`，其餘 σ 一律帶 `--lead-in` 與**同一個** `--torch-seed`。`--lead-na` 預設 None，sailor 傳 10 並 assert `policy.config.n_action_steps == 10`。`tests/test_replan_cadence.py` 之外新增 `tests/test_lead_in_roundtrip.py`（`--lead-in` 重播與原 run 的 `anchor_state` 逐位元相同；不需要 GPU 的話用 fake policy）。
- **M2（#6 + #9，阻斷 G6）** labels.npz 必須新增字串欄位 `anchor_set ∈ {"train","test"}`（由 `--tag` 決定），`train_value_model.py::_load_stage2` 斷言 `(anchor_set == "train").all()`。單靠路徑參數化與路徑 assert 不夠（見 §5.1 N18c）。同時 label 腳本加 assert：同一 `seed_index` 的所有 branch 的 `lead_acts` / `anchor_state` / `agent_state_hist[0]` 逐位元相同（M1 的性質要被程式釘住）。
- **M3（#6 / T5）** 表 D 的重複接續必須只從 stage-2 的 `fit` + `es` anchors 抽樣（25 個，抽樣 seed 寫進 config），**不得**用 val 或 test（見 O3）。`--cont-seed` 與 `label.cont_seed`（新 namespace 770000/770001）一起加；兩次重跑除 `--cont-seed` 外 job 順序與 `--batch` 必須相同。表 D 的統計量必須**在 G3 的同一個 grouping（`seed_id`, r, σ）內**計算：每組 k=4 列各跑兩次，回報 (i) 邊際 y 一致率、(ii) **組內** `AUROC(y1 當標籤, y2 當分數)` 的平均與 bootstrap CI —— 只有 (ii) 才是 G3 的天花板；邊際 AUROC(y1,y2) 不構成 within-anchor 排序的上界。
- **M4（#6）** `label_imagined_branches.py`：`--batch` 必須從 config 讀（不再用 CLI 預設 100），接續的 `seed=` 改成 `label.cont_seed`（取代 `:235` 的 `seed=0`）；建議把 job list 依 `(seed_id, sigma, cand, r)` 排序後再切 batch（O9）。
- **M5（#3）** `collect_value_states.py` 的 shard 與 `build_queries` 必須多輸出 `q_noised = (sigma > 0) & (perturb_p > 0)`（O11），`train_value_model.py` 載入為 `noised` 並帶進 `val_predictions.npz`；所有按噪聲分層的診斷改用 `noised` 而非 `sigma > 0`。同時 `val_predictions.npz` 必須帶 `t`（D8 要用）。
- **M6（#4）** `critic/make_reset_anchors.py` 必須套 D2：`rollout_batch(..., perturb_mode="prefix", sigma=0.1, perturb_p=draw_perturbations(...))`，且抽 anchor 時限制 `t >= perturb_p`（與 stage 1 的 `keep_query` 同一個純函式）。seeds npz 必須存 `src_seed`、`src_episode`、`t`、`perturb_p`（O10）。
- **M7（#5）** round 迴圈必須加兩個 assert：`assert len(past) == (H - 1) * fi`（現在的 15 寬切片靠 `stream` 長度剛好截成 10，是隱性不變量，見 §5.1 N15.2）與 `assert cur_chunks.shape[1] >= off + need - len(past)`（horizon ≥ 15，實測只多 1 步）。branch npz 多存 `fi / hist / replan_every_rounds / horizon / lead_na`，label 腳本讀出來 assert。
- **M8（#1）** `critic/labels.py` 的 `v_target` 必須接受 `T = -1`（未成功）、`T = 0`（pre_success → 1.0）與 `T > B`（→ 0，`cov_end` 取 `t+B` 處），並且 `cov_end_index(t, T, B)` 對「`T > B` 但最終成功」回 `t+B`（D3）。`tests/test_v_target.py` 必須包含這三個邊界，外加 §3 已列的兩個閉式值。
- **M9（#2）** `cov_per_step` 的 forward-fill 必須涵蓋 `term`、`trunc` 與 `done.all()` 提早 break 三條路徑（§5.1 P1 第 3 點），測試要把「成功後的 index 不是 0」釘住。
- **M10（#8）** `tests/test_sailor_config.py` 必須斷言三個 seed 與 common 的 `sailor_v1_*` 相等且兩兩相異（O10），以及 `stage2.replan_every_rounds` 是 `stage2.h_max` 的因數。
- **M11（#9 / T5，非 T3 阻斷但必須現在定案）** `run_overfit_test` 必須顯式傳 `target_override=rows["y"].astype(np.float32)`（§5.1 N14），不能依賴 `compute_loss` 的預設目標；`compute_loss` 的 `y_override` 更名為 `target_override`。
- **M12（T5，記在這裡以免遺漏）** `value_gate_report.py`：`write_plots` 的 `:455` `m1 if m1.any() else gm` 退回行為必須刪掉（改成 `source=="reset"` 之後不再需要，且靜默換資料的圖不可接受）；`cfg["gate"]` 缺任何 key 必須 fail closed。

T3 不需要、但 T5 前必須寫進 pre-reg §9 的定義（已在 §5.2 / §5.3 給出）：ECE_soft 的公式、G6 的候選集與 tie-break（升序 `(sigma, cand)`，即先 σ=0 再低 cand）、`S_random` / `Δ_β` / one-sided 下界、G6 對 `pre_success` 的例外、2×2 表的成對 bootstrap 與第三欄、表 D 的來源 split。

### 5.5 結論：**PASS-WITH-CHANGES**

§1–§3 的讀碼品質高：抽查的 7 項全部確認，行號誤差 0–2 行，P5 的更正是對的（`place_seed` 確實已經用 `nwaug.pusht_state.place_body`），對 P1 提出的 `cov_per_step` 方案比 plan 原本的「網格延到 300」更好，N15.2 的 `off = (r%R)*fi` 索引推導與 `horizon >= 15` 的結論經獨立重推**完全正確**，N18a / N18c 是真的阻斷性問題。§3 的 10 項清單方向正確，全部保留。

不給 PASS 的理由是三個會讓終點指標失效、而兩份文件都沒抓到的問題：**O1**（σ=0 與 σ=0.1 的 branch 不共享 anchor 狀態 ⇒ 表 C 的「8 個候選」沒有定義，G6 量錯東西，且不會被任何 assert 抓到）、**O2**（b/d 兩格的 agent 狀態都是真值 ⇒ (d−b)−(c−a) 只量到 block-pose 幻覺，§5 的分解文字會宣稱一個沒量到的量）、**O3**（表 D 用 val 列估天花板卻要用來在看 val 之前定 T3，程序上自相矛盾）。加上 **O5/O6/O7** 讓 G6 的檢定從一開始就有定義且誠實。

**T3 可以開工**，但必須含 M1–M10（M11/M12 屬 T5，但定義現在就固定），且 T3 的 commit 必須在 `--smoke` 跑通之外附上 `tests/test_lead_in_roundtrip.py` 與 M2 的 `anchor_set` assert 的證據。D1–D8 全部定案（見 §5.2），pre-reg §9 的修訂紀錄已同步寫入。
