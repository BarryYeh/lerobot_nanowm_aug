# SAILOR v1 — value / reward model 預先註冊文件

狀態：**草稿（T0）**。Opus 審核並在 T5 填入 gate 門檻後才允許 full run（T4 之後的訓練與評估）。T5 修訂：G1/G2/G3/G5 改為無門檻診斷，只有 G6 是決策 gate（見 §9）。
本文件是「看到 val / test 之前」的承諾；之後只能追加「修訂紀錄」，不能改寫已承諾的規則。

- 分支：`feat/sailor-v1-value`（自 `refactor/nanowm-cycle1` @ `e46cc24c` 切出；不開 PR，直接 commit）
- 計劃：`~/.cursor/plans/sailor_v1_value_model_plan_091ed530.plan.md`
- Config：`configs/sailor_v1_reset.yaml`（T3 新增，extends `nanowm_pusht_common.yaml`）
- 資料位置：`~/datastorage/nanowm_pusht_aug/value_model/sailor_v1/`（`~/datastorage` → `/data/wrd/projects/students/fengen.yeh`）。repo 內 `outputs/` 不寫任何本計劃資料。
- 不改動：`configs/value_model_topband.yaml`、`outputs/value_model/topband`、CUPID configs 與資料、`closed_loop_rollout.py` 的預設節拍。



## 1. 問題

SAILOR（arXiv 2506.05294）用 RM + V 在 world model 裡替 base policy 的候選 action chunk 打分。本計劃要回答：

> 在 PushT 上，用 NanoWM latent 訓練的 ValueNet（V + RM 兩個 head）能否在**想像的**（NanoWM）rollout 上對 base policy 的候選 chunk 排序，且離線 Best-of-N 的成功率提升顯著大於 0？

對照 OracleNet（真實 8 維狀態）把差距拆成「編碼器損失」與「WM 幻覺」。線上 Best-of-N **不在**本計劃內。

## 2. 資料設計（已定案）


| 項目                | stage 1（真實影格）                                                                              | stage 2（想像影格）                                                   |
| ----------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| 起點                | 隨機 `env.reset(seed=730000+i)`，n=2000                                                       | anchor：另一批 reset rollout（seed 740000+）在 t∈{10,…,100} 抽的狀態，n=250 |
| 執行者               | base policy（`ckpt.base`），`n_action_steps=10`                                               | NanoWM 想像 k=4 個 branch，每 **10 步（2 round）** 重取樣 chunk            |
| 噪聲                | 每步 0.1σ（`noise_hold=1`，`SIGMA_PX≈20.8 px`），**只加在切換點 p 之前**；p ∈ {0,10,…,100} 每 episode 隨機一個 | `inject_sigma ∈ {0.0, 0.1}`，每步；只加在 query 之前的 chunk              |
| query             | t = 10,…,100（每 episode 10 個），只保留 **t ≥ p**                                                 | r ∈ {0,2,4,6,8}（= t 0,10,…,40 步）                                |
| 標籤來源              | 同一條 episode 的後續（乾淨 base policy）                                                            | 精確 sim 重播（1e-6 assert）後乾淨 base policy 接續 B=200 步                |
| episode 上限        | 300（gym 上限；不再 310）                                                                         | 接續 200 步                                                        |
| 影格網格              | 每 5 步，到 t=100                                                                              | NanoWM 每 round 一格                                               |
| coverage / CoG 網格 | 每 5 步，**到 300**（供 Φ_end）                                                                   | 接續期間每 5 步                                                       |


備案（依 T2(a) 決定，見 §6）：若 0.1σ 每步噪聲對 base policy 1000 episode 成功率的改變在 ±2 pp 內，則改為「整條加噪聲、不丟 query、stage 2 接續也加噪聲」。這個決定在 T2 結束時寫入修訂紀錄，之後不再變。

## 3. 標籤（已定案）

符號：query 時刻 t，成功時刻 T_succ（首次 coverage > 0.95 的 sim step；未成功 = ∞），T = T_succ − t，窗口 B=200，
Φ_t = coverage(t)，T_end = min(T, B)，Φ_end = coverage(t+T_end)，γ = 0.995。

- **二元 y**：`y = 1[T ≤ B]`（供 AUROC / ECE / G1–G3；stage 1 亦含 `pre_success` 的處理沿用現行程式）。
- **V 目標（soft label）**：`v_target = γ^T·1[T ≤ B] + β·γ^{T_end}·Φ_end`，其中 β 只在 **搜尋分數** 用；訓練 V head 時 **β=0**，即 `v_target = γ^T·1[T≤B]`（來自 potential-based shaping Φ=coverage、稀疏成功為吸收態，去掉常數 −βΦ_t）。`cov_end=Φ_end`、`T_succ` 另存欄位，讓 β 可離線套用。
- **RM 目標**：真實 sim coverage。stage 1 = `coverage(t)`；stage 2 = `sim_cov`（重播後 `_get_coverage()`），`cov_imagined` 只作診斷。
- **搜尋分數**：`score = V(z_end) + β·RM(z_end)`，β ∈ {0, 0.1, 0.3}（預先固定，不再加候選）。

閉式檢查（`tests/test_v_target.py`）：T=50、B=200 → v = 0.995^50 ≈ 0.7783；T=∞、Φ_end=0.6 → v(β=0)=0，v(β=0.1)=0.1·0.995^200·0.6 ≈ 0.0220。

## 4. 模型與訓練（沿用現行程式）

- ValueNet：3 幀 SD-VAE latent（H=3, fi=5）+ agent state → V head（BCE 對 `v_target`）、RM head（sigmoid 後 MSE 對 cov）、pose aux。
- OracleNet：8 維真實狀態（agent xy/vel、CoG xy/512、sinθ、cosθ；`nwaug.pusht_state` 的 CoG 約定），同一程序訓練。
- ensemble 5、bootstrap、`v_mean2`（mean-of-2-random）；shuffled-label 對照必跑。
- 分割：grouped 三切分 val / es / fit（以 episode / anchor 為 group）；Normalizer 只在 fit split 擬合；band bucket 允許為空。
- 訓練資料：stage 1 + stage 2 都是訓練資料（`mix_stage1`），val 在各自 bucket 內切；test anchors（seed 750000+，n=100）只用於 G6 確認，訓練與選 β 都不碰。



## 5. 評估（門檻由 Opus 在 T5、看到任何 val 之前填入 `configs/sailor_v1_reset.yaml` 的 `gate:`）



### 表 A：2×2 診斷表（AUROC / ECE / within-anchor AUROC / RM MAE，bootstrap 95% CI）


|                  | stage-1 val（真實影格） | stage-2 val r≥2（想像影格） |
| ---------------- | ----------------- | --------------------- |
| ValueNet（latent） | a                 | b                     |
| OracleNet（state） | c                 | d                     |


分解：c−a = 編碼器損失；d−b = 編碼器 + WM 幻覺；(d−b)−(c−a) ≈ WM 幻覺；c−d = 想像分布本身較難。

### 表 B：gate


| gate | 指標                                        | 門檻        | 資料                          |
| ---- | ----------------------------------------- | --------- | --------------------------- |
| G1   | stage-1 reset AUROC ≥ T1，ECE ≤ E1         | TBD（Opus） | val                         |
| G2   | stage-2 r≥2 AUROC ≥ T2                    | TBD       | val                         |
| G3   | within-anchor AUROC ≥ T3（T3 < 表 D 的雜訊天花板） | TBD       | val                         |
| G4   | ensemble std 有資訊量：                        | V−y       | 的 top/bottom quintile 差 > 0 |
| G5   | RM MAE（對真實 cov）stage1 ≤ M1、stage2 ≤ M2    | TBD       | val                         |
| G6   | 離線 Best-of-N gap closed 的 95% CI 不含 0     | 固定        | **test anchors**            |




### 表 C：G6 離線 Best-of-N

每個 anchor 8 個候選（k=4 × σ∈{0,0.1}），r=2 的 query；比較「隨機挑」「ValueNet β∈{0,0.1,0.3}」「OracleNet」「事後最好」。
`gap closed = (S_β − S_random) / (S_oracle − S_random)`，S_oracle = 事後最好。**β 在 val 上選（最高 S_β）；只有選出的那一個 β 在 test anchors 上報 CI。**

### 表 D：雜訊天花板

20–30 個 val 狀態各重跑兩次乾淨接續，報 y 一致率與 AUROC(y1, y2)；G3 的 T3 必須低於此值。

## ++6++. 決策規則與停損

- T2(a)：|Δ成功率| ≤ 2 pp → 採備案（整條噴噪聲）；否則採 prefix 規則。T2(b)：若 ≥ 20% 的失敗 episode 在 t>100 才「決定」（coverage 最後一次上升在 t>100），在修訂紀錄註明 query 範圍的侷限但不改設計。
- T6：G2 **或** G3 未過 → 停損，轉去處理 NanoWM 準確度，不進 T7。
- T7：選出的 β 在 test 上 gap closed CI 含 0 → 停損，結案報告記為「離線不顯著」。
- 任何門檻在看到 val 之後都不得下修。



## 7. Seed namespace（`configs/nanowm_pusht_common.yaml`）


| 用途                                    | seed                              |
| ------------------------------------- | --------------------------------- |
| stage 1 reset episode                 | `sailor_v1_stage1_reset: 730000`  |
| stage 2 anchor 來源 rollout             | `sailor_v1_stage2_anchor: 740000` |
| G6 test anchor 來源 rollout             | `sailor_v1_test_anchor: 750000`   |
| T2 模擬器檢查                              | `sailor_v1_t2_checks: 760000`     |
| stage-2 接續 policy 取樣（label；表 D 重跑 +1） | `sailor_v1_label_cont: 770000`    |




## 8. Phase 起點 sha 與審核紀錄


| phase | 起點 sha     | 審核                                 | 結果                                                                                                                                                                                                                             |
| ----- | ---------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| T0    | `e46cc24c` | —                                  | 完成（`f9eeb5d6`）                                                                                                                                                                                                                 |
| T1    | `f9eeb5d6` | Opus：`sailor_v1_code_review.md` §5 | PASS-WITH-CHANGES（M1–M12 併入 T3/T5）                                                                                                                                                                                             |
| T2    | 46ebbf5a   | —                                  | 乾淨 68.5% vs 噪聲 69.8%，Δ=+1.30 pp（CI [−2.4, +5.0]）→ 採備案（整條噴噪聲）；失敗 episode 89% 在 t>100 才「決定」（旗標）；0.1σ ≈ 2.9 px/步目標偏移、10 步後 agent 偏 ≈1.5 px                                                                                        |
| T3    | `48a37b1c` | Opus：`sailor_v1_t3_review.md`      | PASS-WITH-CHANGES（F1–F3 已修、F4–F8 記錄並接受；程式 commits `8db61061`…`f2bcea17`；`--smoke` 的 collect / encode / anchors / imagine / label / test_anchors / test_imagine / test_label / train 全部通過，report 依設計在 `gate: {}` 上 fail closed） |




## 9. 修訂紀錄

- T0：初稿。
- T1（Opus 審核）：審核意見與逐條抽查見 `sailor_v1_code_review.md` §5；結論 **PASS-WITH-CHANGES**。以下是**改變或收緊 §2–§6 既有文字**的決定，其餘（γ=0.995、B=200、β 候選、V 用 β=0 訓練、RM=真實 sim coverage、prefix 噪聲規則、test anchors 只用於 G6、不做線上 Best-of-N）一律不動。
  - **§2 stage-2 起點（D2）**：anchor 來源 rollout 套用與 stage 1 **相同**的 prefix 噪聲（σ=0.1、`noise_hold=1`、`mode=prefix`、p ∈ {0,10,…,100} per source episode），且只接受 `t ≥ p` 的 anchor。理由：2×2 表的 c−d 與 (d−b)−(c−a) 建立在「stage-1 val 與 stage-2 的狀態分佈只差在想像」，乾淨 anchor 會多出一個無法歸因的分佈差異來源。噪聲在 `place_body` 重新擺放時已被抹掉，只透過「取樣到哪些狀態」起作用。
  - **§2 網格（D7）**：表格「coverage / CoG 網格 每 5 步，到 300」修訂為「**coverage：每步記錄，到 300**（新 `cov_per_step`）；**CoG / agent 網格：每 5 步，到 100**；影格網格：每 5 步，到 100」。100 步之後的 CoG / agent 沒有下游消費者。
  - **§2 stage-2 節拍與 anchor 的有效時刻**：lead-in 強制 `n_action_steps=10`（D5(b)；ckpt 實測是 8，會讓 anchor 落在 chunk 中間）。並註記：imagined branch 的實際起點是「取樣到的 anchor 狀態**再往後 10 步**」，所以 stage-2 的有效 t 是 `t + 10 + 5r`，且 anchor 的速度歷史是從靜止起算的 10 步暫態（stage-1 同一個 t 有 t 步歷史）。
  - **§3 撤回一句推導**：`score = V(z_end) + β·RM(z_end)` **不是** potential-based shaping 的 value（shaped value 是 `E[γ^T·1] + β·E[γ^{T_end}Φ_end] − β·Φ(s)`，符號相反且缺一項；8 個候選在 r=2 各處於不同狀態，`Φ_t` 不是可去掉的常數）。score 的式子、β 候選與「V 用 β=0 訓練」全部保留，但改定性為 **greedy coverage bonus**。`cov_end` / `T_succ` 欄位仍要存，用途改為標籤診斷與表 D。
  - **§3 標籤邊界（D3）**：`T > B` 但最終成功 → `y=0`、`v_target=0`、`cov_end` 取 `t+B` 處；`T` 欄位仍存真值以便統計被窗口切掉的成功數。未成功 → `T=-1`。兩個 stage 都額外輸出統一的 `T` / `cov_end` / `v_target` 欄位（D1）。
  - **§4 Normalizer（D6）**：§4 寫的「只在 fit split 擬合」為準，**改程式**（現行 `train_value_model.py:556-557` 是在 fit+es 上擬合），並額外排除 stage-2 `wall_artifact` 列（那些列已被 loss 丟掉）。`critic/value_model.py:256-257` OracleEnsemble 聲稱「fit independently」的 docstring 是錯的（實際與 ValueEnsemble 共用同一個 normalizer 物件），一併更正。
  - **§5 表 A 列定義（D4 / 審核 O2 / O8）**：(i) 2×2 表新增**第三欄「stage-2 r=0（真實影格、stage-2 狀態分佈）」**——`(r=0) − (stage-1)` 隔離狀態分佈位移，`(r≥2) − (r=0)` 隔離想像本身；r=0 每個 anchor 只有一列、σ 一律 0.0，CI 會寬。(ii) **「WM 幻覺」的定義收緊為「block pose 的 WM 幻覺」**：兩個 stage 的 `agent_state` 都是真值（stage-2 用 `agent_state_hist[r]`，與 sim 在 1e-6 內相等），所以 b 格的 ValueNet 也拿到真實 agent pos/vel，WM 誤差只能透過 latent 影像進來（這對線上用法是忠實的，線上 agent 狀態同樣已知）。(iii) 欄內 Value−Oracle 差（a vs c、b vs d 是同一批列）必須用**成對** grouped bootstrap；跨欄差維持獨立重抽；差中之差用「兩欄獨立重抽、欄內成對」。
  - **§5 表 B G1（D4）**：ECE 改對 `v_target` 算，定義為在 G1 列集合上取 `p_i = v_mean_all_i`，[0,1] 10 個等寬 bin，`ECE_soft = Σ_b (n_b/N)·|mean_{S_b}(v_target) − mean_{S_b}(p)|`；E1 門檻掛在 `ECE_soft`，另無門檻回報對二元 `y` 的 `ECE_bin`。可靠度圖 y 軸改成 `mean v_target`。AUROC / within-anchor AUROC 仍對二元 `y`。`--overfit-test` 繼續用二元 `y`（soft target 的 BCE 下界是目標的熵，`bce<0.05` 必然 FAIL）。
  - **§5 表 B G4（D4）**：改為 `|V − v_target|` 的 top/bottom std quintile 差，另無門檻回報二元版。
  - **§5 表 C 候選集與 tie-break（審核 O1）**：候選集定義為 **4 支 r=0 base chunk × σ∈{0,0.1} = 8 個**，其中 σ=0.1 的候選 j 是候選 j **加噪聲後的同一支 chunk**（兩次 σ run 共用同一個 `--torch-seed` 並重播同一份 lead-in，所以 8 個候選證明性地共享同一個 anchor 狀態）。因此「隨機挑」**不等於**「base 取樣的成功率」，原括號註「= base 取樣」撤回，改為「在這個固定的 8 候選集上均勻挑」。tie-break：分數相同時取 `(sigma, cand)` 升序（先 σ=0、再低 cand）。分數用 `v_mean_all + β·rm_mean_all`（**不用** `v_mean2`：`_random_pair` 在 `rng=None` 時用 OS entropy 且整批共用一組 member，不可重現；`v_mean2` 只作固定 seed 的敏感度檢查）。
  - **§5 表 B G6 / §6 T7（審核 O5–O7）**：分析單位 = test anchor。`S_random(i) = mean_j y_ij`（8 候選平均，隨機挑的精確期望，不再抽一次）；`S_β(i) = y_{i,argmax}`；`S_oracle(i) = max_j y_ij`；`Δ_β(i) = S_β(i) − S_random(i)`。**gate 掛在** `mean_i Δ_β` **的 one-sided 95% 下界 > 0**（對 anchor 的成對 bootstrap，10000 次；方向在看資料前預先承諾為「優於隨機」），`gap closed` 只作描述統計（其分母在重抽中可能趨近 0，比值 CI 不可作 gate）。必須回報 `n_used`、`n_informative`（`0 < Σ_j y_ij < 8` 的 anchor 數）、`SD(Δ)`；**若** `n_informative < 40`**，報告記為「underpowered」而非「離線不顯著」**，T7 的停損不適用。G6 **保留** `pre_success` **候選並記 y=1**（這是對「pre_success 一律排除 gate metric」的預先註冊例外，理由：G6 量選擇成功率而非判別力）；任一候選是 `wall_artifact` 或缺列 → 整個 anchor 退出 G6 並回報退出數。β 仍只在 val 上選（用同一個 `mean Δ_β`），三個 β 的 val 數字全部回報，只有勝出者在 test 上算一次。
  - **§5 表 D 來源（審核 O3）**：改用 **stage-2 的** `fit` **+** `es` **split anchors**（隨機 25 個，抽樣 seed 寫進 config），**不得用 val**——原文「20–30 個 val 狀態」與「T3 在看到 val 之前填入」互相矛盾。統計量必須在 G3 的同一個 grouping（`seed_id`, r, σ）**組內**計算：每組 k=4 列各跑兩次，回報邊際 y 一致率與**組內** `AUROC(y1 當標籤, y2 當分數)` 的平均 + bootstrap CI；只有後者才是 G3 的天花板。val 的天花板在 T6 只當一致性檢查，不得反過來調 T3。
  - **§5 / §6 分層回報（D8）**：不對 t 重新加權，只在報告中按 t ∈ {10–40, 50–70, 80–100} 分層回報 n / y 比例 / AUROC / `ECE_soft`，並附 t 直方圖；任一層 n < 300 或 y 比例 <5%／>95% 標記為不可解讀。噪聲分層一律用新欄位 `noised = (sigma>0) & (perturb_p>0)`，不用 `sigma>0`——prefix 規則下 `p=0`（約 1/11 的 episode）其實完全乾淨。
  - **§7 新增 seed namespace**：`sailor_v1_label_cont: 770000`（stage-2 接續的 policy 取樣；現行寫死 `seed=0`），表 D 的重跑用 `770001`。接續標籤依賴 batch 組成（`torch.manual_seed(seed + batch_start)`，整批共用一條 stream），所以 `label --batch` 必須寫進 config、train 與 test anchors 用同一個值、表 D 兩次重跑除 `--cont-seed` 外組成完全相同。
  - **§8 事實更正**：base ckpt 已就緒，實測 `horizon=16`、`n_action_steps=8`、`n_obs_steps=1`、`drop_n_last_frames=8`。`--replan-every-rounds 2` 的奇數 round lookahead 需要 `horizon ≥ 15`，**剛好滿足（只多 1 步）**，driver 必須 assert；該 lookahead 落在 policy 部署時從不執行的 horizon 尾段，列為報告 caveat（D5(a)：接受此近似，因為被執行的 5 步與 R=1 逐位元相同、標籤來自精確 sim 重播）。
- T2（模擬器檢查）：結果在 `~/datastorage/nanowm_pusht_aug/value_model/sailor_v1/t2_checks/`（`T2_SUMMARY.md`、`t2a_success.json`、`t2b_timing.json`、`t2c_pixels.json`、`episodes.npz`；程式 `critic/t2_checks.py`，commit `25e716f3`；模擬是在該 commit 之前的工作樹上跑的，之後只有 ruff 格式 / import 排序變動）。設定：base policy、na=10、300 步、1000 個成對起點（`reset(seed=760000+i)`）、policy seed 760000。
  - **(a)** 乾淨 685/1000 = 68.5%，噪聲（σ=0.1、`noise_hold=1`、整條 episode、`max_z=3`）698/1000 = 69.8%；成對 Δ = **+1.30 pp**，95% CI（10000 次成對 bootstrap）[−2.40, +5.00]；不一致對：僅乾淨成功 165、僅噪聲成功 178，McNemar 精確 p = 0.52。平均 `success_step` 181.9（乾淨）vs 182.0（噪聲）。**依 §6 規則 |Δ| = 1.30 pp ≤ 2 pp → 採備案：整條加噪聲、不丟 query、stage-2 接續也加噪聲**（程式面 `perturb_p ≥ episode_steps` 的 prefix 模式即可）。誠實註記：判準是點估計；CI 寬度 ±3.7 pp，並沒有排除 >2 pp 的差異，只是沒有任何證據顯示噪聲降低成功率（點估計方向反而是 +）。此決定寫入後不再變（§2 備案條款）。
  - **(b) 旗標成立**：乾淨臂 315 個失敗 episode 中 **89.2%** 的「決定時刻」（coverage 最後一次單步上升 >0.01）在 t>100（>200：83.8%），遠超 20% 門檻；失敗 episode 的最大 coverage 步中位數 210。成功時間佔全部乾淨 episode：≤100 步 6.4%、≤200 步 42.6%、≤300 步 68.5%。query t∈{10,…,100} 只涵蓋 episode 的前三分之一，**query 範圍的侷限**：V 看到的狀態多半是「還在推」的早期狀態，晚期決策點（t>100）不在訓練分佈內；不改設計。各 t 的預期類別平衡（乾淨臂，扣除 wall artifact 與已終止者；`y=1` ⇔ 之後 200 步內成功）：t=10: 0.468、20: 0.502、30: 0.532、40: 0.559、50: 0.577、60: 0.602、70: 0.626、80: 0.644、90: 0.662、100: 0.677（n_alive 980→925）；皆在 5%–95% 內，另有 T>200 才成功（被窗口切掉）的列：t=10 有 222、t=100 為 0。
  - **(c)** `SIGMA_PX` = [20.847, 20.675] px/步；0.1σ（`noise_hold=1`）下每步目標偏移 |noised − clean|：mean 2.86 / median 2.49 / p95 5.41 px（10854 個乾淨臂 chunk；200 個抽樣：2.83 / 2.53 / 5.36）；10 個執行步之後 agent 的實際偏移（PD 模型）final-step mean 1.55 / median 1.23 / p95 2.78 px；真實 sim 重播 200 個 chunk：agent 1.51 / 1.22 / 2.77 px，block CoG 0.56 / 0.32 / 1.82 px。即 0.1σ 噪聲幾乎不改變狀態軌跡（agent 平均 ≈1.5 px，block ≈0.6 px）。**N15.3 的 cap/clip 量測（level=0）**：`inject_noise(level=0)` 改動 2.61% 的步（`max_z` cap 2.23%、`AGENT_BOX` clip 0.38%），8.82% 的 chunk 至少有一步被改，被改的步平均改動 12.0 px（p95 36.6 px），但 10 步後 agent 偏移 mean 0.31 / p95 0.12 px——影響很小。
  - 附帶：`critic/labels.py`（`v_target`、`cov_end_index`、`keep_query`、`success_T`、`chunks_active`、`forward_fill_cov`）、`rollout_batch` 新增 `perturb_mode` / `cov_per_step` / `chunk_hook`（T3 清單 #1/#2、M8、M9 已先行實作並有測試）；`test_cross_venv_imports.py` 加入 `critic.labels`。
- T3（資料收集修改；起點 `48a37b1c`；Opus 審核見 `sailor_v1_t3_review.md`）：以下是 pre-reg 沒有明文、T3 採保守選擇的實作決定，以及 T2(a) 備案在程式上的對應。§2–§6 與 T1 / T2 的既有規則一律不變。
  - **備案的程式對應**：stage 1 `perturb: {sigmas: [0.1], noise_hold: 1, mode: prefix, switch: all}` ⇒ `perturb_p = episode_steps = 300`，每個 chunk 都加噪聲；`stage1.query_keep: all`，不丟 query（`keep_query` 的 `t ≥ p` 在 p=300 下會丟掉**所有** query，所以 query 規則是顯式 config，不從 p 推導）。anchor 來源 rollout 套同一組設定（D2）。stage-2 接續：`label.cont_perturb = {sigma: 0.1, noise_hold: 1, max_z: 3.0}`，整段 B=200 步都加（`perturb_p = B`），σ=0 與 σ=0.1 的 branch 一律加——即「stage-2 接續也加噪聲」。主設計（`switch: random` + `query_keep: after_switch`）程式仍保留可跑，本計劃不用。
  - `noised` **欄位在備案下的語意**：stage 1 的 `q_noised = (sigma>0) & (perturb_p>0)` 全為 True，沒有資訊量；stage 2 的 `noised = (inject_sigma > 0)` 只描述 query 之前的想像段，接續一律有噪聲。D8 的「按噪聲分層」在 stage 1 退化成單一層。
  - **stage-2 lead-in 保持乾淨**（pre-reg 未規定）：anchor 之後 10 步的 lead-in 用乾淨 base policy（`--lead-na 10`），且所有 σ run 共用同一份（O1 的必要條件）；σ=0 branch 的想像段也乾淨（pre-reg `inject_sigma ∈ {0.0, 0.1}`）。理由：lead-in 若要加噪聲，σ run 之間就必須共用同一串噪聲，等於再多一套 seed 機制；而 T2(c) 顯示 0.1σ 在 10 步後只讓 agent 偏 ≈1.5 px、block ≈0.6 px，與 stage 1「歷史全程有噪聲」的差異可忽略。列為報告 caveat。
  - **anchor 的拒絕規則與來源長度**：`success_step ≤ t` 的來源 episode 不產生 anchor（stage 1 對 `s == t` 保留為 pre_success；anchor 必須是決策點）。每個來源 episode 最多一個 anchor，t 從 {10,…,100} 均勻抽，且每個來源 episode 恰好消耗一次抽樣（不論是否被拒絕）。來源 rollout 只跑到 `max(t_choices) = 100` 步（前 100 步的 policy / 噪聲 stream 與跑滿 300 步相同），所以 anchor 的 `wall_artifact` 只看 0–100 步；label 另對 replay + 接續重算 wall。
  - **namespace 內的 seed 位移**（§7 只給 namespace 起點）：`+0` = policy torch seed（`rollout_batch` 的 `seed + batch_start`）與 sigma / p 的抽樣，`+1` = 來源 rollout 的動作噪聲 rng，`+2` = anchor 時刻抽樣，`+3` = driver 的 `--noise-seed`；driver 的 `--torch-seed` = namespace 起點，所有 σ run 共用。即 stage 1：730000(+batch_start) / 730000 / 730001；stage-2 anchors：740000…740003；test anchors：750000…750003。接續：torch `770000 + batch_start`、噪聲 rng 770001；表 D 重跑 `--cont-seed 770001`（其噪聲 rng 770002），表 D 的 anchor 抽樣 seed 770003。test anchors 的接續與 train 用同一個 `cont_seed` 與 `label.batch = 100`（O9：兩者的標籤程序完全相同）。
  - **label 的 job 排序**：先依 `(seed_id, sigma, cand, r)` 排序再切 batch（採納 O9 的建議）。這項與 `cov = sim_cov` 同樣會改變**舊 topband config 重新 label** 的結果；topband 既有輸出不動，本計劃也不重跑，僅此記錄。
  - **配對候選由程式保證**：label 對同一個 `(seed_index, cand)` 的各 σ run 斷言 `chunk0_clean`（加噪前的 round-0 chunk）相差 ≤ 1e-3 px（smoke 實測為 0，逐位元相同），讓 T1 修訂「σ=0.1 的候選 j 就是候選 j 加噪聲」不只靠論證。若 T4 在此 assert 失敗（例如 GPU 非決定性），停下回報，不放寬容差。
  - **T4 的 stage 範圍**：T4 只跑 `collect encode anchors imagine label`；`test_anchors / test_imagine / test_label` 依 plan §5 留到 T7，所以 test anchors 的標籤在 T5 寫入門檻之前不會產生。
- T5（模型與 report 修改；起點 `4ec4908b`；程式 commits `4b011576`…`d2179c00`，變更說明 `sailor_v1_t5_changes.md`）：**在看到任何 val 之前**（T5 只跑過 smoke 與 3 步的 shape 檢查，沒有完整訓練、沒有 val 指標）由使用者決定降低評估複雜度。以下改寫 §5 表 B 與 §6 的 T6 停損，其餘不動。
  - **G1 / G2 / G3 / G5 改為無門檻的診斷**：這四個指標的合理門檻目前無法事前決定，所以不設門檻、不判通過與否，只照常計算並回報（AUROC、`ECE_soft` / `ECE_bin`、within-anchor AUROC、RM MAE，含 CI）。表 A（2×2）與 real-vs-imagined 區塊照常輸出。config 為 `gate: {diagnostic_only: true}`；`value_gate_report.py` 在此模式下這四個 gate 的 `meets_threshold = None`、不 fail closed，且同時給門檻會 assert 失敗。topband config 仍用門檻，行為不變。
  - **唯一的決策 gate 是 G6**：規則沿用 T1 修訂（test anchors、`mean_i Δ_β` 的 one-sided 95% 下界 > 0、β 在 val 上選、`n_informative < 40` 記為 underpowered）。G4 的固定規則照常回報，但不是停損條件。
  - **§6 的 T6 停損撤回**：「G2 或 G3 未過 → 停損」不再適用；T6 完成後直接進 T7。G6 未過時，才用 G1–G5、表 A 與 real-vs-imagined 找原因。
  - **表 D（雜訊天花板）改為選用**：它原本是 G3 門檻的上限；G3 不再有門檻，所以 T6 不必產生 `labels_nc0/1.npz`，只在 G6 結果需要解讀時才跑。
  - **只看標籤的 headroom 檢查**（stage-2 訓練 anchors 的 T4 標籤，不含任何模型輸出）：r=2 時 250 個 anchor 都有 8 個候選，226 個（90%）候選間 y 不一致；隨機挑的平均成功率 0.631，事後挑最好 0.984。r=4：223 個（89%）、0.664、0.984。即挑選空間存在；事後最好含單次接續的運氣，所以判定看 Δ（相對隨機），不看 gap closed。
  - **T5 審核問題的處理**：D6（Normalizer 只在 fit split、排除 wall 列）屬 T1 已註冊的修訂，接受。stage-1 的 within-anchor AUROC 格（依 episode 分組、混合不同 t）保留但只作描述。D8 的 t 分層回報延後，只在 G6 結果需要解讀時補。
- T6（訓練 + report；HEAD `d6ef9c35`；log `t6_train_report.log`，報告 `gate_report.md/json/png`，皆在 `out_dir`）：27,632 列 → fit 19,895 / es 2,202 / val 5,535。sanity：shuffled-label AUROC 0.486、permuted-val 0.506。以下為無門檻的診斷數字（val）。
  - G1：stage-1 reset AUROC 0.658 [0.624, 0.692]，`ECE_soft` 0.012。G2：stage-2 r≥2 AUROC 0.600 [0.543, 0.651]。G3：within-anchor AUROC 0.509（275 組）。G4：0.013 [−0.002, 0.028]，固定規則未過。G5：RM MAE 0.034（stage 1）/ 0.040（stage 2）。
  - OracleNet：stage-1 AUROC 0.680，stage-2 r≥2 0.585，within-anchor **0.470**。真實狀態也幾乎無法在同一 anchor 的候選間排序，表示候選之間的 y 差異可能大多來自接續的隨機性；這正是表 D 要量的。
  - **OracleNet 未收斂**：5 個 member 都跑滿 `max_epochs=40`（ValueNet 在 21–36 epoch early stop），RM MAE 0.094 反而比 ValueNet 的 0.034 差（coverage 是 block pose 的確定函數）。表 A 中 Oracle 為基準的分解（c−a、(d−b)−(c−a) 為負）因此不可解讀。
  - **事後修訂（看過 val 之後，標記為 post-hoc；HEAD `adbf5dfc`）**：OracleNet 重訓，新增 `train.oracle_max_epochs: 300`（只影響 OracleNet；ValueNet 與 shuffled 控制不動）。結果：5 個 member 在 59–86 epoch early stop，RM MAE 0.094 → 0.069（仍高於 ValueNet 的 0.034），stage-1 AUROC 0.685、within-anchor 0.904，stage-2 r≥2 AUROC 0.588、within-anchor 0.487。`train` stage 會整個重跑，ValueNet 也以相同 seed 重訓，GPU 非決定性使數字略有變動（stage-1 0.663、stage-2 r≥2 0.606、within-anchor 0.510、G4 0.024 [0.008, 0.040]，這次 G4 固定規則過了）。第一次的輸出保留在 `model_t6_oracle40/`。OracleNet 只是參考線，不影響 G6 的判定。
  - **表 D（雜訊天花板，同 25 個 fit+es anchors、各重跑一次接續，cont_seed 770000 / 770001）：y 一致率 0.599 [0.561, 0.636]；組內 AUROC(y1, y2) 0.514 [0.464, 0.566]（135 個有混合 y 的組）。** 當 base rate ≈ 0.63 時，兩次獨立抽樣的隨機一致率約 0.53，所以同一個 branch 重跑兩次的結果幾乎不比抽籤一致。ValueNet 的 within-anchor 0.510 與這個天花板（0.514）相同，OracleNet 的 0.487 也在其內。結論：**同一 anchor 的 8 個候選，單次接續的 y 在組內幾乎是純雜訊，G3 的 0.51 不是模型缺陷，而是標籤無法分辨候選。** 這同樣限制 G6：用單次接續的 y 去量「挑出的候選是否比隨機好」，信噪比極低（val 的 SD(Δ) ≈ 0.45）。
  - val 上的 G6 預覽（50 個 val anchors，47 informative；只供選 β，不是決策）。第一次訓練：ValueNet 三個 β 的 Δ 都是 0.075 [−0.050, 0.195]，下界 −0.030；OracleNet β=0.3 的 Δ 0.115，下界 0.030；SD(Δ) ≈ 0.445，所以 100 個 test anchors 的下界 > 0 約需 Δ ≥ 0.073。**OracleNet 重訓後（最新 `gate_report.md`）**：ValueNet β=0 / 0.1 的 Δ 都是 0.015 [−0.107, 0.138]、β=0.3 為 −0.005；OracleNet 0.015–0.035，下界皆 < 0。ValueNet 的 Δ 從 0.075 變成 0.015，是同樣 seed 重訓後 Δ 在雜訊內跳動（SD(Δ)≈0.45、n=47，標準誤約 0.065），再次顯示 val 的 G6 預覽無法分辨 ±0.08 以內的差異；report 選出 β=0。

- T7（test anchors + G6；HEAD `6a6dec01`；log `t7.log`，報告 `gate_report.md/json/png`，皆在 `out_dir`）：依預先註冊規則跑完 `test_anchors / test_imagine / test_label`（100 個 test anchors，seed 750000，label 約 33 分鐘）。模型沿用 T6 重訓後的 `model/`；β 在 val 上選出 0.0。
  - **G6（test，ValueNet β=0）未通過**：`n_used=94`（6 個 anchor 因 wall_artifact 退出）、`n_informative=83`（非 underpowered）。`S_random`=0.586、`S_selected`=0.628、`S_hindsight`=0.957；`mean Δ`=+0.041，95% CI [−0.044, 0.128]，SD(Δ)=0.429，one-sided 95% 下界 **−0.029**（需 > 0）；gap closed 0.111 [−0.121, 0.341]。
  - OracleNet（真實狀態，β=0）：`S_selected`=0.543，`mean Δ`=−0.044 [−0.133, 0.044]，下界 −0.118，即連真實狀態也不優於隨機，與表 D 的雜訊天花板一致（同 anchor 候選的成敗大多是接續的隨機性）。
  - G4 在 T7 後的 report 為 True（與重訓後的 T6 一致）；G1/G2/G3/G5 仍為無門檻診斷。
  - **依 §6 / T1 修訂，T7 的停損成立**：選出的 β 在 test 上下界 ≤ 0，結案報告記為「離線不顯著」（檢定力足夠，非 underpowered）。解讀：點估計 +4 pp，與雜訊同量級；表 D 顯示單次接續標籤在 anchor 內近乎雜訊，所以此結果不能區分「V 排序無用」與「標籤太吵」。後續設計見 `sailor_v2_plan.md`。
