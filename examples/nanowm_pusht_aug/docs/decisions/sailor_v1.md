# SAILOR v1 — value / reward model 預先註冊文件

狀態：**草稿（T0）**。Opus 審核並在 T5 填入 gate 門檻後才允許 full run（T4 之後的訓練與評估）。
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

| 項目 | stage 1（真實影格） | stage 2（想像影格） |
| --- | --- | --- |
| 起點 | 隨機 `env.reset(seed=730000+i)`，n=2000 | anchor：另一批 reset rollout（seed 740000+）在 t∈{10,…,100} 抽的狀態，n=250 |
| 執行者 | base policy（`ckpt.base`），`n_action_steps=10` | NanoWM 想像 k=4 個 branch，每 **10 步（2 round）** 重取樣 chunk |
| 噪聲 | 每步 0.1σ（`noise_hold=1`，`SIGMA_PX≈20.8 px`），**只加在切換點 p 之前**；p ∈ {0,10,…,100} 每 episode 隨機一個 | `inject_sigma ∈ {0.0, 0.1}`，每步；只加在 query 之前的 chunk |
| query | t = 10,…,100（每 episode 10 個），只保留 **t ≥ p** | r ∈ {0,2,4,6,8}（= t 0,10,…,40 步） |
| 標籤來源 | 同一條 episode 的後續（乾淨 base policy） | 精確 sim 重播（1e-6 assert）後乾淨 base policy 接續 B=200 步 |
| episode 上限 | 300（gym 上限；不再 310） | 接續 200 步 |
| 影格網格 | 每 5 步，到 t=100 | NanoWM 每 round 一格 |
| coverage / CoG 網格 | 每 5 步，**到 300**（供 Φ_end） | 接續期間每 5 步 |

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

|  | stage-1 val（真實影格） | stage-2 val r≥2（想像影格） |
| --- | --- | --- |
| ValueNet（latent） | a | b |
| OracleNet（state） | c | d |

分解：c−a = 編碼器損失；d−b = 編碼器 + WM 幻覺；(d−b)−(c−a) ≈ WM 幻覺；c−d = 想像分布本身較難。

### 表 B：gate

| gate | 指標 | 門檻 | 資料 |
| --- | --- | --- | --- |
| G1 | stage-1 reset AUROC ≥ T1，ECE ≤ E1 | TBD（Opus） | val |
| G2 | stage-2 r≥2 AUROC ≥ T2 | TBD | val |
| G3 | within-anchor AUROC ≥ T3（T3 < 表 D 的雜訊天花板） | TBD | val |
| G4 | ensemble std 有資訊量：|V−y| 的 top/bottom quintile 差 > 0 | 固定 | val |
| G5 | RM MAE（對真實 cov）stage1 ≤ M1、stage2 ≤ M2 | TBD | val |
| G6 | 離線 Best-of-N gap closed 的 95% CI 不含 0 | 固定 | **test anchors** |

### 表 C：G6 離線 Best-of-N

每個 anchor 8 個候選（k=4 × σ∈{0,0.1}），r=2 的 query；比較「隨機挑」「ValueNet β∈{0,0.1,0.3}」「OracleNet」「事後最好」。
`gap closed = (S_β − S_random) / (S_oracle − S_random)`，S_oracle = 事後最好。**β 在 val 上選（最高 S_β）；只有選出的那一個 β 在 test anchors 上報 CI。**

### 表 D：雜訊天花板

20–30 個 val 狀態各重跑兩次乾淨接續，報 y 一致率與 AUROC(y1, y2)；G3 的 T3 必須低於此值。

## 6. 決策規則與停損

- T2(a)：|Δ成功率| ≤ 2 pp → 採備案（整條噴噪聲）；否則採 prefix 規則。T2(b)：若 ≥ 20% 的失敗 episode 在 t>100 才「決定」（coverage 最後一次上升在 t>100），在修訂紀錄註明 query 範圍的侷限但不改設計。
- T6：G2 **或** G3 未過 → 停損，轉去處理 NanoWM 準確度，不進 T7。
- T7：選出的 β 在 test 上 gap closed CI 含 0 → 停損，結案報告記為「離線不顯著」。
- 任何門檻在看到 val 之後都不得下修。

## 7. Seed namespace（`configs/nanowm_pusht_common.yaml`）

| 用途 | seed |
| --- | --- |
| stage 1 reset episode | `sailor_v1_stage1_reset: 730000` |
| stage 2 anchor 來源 rollout | `sailor_v1_stage2_anchor: 740000` |
| G6 test anchor 來源 rollout | `sailor_v1_test_anchor: 750000` |
| T2 模擬器檢查 | `sailor_v1_t2_checks: 760000` |
| stage-2 接續 policy 取樣（label；表 D 重跑 +1） | `sailor_v1_label_cont: 770000` |

## 8. Phase 起點 sha 與審核紀錄

| phase | 起點 sha | 審核 | 結果 |
| --- | --- | --- | --- |
| T0 | `e46cc24c` | — | 完成（`f9eeb5d6`） |
| T1 | `f9eeb5d6` | Opus：`sailor_v1_code_review.md` §5 | PASS-WITH-CHANGES（M1–M12 併入 T3/T5） |

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
    - **§5 表 B G6 / §6 T7（審核 O5–O7）**：分析單位 = test anchor。`S_random(i) = mean_j y_ij`（8 候選平均，隨機挑的精確期望，不再抽一次）；`S_β(i) = y_{i,argmax}`；`S_oracle(i) = max_j y_ij`；`Δ_β(i) = S_β(i) − S_random(i)`。**gate 掛在 `mean_i Δ_β` 的 one-sided 95% 下界 > 0**（對 anchor 的成對 bootstrap，10000 次；方向在看資料前預先承諾為「優於隨機」），`gap closed` 只作描述統計（其分母在重抽中可能趨近 0，比值 CI 不可作 gate）。必須回報 `n_used`、`n_informative`（`0 < Σ_j y_ij < 8` 的 anchor 數）、`SD(Δ)`；**若 `n_informative < 40`，報告記為「underpowered」而非「離線不顯著」**，T7 的停損不適用。G6 **保留 `pre_success` 候選並記 y=1**（這是對「pre_success 一律排除 gate metric」的預先註冊例外，理由：G6 量選擇成功率而非判別力）；任一候選是 `wall_artifact` 或缺列 → 整個 anchor 退出 G6 並回報退出數。β 仍只在 val 上選（用同一個 `mean Δ_β`），三個 β 的 val 數字全部回報，只有勝出者在 test 上算一次。
    - **§5 表 D 來源（審核 O3）**：改用 **stage-2 的 `fit` + `es` split anchors**（隨機 25 個，抽樣 seed 寫進 config），**不得用 val**——原文「20–30 個 val 狀態」與「T3 在看到 val 之前填入」互相矛盾。統計量必須在 G3 的同一個 grouping（`seed_id`, r, σ）**組內**計算：每組 k=4 列各跑兩次，回報邊際 y 一致率與**組內** `AUROC(y1 當標籤, y2 當分數)` 的平均 + bootstrap CI；只有後者才是 G3 的天花板。val 的天花板在 T6 只當一致性檢查，不得反過來調 T3。
    - **§5 / §6 分層回報（D8）**：不對 t 重新加權，只在報告中按 t ∈ {10–40, 50–70, 80–100} 分層回報 n / y 比例 / AUROC / `ECE_soft`，並附 t 直方圖；任一層 n < 300 或 y 比例 <5%／>95% 標記為不可解讀。噪聲分層一律用新欄位 `noised = (sigma>0) & (perturb_p>0)`，不用 `sigma>0`——prefix 規則下 `p=0`（約 1/11 的 episode）其實完全乾淨。
    - **§7 新增 seed namespace**：`sailor_v1_label_cont: 770000`（stage-2 接續的 policy 取樣；現行寫死 `seed=0`），表 D 的重跑用 `770001`。接續標籤依賴 batch 組成（`torch.manual_seed(seed + batch_start)`，整批共用一條 stream），所以 `label --batch` 必須寫進 config、train 與 test anchors 用同一個值、表 D 兩次重跑除 `--cont-seed` 外組成完全相同。
    - **§8 事實更正**：base ckpt 已就緒，實測 `horizon=16`、`n_action_steps=8`、`n_obs_steps=1`、`drop_n_last_frames=8`。`--replan-every-rounds 2` 的奇數 round lookahead 需要 `horizon ≥ 15`，**剛好滿足（只多 1 步）**，driver 必須 assert；該 lookahead 落在 policy 部署時從不執行的 horizon 尾段，列為報告 caveat（D5(a)：接受此近似，因為被執行的 5 步與 R=1 逐位元相同、標籤來自精確 sim 重播）。
