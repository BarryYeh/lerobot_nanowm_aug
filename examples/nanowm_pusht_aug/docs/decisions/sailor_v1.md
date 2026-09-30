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

## 8. Phase 起點 sha 與審核紀錄

| phase | 起點 sha | 審核 | 結果 |
| --- | --- | --- | --- |
| T0 | `e46cc24c` | — | — |

## 9. 修訂紀錄

- T0：初稿。
