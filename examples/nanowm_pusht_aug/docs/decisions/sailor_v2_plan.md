# SAILOR v2：貼近微調用途的 value model 測試（草稿）

狀態：**草稿**，尚未預先註冊。v1 的結果與修訂見 `sailor_v1.md`（尤其 §9 的 T5、T6）。
v2 只在 v1 的 T7（G6）與 v1 微調對照跑完之後才定稿；定稿前不得看 v2 的任何 val 數字。

## 1. 為什麼需要 v2

實際用途：在 NanoWM 裡 rollout diffusion policy 並加入隨機動作，用 V（+ RM）替每條想像 rollout 打分，再用高分 rollout 微調 policy。v1 的 T6 顯示，現在的測試和這個用途有三個落差：

1. **候選擾動太小**：v1 的候選是 `inject_sigma ∈ {0, 0.1}`、`noise_hold=1`。T2(c) 量到 0.1σ 在 10 步後只讓 agent 偏 ≈1.5 px、block ≈0.6 px，8 個候選幾乎是同一個狀態。微調用的想像 pool（`configs/cupid_select_topband.yaml` 的 `generate`）是 `inject_sigma ∈ {0.5, 1.0}`、`noise_hold=5`，大 5–10 倍。
2. **單次接續的標籤太吵**：表 D 中同一候選重跑接續，y 一致率 0.599、組內 AUROC(y1, y2) 0.514。單次 0/1 標籤在組內幾乎是雜訊，所以 G3 / G6 無法分辨「V 排對了期望成功率」和「V 在亂猜」。
3. **評估沒有測到微調本身**：G6 只回答「挑選是否有用」，不回答「用挑出的 rollout 微調後 policy 是否變好」。

## 2. 設計變更（相對 v1）

| 項目 | v1 | v2 |
| --- | --- | --- |
| 候選擾動 | `inject_sigma ∈ {0, 0.1}`、`noise_hold=1` | 與微調 pool 相同：`inject_sigma ∈ {0.5, 1.0}`、`noise_hold=5`、`max_z=3`（另保留 σ=0 一支當基準） |
| 標籤 | 每個 (候選, r) 一次接續，y ∈ {0,1} | 每個 (候選, r) **M=8 次**接續（只換 `cont_seed`，batch 組成固定），標籤 `p̂ = mean_m y_m`、`v̂ = mean_m v_target_m` |
| V 的訓練目標 | 對單次 `v_target` 做 BCE | 對 `v̂` 做 soft-label BCE（期望值，雜訊小 ≈√8 倍） |
| 主要指標 | G6：`mean(S_β − S_random)` | **組內排序**：每個 anchor 內 V 與 `p̂` 的 Spearman ρ（平均 + CI），以及 G6 改用 `p̂`：`Δ = p̂_{argmax V} − mean_j p̂_j` |
| 雜訊天花板 | 另跑兩次接續（表 D） | 由 M 次接續的 split-half（4 vs 4）直接估計，不必另跑 |
| advantage | 無 | `A_j = V_j − mean_{j'} V_{j'}`（同 anchor），回報 A 與 `p̂_j − mean p̂` 的相關 |
| 端到端 | 無 | V 挑選 vs 隨機挑選的微調對照（§4） |

不變：γ=0.995、B=200、RM = 真實 sim coverage、grouped 三切分、shuffled 對照、OracleNet、`gate: {diagnostic_only: true}`（G1/G2/G3/G5 只回報數字）。

## 3. 成本估計（依 T4 實測）

- 想像：T4 是 250 anchors × 8 候選 ≈ 2 h 17 m（與 collect 並行時）。v2 若用 100 個 anchors ≈ 50 分鐘。
- 標籤：T4 的 label 是 2000 個 branch × 5 個 round ≈ 1 h 13 m。M=8 倍 → 250 anchors 約 9.7 h，**100 anchors 約 3.9 h**。
- 建議先用 25 個 anchors 做 pilot（約 1 h），確認 split-half 天花板明顯高於 0.5 再擴大。

## 4. 端到端微調對照

- **pool**：v2 的想像候選（或 v1 的 test anchors，見 v1 的微調對照），訓練 V 時沒看過的 anchors。
- **arm V**：每個 anchor 取 V（或 advantage）最高的候選；**arm R**：每個 anchor 均勻隨機取一個候選（固定 seed）。兩個 arm 的 pair 數、真實資料、normalization（pin 在 real-only stats）、微調超參數完全相同，只差挑哪些候選。
- **微調**：沿用 `finetune_cl.sh`（從 `ckpt.base`、lr 1e-5、batch 64、20k steps、同一 seed）。
- **評估**：random-reset 1000 episodes，成對（同一組起點與 seed），報 arm V − arm R 的成功率差與成對 bootstrap CI；base policy 一併評估當參考。

## 5. 定稿前要決定的事

- 候選擾動是否完全照搬 CUPID pool（`noise_hold=5`），或另掃一個較小的值。
- M=8 是否足夠（pilot 的 split-half 天花板決定）。
- 主要指標用 Spearman ρ 還是 `p̂` 版的 G6；門檻是否也維持 diagnostic-only。
- 微調 pool 的劑量（pair 數）：v1 的 CUPID arm 用 k=500。
