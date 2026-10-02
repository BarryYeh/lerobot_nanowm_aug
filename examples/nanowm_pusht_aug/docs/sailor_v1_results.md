# SAILOR v1 實驗結果整理

分支 `feat/sailor-v1-value`，HEAD `914bf507`。規則出處：[`decisions/sailor_v1.md`](decisions/sailor_v1.md)（§5 表格、§9 修訂 T1–T7）。
數字來源：`~/datastorage/nanowm_pusht_aug/value_model/sailor_v1/gate_report.{md,json,png}`（T7 之後重產，含 test G6；T6 當時版本未另存，數字在 §9 T6 條目）。

## 一句話結論

**G6（唯一決策 gate）未通過 → 結案記為「離線不顯著」（檢定力足夠，非 underpowered）。**
ValueNet 在 test anchors 上挑出的候選只比隨機高 +4.1 pp（one-sided 95% 下界 −0.029 < 0）。但表 D 顯示單次接續標籤在 anchor 內幾乎是純雜訊，因此此結果**不能區分**「V 排序無用」與「標籤太吵」。

## 1. 決策 gate：G6（離線 Best-of-N，r=2，8 候選/anchor）

β 在 val 上選出 **0.0**（val 三個 β 差異都在雜訊內）。Test（n_used=94，6 個因 wall_artifact 退出；n_informative=83）：

| 模型 (β=0) | S_random | S_selected | S_hindsight | mean Δ [95% CI] | SD(Δ) | one-sided 95% 下界 | gap closed [CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ValueNet | 0.586 | 0.628 | 0.957 | +0.041 [−0.044, 0.128] | 0.429 | **−0.029** | 0.111 [−0.121, 0.341] |
| OracleNet（真實狀態） | 0.586 | 0.543 | 0.957 | −0.044 [−0.133, 0.044] | 0.438 | −0.118 | −0.118 [−0.353, 0.118] |

- 挑選空間存在：標籤上事後最好 0.957 vs 隨機 0.586（訓練 anchors r=2：90% anchor 的候選 y 不一致）。
- 連使用真實狀態的 OracleNet 也不優於隨機 → 瓶頸在標籤，不在編碼器。
- Val 預覽（50 anchors）：ValueNet Δ=+0.015 [−0.107, 0.138]；重訓前 0.075，重訓後跳動屬雜訊（標準誤 ≈0.065）。

## 2. 為什麼 G6 無法判別：表 D 雜訊天花板

同 25 個 fit+es anchors，各重跑一次乾淨接續（cont_seed 770000 / 770001）：

- y 一致率 **0.599** [0.561, 0.636]（base rate 0.63 時隨機一致率約 0.53）
- 組內 AUROC(y1, y2) **0.514** [0.464, 0.566]（135 個混合 y 的組）

→ 同一 branch 重跑兩次的成敗幾乎不比抽籤一致。ValueNet within-anchor AUROC 0.510、OracleNet 0.487，都落在天花板內，即 G3 的 0.51 不是模型缺陷。

## 3. 無門檻診斷（val）與表 A（2×2）

| 指標 | 值 |
| --- | --- |
| G1 stage-1 reset AUROC / ECE_soft | 0.663 [0.629, 0.697] / 0.013（ECE_bin 0.256） |
| G2 stage-2 r≥2 AUROC | 0.606 [0.549, 0.656] |
| G3 within-anchor AUROC | 0.510（275 組） |
| G4 top-bottom std quintile 差 | 0.024 [0.008, 0.040]，固定規則通過（非停損條件） |
| G5 RM MAE stage-1 / stage-2 | 0.034 / 0.039 |
| stage-2 r=0（真實影格）AUROC | 0.680（n=50，CI 寬 [0.50, 0.83]） |

| 格 | AUROC | within-anchor AUROC | RM MAE |
| --- | --- | --- | --- |
| a ValueNet / stage-1 | 0.663 | 0.806 | 0.034 |
| c OracleNet / stage-1 | 0.685 | 0.904 | 0.069 |
| b ValueNet / stage-2 r≥2 | 0.606 | 0.510 | 0.039 |
| d OracleNet / stage-2 r≥2 | 0.588 | 0.487 | 0.088 |

- 編碼器損失 c−a：AUROC +0.023 [0.006, 0.038]（很小）。
- 「block-pose WM 幻覺」(d−b)−(c−a)：AUROC −0.041 [−0.071, −0.015]。Oracle 沒有獲得比 ValueNet 更多想像資訊，但注意 OracleNet 為事後重訓（見下）且 MAE 仍遠高於 ValueNet，Oracle 為基準的分解只作參考。
- AUROC 隨 r 遞減：r=2 0.610、r=4 0.616、r=6 0.606、r=8 0.567（想像越長越差，但 r=8 才明顯）。
- 想像 vs 真實（unpaired，母體不同）：AUROC(V) 差 −0.059 [−0.125, 0.004]，RM AUROC 差 −0.071（CI 不含 0），RM MAE 差 +0.006。
- stage-1 pose-aux 角度誤差中位數 12.4°。

## 4. 實驗前置檢查（T2）

- 噪聲 vs 乾淨成功率（1000 成對 episode）：68.5% vs 69.8%，Δ=+1.30 pp [−2.4, +5.0]，McNemar p=0.52 → 採備案（整條加噪聲、不丟 query、stage-2 接續也加噪聲）。注意 CI 未排除 >2 pp 的差異。
- 失敗 episode 的「決定時刻」89.2% 在 t>100 → query 範圍（t∈10–100）只涵蓋 episode 前三分之一，是已知侷限，V 沒看過晚期決策點。
- 0.1σ 噪聲幾乎不改狀態軌跡（agent ≈1.5 px、block ≈0.6 px）。

## 5. 需誠實標註的事項

1. **OracleNet 事後重訓（post-hoc）**：初訓 40 epoch 未收斂（RM MAE 0.094），看過 val 後加 `oracle_max_epochs: 300` 重訓；整個 `train` stage 重跑，ValueNet 以相同 seed 重訓後數字略變（stage-2 AUROC 0.600→0.606、G4 0.013→0.024，G4 規則由未過變過）。初訓輸出留在 `model_t6_oracle40/`。OracleNet 只是參考線，不影響 G6。
2. **T5 在看到 val 前降低評估複雜度**：G1/G2/G3/G5 無門檻，T6 停損撤回；唯一 gate 為 G6。
3. G6 保留 `pre_success` 候選（y=1）為預先註冊例外；tie-break 與分數均依 T1 修訂。
4. stage-2 lead-in 保持乾淨、有效 t 為 `t+10+5r`、奇數 round lookahead 落在 horizon 尾段（`horizon=16` 剛好夠用）：皆為報告 caveat。
5. 檢定力：需 Δ≈0.07 才能讓 100 個 test anchors 的下界 >0；實測 +0.041 低於此，且 SD(Δ)≈0.43 使小效應難以偵測。

## 6. 端到端微調對照（探索性，未預先註冊）

**結論：V 挑選沒有優於隨機；兩個微調模型都沒有比 base 好，V-FT 顯著退步。**

設定：
- pool = T7 的 100 個 test anchors 的想像 branch；去掉 G6 排除的 6 個，剩 94 個 anchor × 8 候選。
- arm V：每個 anchor 取 `V(z_end)` 最高的候選（r=8，β=0.0，即 val 選出的 β，沒有 coverage 項）；arm R：均勻隨機取一個（seed 0）。兩個 arm 的聯集 174 個 branch，V 與 R 挑到同一個的有 14 個 anchor。
- 打包用 `--ep-len 9 --include-anchor`，每個 branch 產生 5 個 pair（4 個想像幀 + 1 個真實 anchor 幀），所以**每個 arm 加 470 個 pair**（94 × 5），加在 206 條真實 episode 上（資料集總共 676 episode、29,880 幀）。真實資料在前，normalization 釘在 real-only 的統計。
- 微調：從 base 出發，`finetune_cl.sh`、seed 1000、20k steps（42.8 epoch）、lr 1e-5、batch 64、`drop_n_last_frames=8`；最終 loss 兩個 arm 都是 0.010。
- 評估：1000 個 episode，seed 100000，`n_action_steps=10`，三個模型起點相同（episode i 用 seed 100000+i）。

| 模型 | 成功數 | 成功率 | 95% CI（Wilson） |
| --- | --- | --- | --- |
| base | 694 | 69.4% | [66.5, 72.2] |
| V-FT | 631 | 63.1% | [60.1, 66.0] |
| R-FT | 662 | 66.2% | [63.2, 69.1] |

| 比較 | 差距 (pp) | 成對 bootstrap 95% CI | 僅 A / 僅 B 成功 | McNemar p |
| --- | --- | --- | --- | --- |
| V − R | −3.1 | [−6.9, +0.7] | 179 / 210 | 0.128 |
| V − base | −6.3 | [−10.1, −2.6] | 153 / 216 | 0.001 |
| R − base | −3.2 | [−6.9, +0.4] | 158 / 190 | 0.096 |

解讀與限制：
- V − R 的 CI 含 0：沒有證據顯示 V 挑的 rollout 比隨機挑的好，與 G6 離線不顯著一致。挑選時的標籤也看得出來：所選候選 r=8 的平均 y，V 0.681、R 0.638，全部候選平均 0.682，即 V 沒有比隨機好。
- 加入想像 pair 後 policy 變差（V-FT 顯著、R-FT 方向相同但不顯著）。原因未驗證，可能是想像幀與真實幀的分布差異，或 branch 動作帶有噪聲；470 個 pair 相對 206 條真實 episode 的比例不小。
- V 挑到 8 個 `pre_success` branch，R 為 5 個；是否與 V-FT 較差有關，沒有驗證。
- 只有單一微調 seed（1000），微調本身的變異沒有量，V − R 的 −3.1 pp 可能在這個變異之內。
- 腳本 `critic/finetune_pool.py`、`attribution/finetune_compare.py` 在 commit `6a6dec01` 標註為未測試草稿；這次跑完，並抽查 `eval_info.json` 的成功數與 only-A/only-B 次數與比較表一致。

### 錄影（每個模型 × 兩個 seed）

`output/sailor_v1/video/<模型>_seed<S>.mp4`，起點 `env.reset(seed=800000/800001)`，`n_action_steps=10`。每個格子只有一個 episode，不能用來比較成功率。

| 模型 | seed 800000 | seed 800001 |
| --- | --- | --- |
| base | 成功，234 幀 | 成功，103 幀 |
| V-FT | 成功，200 幀 | 成功，105 幀 |
| R-FT | 成功，155 幀 | 成功，104 幀 |

## 7. 後續與未完成

- **v2 方向**：[`decisions/sailor_v2_plan.md`](decisions/sailor_v2_plan.md)（重點：降低標籤雜訊，例如每候選多次接續取平均）。
- 微調劑量與資料品質：這次加入想像 pair 反而讓 policy 變差，v2 的端到端對照（§4）需要先決定劑量與是否限制 pair 品質。
- 微調的 seed 變異尚未量測（只有 seed 1000）。

## 8. 產出檔案索引（`~/datastorage/nanowm_pusht_aug/value_model/sailor_v1/`）

`gate_report.{md,json,png}`、`model/curves.png`、`t2_checks/T2_SUMMARY.md`、`stage2/labels_nc{0,1}.npz`（表 D）、`test_stage2/labels.npz`（G6）、`videos/perturb_s0.1_reset730000-730003.{mp4,_sheet.png}`、各階段 log（`t4_*`、`t6*`、`t7.log`）。

微調對照（不在上述目錄）：pool 的挑選結果 `~/datastorage/nanowm_pusht_aug/value_model/finetune/pool/`（`selection.json`、`branch_scores.csv`、`datasets.json`）；checkpoint `~/datastorage/nanowm_pusht_aug/finetune/train/pusht_plus_sailorv1_{v,r}_d8_s1000/`；評估與比較表 `~/datastorage/nanowm_pusht_aug/finetune/eval/{base,v,r}/`、`comparison.{md,json}`；log `finetune/logs/`；影片 `examples/nanowm_pusht_aug/output/sailor_v1/video/`（含 `summary.json`）。
