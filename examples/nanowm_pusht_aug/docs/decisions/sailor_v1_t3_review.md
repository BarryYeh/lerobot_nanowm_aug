# SAILOR v1 — T3 phase review（Opus）

- 範圍：`git diff 48a37b1c..f2bcea17`（T3 起點 = T2 結束的 `48a37b1c`）
- 結論：**PASS-WITH-CHANGES**——F1–F3 已在本 phase 內修正，F4–F8 記錄並接受（無一阻擋 T4）。
- 執行方式：T3 的程式大部分來自一個中途被中斷的 Sonnet worker（未 commit、未測、未審）；本輪無法再派工給 subagent，所以 Opus 直接審核、修正、補測試、跑 smoke 並 commit。中斷 worker 的輸出**全部保留**，沒有任何檔案被丟棄。

## Commits

| sha | 內容 |
| --- | --- |
| `8db61061` | `labels.draw_perturbations`（`switch: random\|all`）與 `pick_anchor` + `test_pick_anchor.py` |
| `890a9f3c` | `closed_loop_rollout.py`：`--replan-every-rounds`（預設 1）、`--torch-seed`、`--lead-na`、`--lead-in` / `lead_in.npz`、`chunk_slices`、M7 asserts、branch metadata + `test_replan_cadence.py`、`test_lead_in_roundtrip.py` |
| `197339ff` | `collect_value_states.py`：query 規則與 `query_ts` 由 config、`q_T / q_T_succ / q_cov_end / q_v_target / q_perturb_p / q_noised`、`band.n = 0`、stale latents、provenance + `test_build_queries_fields.py`、`test_make_starts_empty_band.py` |
| `031b747b` | `critic/make_reset_anchors.py`（D2 噪聲、每來源 episode 一個 anchor、`cog_to_body` + 0.5 px round-trip） |
| `13f6bb14` | `label_imagined_branches.py`：`T / success_step / cov_end / v_target`、`cov = sim_cov` + `cov_imagined`、`--tag train\|test` + `anchor_set`、`cont_seed` / `batch` / `cont_perturb` 由 config、job 排序、lead-in 一致性 assert |
| `63f42502` | `configs/sailor_v1_reset.yaml`、`run_value_model.sh` 的 `anchors / test_anchors / test_imagine / test_label` stage + `test_sailor_config.py` |
| `f2bcea17` | 審核修正 F3：label 斷言配對候選的 round-0 chunk 一致 + `test_label_pairing.py` |

## 清單對照

T1 §3 #3–#8 與 §5.4 M1–M10 的 T3 部分全部完成：M1（lead-in 輸出 / 重播、`--torch-seed`、`--lead-na`）、M2 的 label 部分（`anchor_set`、同 anchor 的 lead-in 一致 assert；`_load_stage2` 的 assert 屬 T5）、M3 的 label 部分（`--cont-seed`、`--anchor-subset`、`--out-name`）、M4、M5 的 collect 部分（`q_noised`；loader 屬 T5）、M6、M7、M10；M8 / M9 已於 T2（`25e716f3`）完成。P5(a)（branch 存 `fi / hist / replan_every_rounds` 並 assert）、N18b（stale latents）、N18d（seeds 必須存在）、N18i（collect / anchors / label 的 npz 內嵌 `provenance_json`）也一併完成。`train_value_model.py` 與 `value_gate_report.py` 未改動（T5）。

`closed_loop_rollout.py` 的預設行為：`R = 1`、無 `--torch-seed` / `--lead-na` / `--lead-in` 時，lead-in（`run_lead_in` 以 `policy_act` 每個 seed 在 t=0 `policy.reset()`）、取樣、`look` / `ex` 切片與舊碼逐位元相同（`test_replan_cadence.py` 釘住 `R = 1` 的切片），只多寫 `work/lead_in.npz` 與 branch npz 的 metadata key。`run_value_model.sh` 對沒有 `stage2.replan_every_rounds` 的 topband config 走舊的 stage 清單與舊的 imagine 指令。

## Findings

- **F1（已修，併入首批 commits 前）中斷 worker 的輸出無法通過 lint**：import 排序、ruff format、`N803`（`R` / `H` 參數）、`N802`（測試名的 `T`）、`C408`。處理：`chunk_slices` / `validate_cadence` 參數改名 `every`，lead-in 函式的 `H` 改名 `hist`（呼叫端同步），測試檔沿用 `labels.py` 的檔頭 `ruff: noqa: N802`，`dict()` 改字面值；`test_sailor_config.py` 刪掉一行 `if False` 死碼。
- **F2（已修，併入首批 commits 前）namespace 內的 numpy seed 重用**：(a) `make_reset_anchors` 的 anchor 時刻 rng 與 `draw_perturbations` 同為 `default_rng(740000)`——在 `switch: random` 下會讓 anchor 的 t 與 p 相關（備案下 p 固定，實際無害，但主設計會中）；改為 `seed + 2`。(b) driver 的 `stage2.noise_seed = 740001`（test 750001）等於 anchor 來源 rollout 的動作噪聲 rng（`perturb.seed + 1`）；改為 `+3`。(c) 表 D 的 `noise_ceiling.sample_seed = 770002` 等於表 D 重跑接續的噪聲 rng（`770001 + 1`）；改為 770003。`test_sailor_config.py` 釘住新位移。位移規則寫入 pre-reg §9 T3。
- **F3（已修，`f2bcea17`）配對候選沒有程式保證**：label 只斷言同 anchor 的 lead-in / anchor 狀態一致，沒有斷言「σ=0.1 的候選 j 與 σ=0 的候選 j 是同一支 base chunk」，而 G6 的候選集定義（pre-reg §9 T1、表 C）依賴這件事；GPU 取樣若非決定性，會靜默破壞配對。新增 `assert_paired_candidate`（`chunk0_clean` 容差 1e-3 px）。smoke 實測差異為 0。
- **F4（接受，記錄）舊 topband config 的重新 label 會變**：`cov = sim_cov`、job 依 `(seed_id, sigma, cand, r)` 排序後切 batch，都會改變 topband 重跑 label 的 RM 目標與接續的 batch 組成。`configs/value_model_topband.yaml` 與 `outputs/value_model/topband` 未動、本計劃不重跑 topband，故只記錄（pre-reg §9 T3）。
- **F5（接受，T5 前要處理）`train_value_model.py` 最後的作圖需要 matplotlib**，而 `--extra training` 沒有它：smoke 的 train 在模型與 `val_predictions.npz` 寫完之後才 `ModuleNotFoundError`。smoke 以 `uv run --extra training --with matplotlib` 跑通；T4 不需要 train。`value_gate_report.py` 在 `gate: {}` 上 `KeyError: 'auroc_s1_min'`——這是預期的 fail closed（門檻是 T5）。
- **F6（接受，記錄）`run_value_model.sh` 的預設 stage 清單**包含 `test_*`、`train`、`report`：T4 必須明確列出 `collect encode anchors imagine label`，test anchors 留到 T7（pre-reg §9 T3）。
- **F7（接受，記錄）備案下的 `noised` 欄位**：stage 1 全為 True、stage 2 只描述想像段；D8 的噪聲分層在 stage 1 退化（pre-reg §9 T3）。
- **F8（接受，caveat）stage-2 lead-in 乾淨、anchor 的 wall 只看 0–100 步**：見 pre-reg §9 T3；`config_io` 照舊把 resolved config 與 `config_provenance.jsonl` 寫在 `out_dir` 根目錄（smoke 也一樣，N13 已知行為）。

## 測試與 lint

- `uv run ruff check` / `ruff format --check`：所有 T3 觸及的 `.py` 通過。
- `uv run --extra training pytest examples/nanowm_pusht_aug/tests`：125 passed（T3 新增 `test_build_queries_fields`、`test_label_pairing`、`test_lead_in_roundtrip`、`test_make_starts_empty_band`、`test_pick_anchor`、`test_replan_cadence`、`test_sailor_config`）。plan 指定的三個測試對應：prefix / query 規則 = `test_prefix_perturb.py`（T2）+ `test_build_queries_fields.py` + `test_pick_anchor.py`；`v_target` 閉式含 D3 的 T>B 邊界 = `test_v_target.py`（T2）+ `test_build_queries_fields.py`；replan 節拍 = `test_replan_cadence.py`。

## Smoke（`~/datastorage/nanowm_pusht_aug/value_model/sailor_v1/smoke/`，HEAD `63f42502`，`git_dirty = False`）

- collect：20 episodes（全 reset、σ=0.1 全程、p=300），190 queries（每個 t 19 列），y = 0.511，wall artifact 1/20；`frames_224 (20,21,224,224,3)`、`cov_per_step (20,301)`；`v_target > 0 ⇔ y`，最小正值 0.371（T=198；下界 0.995^200 = 0.367），成功列 `cov_end > 0.95`，13 列 `T > B`（被窗口切掉的成功）。
- encode：`latents_000.npz (20,21,4,32,32)`。
- anchors：4 個 / 6 個來源 episode（0 拒絕），t = {10,10,50,80}，round-trip 誤差 0.000 px；test anchors：2 個 / 3 個來源 episode。
- imagine（k=2、h_max=8、R=2、σ∈{0.0,0.1}）：σ=0.1 run 逐位元重播 σ=0 的 lead-in；branch 的 `horizon=16`、`lead_na=10`；8 個 (anchor, cand) 的 `chunk0_clean` 跨 σ 完全相同，前 10 個執行動作只差噪聲（3–7 px）。
- label：68 列（4 個 r=0 + 16 branch × 4 rounds），y = 0.574，replay asserts 誤差 0，`cov == sim_cov`，r≥2 的 `|cov − cov_imagined|` 平均 0.037；test label 34 列，y = 0.618，`anchor_set = test`。重跑 label 得到相同 y（可重現）。
- train：258 列（stage1 190 / stage2 68），split 與 shuffled / permuted 對照都正常（AUROC ≈ 0.5，樣本太小，無意義）。report：fail closed（F5）。
