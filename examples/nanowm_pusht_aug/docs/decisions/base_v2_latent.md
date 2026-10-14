# Base v2：以 SD-VAE latent 為輸入的 diffusion policy

狀態：**已訓練與評估**（2026-10-07）。v2 成功率比 v1 低 9.6 個百分點，顯著（§5）。

## 1. 目的

v1 base（`ckpt.base` = `pusht_imgonly/obs1_s100000`）吃 96×96 RGB。NanoWM 的世界狀態是 SD-VAE latent（4×32×32），value model 也吃 latent。v2 讓 policy 直接吃同一個 latent 空間：

- 在 NanoWM 裡 rollout 時，policy 可以直接讀生成的 latent，不必先 decode → RGB → resize 96。
- policy、world model、value model 共用同一種觀測。

v2 和 v1 只差在觀測與 encoder，其他設定全部相同，兩者的成功率差可以歸因到輸入表示上。

## 2. 設計

| 項目 | v1 base | v2 base |
| --- | --- | --- |
| demo | `lerobot/pusht` 206 條 / 25650 幀，fps 10 | 同一批 demo（diffusion_policy 原始 zarr），動作逐幀相同 |
| 觀測 | `observation.image` (3,96,96) RGB | `observation.latent` (4,32,32) SD-VAE latent |
| 觀測來源 | gym_pusht 96px 渲染 | gym_pusht **224px** 渲染 → stretch 256 → SD-VAE encode（NanoWM 的路徑） |
| encoder | resnet18（ImageNet 初始化）+ SpatialSoftmax(32) | `latent_conv`（4 層 3×3 conv + GroupNorm，從頭訓練，1.55M 參數）+ SpatialSoftmax(32) |
| 特徵圖 | 84×84 crop → 512×3×3 | 28×28 crop → 512×14×14 |
| 增強 | random crop 84/96 + image_transforms | random crop 28/32（比例相同）；image_transforms 只作用於影像 key，因此不生效 |
| 正規化 | VISUAL MEAN_STD，逐 RGB 通道 | VISUAL MEAN_STD，逐 VAE 通道（stats.json 改寫為 (4,1,1)） |
| 超參數 | n_obs 1、horizon 16、n_action 8、drop 8、batch 64、100k steps、lr 1e-4、seed 100000 | 相同 |
| 評估 | `proto_eval_pusht.sh`（lerobot-eval） | `eval/eval_latent_policy.py`（lerobot-eval + 224 渲染 + 迴圈內 encode），seed 與 episode 順序相同，可成對比較 |

## 3. 為什麼要重新渲染 224，而不是放大 96

NanoWM 是用 224px 的 PushT 渲染訓練的。把 96 圖放大到 256 再 encode，得到的 latent 會比較模糊，統計分佈也和 NanoWM 產生的 latent 不同。`lerobot/pusht` 沒有 block pose，無法重新渲染，所以改從 diffusion_policy 原始的 `pusht_cchi_v7_replay.zarr` 取得完整的 5 維 state（sha256：zip `63d52a11…`）。

## 4. 驗證

| 檢查 | 結果 |
| --- | --- |
| state 慣例：每幀用 state 重渲染 96 圖，和 zarr 原始影格比對 | angle-first（`place_body`）MAE 平均 0.076、最大 0.345（/255，全部 25650 幀）；position-first 約 2.5 → zarr 的 state 是 body-origin，必須先設 angle |
| 動作與 `lerobot/pusht` 比對 | 完全相同（max\|d\| = 0） |
| SD-VAE posterior 雜訊 | scaled std 平均 1.8e-5、最大 2.2e-4；latent std 約 0.4 → `sample()` ≈ `mode()`，一律用 `mode()` |
| 和 NanoWM venv 的 encode 做 parity（64 幀） | TF32 關閉時 max 3e-4 / mean 2e-6；cuDNN TF32 開啟（PyTorch 預設）時 max 4e-2 → `encode_224` 在內部關閉 TF32 |
| eval 路徑 vs dataset 路徑 | 逐位元一致（`test_eval_encode_step_matches_dataset_encode`，atol 1e-5）；注意 uint8 輸入必須是連續的 NHWC，否則會走不同 kernel，誤差到 3e-4 |
| latent 逐通道 mean / std（全部資料） | mean [2.011, 1.279, 0.121, -0.894]，std [0.466, 0.422, 0.358, 0.471] |

## 5. 結果

1000 episodes，seed 100000（episode i 用 seed 100000+i），`n_action_steps=10`，100k 步的 checkpoint。v1 的數字取自 `~/datastorage/nanowm_pusht_aug/finetune/eval/base/`，v2 的取自 `outputs/eval/pusht_latent/obs1_s100000_na10/`。

| | 成功率 | avg max reward | 失敗數 | 失敗中 max reward ≥ 0.9 | 失敗中 max reward < 0.5 |
| --- | --- | --- | --- | --- | --- |
| v1 RGB | 69.4% | 0.947 | 306 | 78.1% | 16.3% |
| v2 latent | **59.8%** | **0.981** | 402 | 94.3% | 2.2% |

成對比較：v2 − v1 = **−9.6 pp**，成對 bootstrap 95% CI [−13.9, −5.4]，McNemar p = 1.1e-5。兩者都成功 411，只有 v2 成功 187，只有 v1 成功 283，兩者都失敗 119。

解讀：v2 的失敗幾乎全是「差一點」（覆蓋率已經接近門檻，卻在 300 步內沒有達到 0.95）。嚴重失敗（max reward < 0.5）從 16.3% 降到 2.2%。也就是說，latent 輸入讓整體的推動策略更穩，但最後的精細對位變差。可能的原因（尚未驗證）：

1. latent 解析度：一個 latent cell 約等於 16 sim px（512/32），對位需要的細節可能被 VAE 壓掉。
2. random crop 在 latent 上是以整格平移（每格 16 sim px）；RGB 的 84/96 crop 每格只有 5.3 sim px。在 latent 上做位移增強，可能讓位置估計變粗。
3. 只有單一 seed 和單一 checkpoint（100k）；中間的 checkpoint 或 `n_action_steps=8` 也許比較好。

## 6. 已知限制與後續

- NanoWM 生成的 latent 是 diffusion sample，分佈會比真實 encode 稍微偏移。v2 只用真實 encode 訓練；policy 在想像 latent 上的表現，要另外拿 NanoWM rollout 來測。
- 資料轉換和評估都需要 SD-VAE 權重（`~/models/sd-vae-ft-mse`，或用 `VAE_MODEL_PATH` 指定），而且評估每一步都要多跑一次 VAE encode。

## 檔案

- `datagen/convert_pusht_latent.py`（`run_converter.py pusht-latent`）→ `local/pusht_latent`
- `nwaug/sdvae.py`：VAE 載入、前處理、encode（lerobot / nanowm 兩個 venv 都可以 import）
- `src/lerobot/policies/diffusion/`：`vision_backbone="latent_conv"`
- `scripts/train_latent_base.sh` → `outputs/train/pusht_latent/obs1_s100000`（`ckpt.base_v2_latent`）
- `eval/eval_latent_policy.py`
- `tests/test_latent_policy.py`
