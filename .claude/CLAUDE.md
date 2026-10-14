# 專案偏好

- 回覆一律使用繁體中文。
- 預設的 Hugging Face repo：`barry0916/nanowm-aug-assets`（model repo，私人）。使用者說「上傳到 HF」而沒指定 repo 時，就上傳到這裡；路徑也存在環境變數 `HF_DEFAULT_REPO`。
- 影片放在該 repo 的 `video/` 資料夾。上傳用 `uv run hf upload barry0916/nanowm-aug-assets <本地資料夾> <repo 內路徑> --repo-type model`（`uv` 在 `~/.local/bin`，PATH 沒有時用完整路徑）。
