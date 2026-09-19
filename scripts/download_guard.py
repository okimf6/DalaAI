"""Однократная загрузка бесплатных весов CLIP; изображения пользователей не передаются."""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_config import ROOT

MODEL_ID = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
FILES = ["config.json", "preprocessor_config.json", "pytorch_model.bin", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "README.md"]


def main() -> None:
    """Сохраняет фиксированную версию модели в model/image_guard (~605 МБ)."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    from huggingface_hub import snapshot_download
    destination = ROOT / "model" / "image_guard"
    print("Загрузка CLIP (~605 МБ). После загрузки проверка работает без интернета.", flush=True)
    snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=FILES,
                      local_dir=destination, token=False, max_workers=2)
    (destination / "revision.txt").write_text(REVISION, encoding="utf-8")
    print(f"Готово: {destination}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Не удалось загрузить проверку снимков: {exc}. Проверьте интернет и повторите команду.", file=sys.stderr)
        sys.exit(1)
