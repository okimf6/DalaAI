"""Экспериментальная локальная проверка содержимого снимка через CLIP.

Это zero-shot фильтр, а не обученный и валидированный ботанический классификатор.
Похожий злак может пройти фильтр; неопределённые результаты блокируют диагноз.
"""
from pathlib import Path
from threading import RLock
import math
from PIL import Image
from utils.model_config import ROOT

GUARD_PATH = ROOT / "model" / "image_guard"
POLICY_VERSION = "clip-wheat-v1"
PROMPTS = {
    "wheat": [
        "a close-up photograph of a wheat leaf",
        "a close-up photograph of a diseased wheat leaf with rust and brown spots",
        "a photograph of green wheat leaves growing in a field",
        "a photograph of a yellow or brown wheat leaf",
    ],
    "other_plant": [
        "a photograph of a broad tree leaf",
        "a photograph of a houseplant",
        "a close-up photograph of a corn leaf",
        "a photograph of a flower",
        "a photograph of lawn grass",
        "a photograph of barley or rice leaves",
    ],
    "unrelated": [
        "a photograph of a person or a face",
        "a photograph of a cat or a dog",
        "a photograph of a car or a vehicle",
        "a screenshot of a computer screen or a document with text",
        "a photograph of food on a plate",
        "a photograph of a building or an indoor room",
        "an empty blurry image or a solid color image",
        "a drawing, logo or cartoon illustration",
        "a photograph of a phone or another household object",
    ],
}
MESSAGES = {
    "accepted": "Снимок похож на пшеницу. Автоматическая проверка может ошибаться.",
    "rejected": "На фото, вероятно, не пшеница. Диагностика заболеваний не выполнялась.",
    "uncertain": "Не удалось подтвердить, что на фото пшеница. Снимите отдельный лист крупно и повторите анализ.",
    "skipped_demo": "В демонстрационном режиме содержимое снимка не проверяется. Результат не является диагнозом.",
}


class GuardUnavailableError(RuntimeError):
    """Проверка недоступна: диагностику нельзя продолжать молча."""


def decide(scores: dict[str, float]) -> dict:
    """Осторожные эвристические пороги косинусного сходства, не вероятности."""
    if set(scores) != set(PROMPTS) or not all(math.isfinite(v) for v in scores.values()):
        raise GuardUnavailableError("Проверка снимка вернула некорректный результат.")
    wheat = scores["wheat"]
    competitor = max(scores["other_plant"], scores["unrelated"])
    if competitor >= .22 and competitor - wheat >= .035:
        status = "rejected"
    elif wheat >= .22 and wheat - competitor >= .015:
        status = "accepted"
    else:
        status = "uncertain"
    return {"status": status, "scores": scores, "policy": POLICY_VERSION,
            "method": "CLIP ViT-B/32 zero-shot", "message": MESSAGES[status]}


class WheatImageGuard:
    """Кэшируемая модель; читает только локальные веса и не вызывает API."""
    def __init__(self, model_path: Path = GUARD_PATH) -> None:
        if not (model_path / "pytorch_model.bin").is_file():
            raise GuardUnavailableError("Проверка снимков не установлена. Выполните python scripts/download_guard.py. До установки настоящая диагностика недоступна.")
        self._lock = RLock()
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                torch.set_num_threads(min(4, torch.get_num_threads()))
            self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
            self.model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(self.device).eval()
            prompts = [prompt for group in PROMPTS.values() for prompt in group]
            inputs = self.processor(text=prompts, padding=True, return_tensors="pt")
            with torch.no_grad():
                features = self.model.get_text_features(**{key: value.to(self.device) for key, value in inputs.items()})
                self.text_features = features / features.norm(dim=-1, keepdim=True)
            # Прогрев выполняется один раз, отдельно от времени анализа снимка.
            self.check(Image.new("RGB", (224, 224), "white"))
        except Exception as exc:
            raise GuardUnavailableError("Не удалось загрузить проверку снимков. Проверьте зависимости и повторите python scripts/download_guard.py.") from exc

    def check(self, image: Image.Image) -> dict:
        """Сравнивает изображение с фиксированными группами текстовых описаний."""
        try:
            with self._lock, self.torch.no_grad():
                inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
                features = self.model.get_image_features(pixel_values=inputs["pixel_values"].to(self.device))
                features = features / features.norm(dim=-1, keepdim=True)
                similarities = (features @ self.text_features.T)[0].cpu().tolist()
            offset = 0
            scores = {}
            for group, prompts in PROMPTS.items():
                scores[group] = float(max(similarities[offset:offset + len(prompts)]))
                offset += len(prompts)
            return decide(scores)
        except GuardUnavailableError:
            raise
        except Exception as exc:
            raise GuardUnavailableError("Не удалось проверить содержимое снимка. Диагностика не выполнялась; повторите попытку.") from exc
