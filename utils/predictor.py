"""Реальный локальный инференс и явно отделённая демонстрация."""
from hashlib import sha256
from pathlib import Path
from threading import RLock
import numpy as np
from PIL import Image

from utils.model_config import CLASSES, MODEL_PATH, build_model, inference_transform


class ModelUnavailableError(RuntimeError):
    """Модель нельзя безопасно использовать."""


class WheatDiseasePredictor:
    """Предиктор четырёх состояний; demo не загружает PyTorch и не диагностирует."""

    def __init__(self, demo_mode: bool = True, model_path: Path = MODEL_PATH) -> None:
        self.demo_mode = demo_mode
        self.device = "демонстрация"
        self._lock = RLock()
        self.model = None
        if demo_mode:
            return
        if not model_path.is_file():
            raise ModelUnavailableError("Веса модели отсутствуют. Поместите wheat_model.pth в папку model или включите демонстрационный режим.")
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                torch.set_num_threads(min(4, torch.get_num_threads()))
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                if "classes" in checkpoint and tuple(checkpoint["classes"]) != CLASSES:
                    raise ValueError("Порядок классов checkpoint не совпадает с приложением.")
                state = checkpoint["model_state_dict"]
            else:
                state = checkpoint
            self.model = build_model()
            self.model.load_state_dict(state, strict=True)
            if not all(torch.isfinite(t).all() for t in self.model.state_dict().values()):
                raise ValueError("Веса содержат некорректные числа.")
            self.model.to(self.device).eval()
            self.transform = inference_transform()
            # Прогрев вне измерения пользовательского анализа.
            with torch.no_grad():
                self.model(torch.zeros(1, 3, 224, 224, device=self.device))
        except Exception as exc:
            raise ModelUnavailableError("Не удалось загрузить модель: проверьте формат весов, порядок классов и версии PyTorch. Можно включить демонстрационный режим.") from exc

    def predict(self, image: Image.Image) -> dict[str, float]:
        """Возвращает вероятности всех классов; одинаковые пиксели дают одинаковое demo."""
        rgb = image.convert("RGB")
        if self.demo_mode:
            digest = sha256(str(rgb.size).encode() + rgb.tobytes()).digest()
            values = np.frombuffer(digest[:16], dtype=np.uint8).reshape(4, 4).sum(axis=1).astype(float) + 1
            # Разная выраженность позволяет продемонстрировать все пороги UI.
            values **= 1 + digest[16] % 8
            values /= values.sum()
            return dict(zip(CLASSES, map(float, values)))
        try:
            import torch
            with self._lock, torch.no_grad():
                logits = self.model(self.transform(rgb).unsqueeze(0).to(self.device))
                values = torch.softmax(logits, dim=1)[0].cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("Некорректные вероятности.")
            return dict(zip(CLASSES, map(float, values)))
        except Exception as exc:
            raise ModelUnavailableError("Не удалось выполнить анализ. Попробуйте другой снимок или демонстрационный режим.") from exc
