"""Метрики классификации без дополнительных библиотек."""
import hashlib
from pathlib import Path
import numpy as np
from utils.model_config import CLASSES


def check_split_leakage(data: Path) -> dict:
    """Запрещает точные файловые дубликаты между train, val и test."""
    seen = {}
    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = {}
        for class_id in CLASSES:
            paths = [p for p in (data / split / class_id).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
            counts[split][class_id] = len(paths)
            for path in paths:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                previous = seen.get(digest)
                if previous and (previous[0] != split or previous[1] != class_id):
                    raise ValueError(f"Утечка данных или конфликт меток: {path.name} совпадает с изображением из {previous}.")
                seen[digest] = (split, class_id)
    return counts


def classification_metrics(targets: list[int], predictions: list[int]) -> dict:
    """Accuracy, precision/recall/F1 каждого класса и confusion matrix."""
    if len(targets) != len(predictions) or not targets:
        raise ValueError("Для метрик нужны непустые списки одинаковой длины.")
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for target, predicted in zip(targets, predictions):
        if not 0 <= target < len(CLASSES) or not 0 <= predicted < len(CLASSES):
            raise ValueError("Неверный индекс класса.")
        matrix[target, predicted] += 1
    per_class = {}
    for index, class_id in enumerate(CLASSES):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        precision = tp / max(int(matrix[:, index].sum()), 1)
        recall = tp / max(support, 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_id] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {"accuracy": float(matrix.trace() / matrix.sum()),
            "macro_f1": float(np.mean([row["f1"] for row in per_class.values()])),
            "sample_count": len(targets), "classes": list(CLASSES),
            "confusion_matrix": matrix.tolist(), "per_class": per_class}


def evaluate_model(model, loader, device) -> dict:
    """Вычисляет метрики, не меняя параметры модели."""
    import torch
    targets, predictions = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            targets.extend(labels.tolist())
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
    return classification_metrics(targets, predictions)
