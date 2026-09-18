"""Загрузка и проверка справочника состояний."""
import json
import re
from copy import deepcopy
from pathlib import Path
from utils.model_config import CLASSES, ROOT

REQUIRED = {"name_ru", "name_en", "risk", "description", "signs", "actions", "color"}


def load_diseases(path: Path = ROOT / "data" / "diseases.json") -> dict:
    """Загружает ровно четыре класса и проверяет поля справочника."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != set(CLASSES):
            raise ValueError("В справочнике должны быть четыре поддерживаемых класса.")
        for info in data.values():
            if not isinstance(info, dict) or not REQUIRED <= info.keys():
                raise ValueError("В справочнике отсутствуют обязательные поля.")
            for field in REQUIRED - {"signs", "actions"}:
                if not isinstance(info[field], str) or not info[field].strip():
                    raise ValueError("Текстовые поля справочника должны быть заполнены.")
            for field in ("signs", "actions"):
                if not isinstance(info[field], list) or not info[field] or not all(isinstance(x, str) and x.strip() for x in info[field]):
                    raise ValueError("Признаки и действия должны быть непустыми списками строк.")
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", info["color"]):
                raise ValueError("Некорректный цвет состояния.")
        return data
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Не удалось загрузить справочник заболеваний: {exc}") from exc


def get_recommendation(class_name: str, diseases: dict | None = None) -> dict:
    """Неизвестный класс возвращает безопасное нейтральное описание."""
    data = diseases if diseases is not None else load_diseases()
    return deepcopy(data.get(class_name, {
        "name_ru": "Неизвестное состояние", "name_en": "Unknown", "risk": "Не определён",
        "description": "Для этого состояния нет достоверной оценки.",
        "signs": ["Недостаточно данных для определения признаков."],
        "actions": ["Повторите съёмку и обратитесь к агроному."], "color": "#64748b",
    }))


def confidence_level(confidence: float) -> str:
    """Пороговая политика по исходной вероятности, до округления процентов."""
    if confidence >= 0.70:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"
