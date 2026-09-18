"""Перевод интерфейса без внешних сервисов; русский текст служит ключом."""
from copy import deepcopy
from functools import lru_cache
import json
from utils.model_config import ROOT


@lru_cache(maxsize=1)
def kazakh_messages() -> dict[str, str]:
    """Загружает локальный словарь перевода."""
    return json.loads((ROOT / "data" / "translations.kk.json").read_text(encoding="utf-8"))


def tr(text: str, language: str = "ru") -> str:
    """Возвращает перевод для kk; русский используется для неизвестных ключей."""
    return kazakh_messages().get(text, text) if language == "kk" else text


def localize_diseases(diseases: dict, language: str = "ru") -> dict:
    """Создаёт копию справочника; стабильные ID классов и цвета не меняются."""
    result = deepcopy(diseases)
    if language == "kk":
        localized = json.loads((ROOT / "data" / "diseases.kk.json").read_text(encoding="utf-8"))
        if set(localized) != set(diseases):
            raise ValueError("Классы казахского справочника не совпадают с основным.")
        for class_id, info in localized.items():
            for field in ("name", "risk", "description", "signs", "actions"):
                if field not in info or not info[field]:
                    raise ValueError("Не заполнен перевод справочника.")
            result[class_id].update(info)
    else:
        for info in result.values():
            info["name"] = info["name_ru"]
    return result
