import json
import pytest
from utils.model_config import CLASSES
from utils.recommendations import load_diseases, get_recommendation, confidence_level


def test_catalog() -> None:
    catalog = load_diseases()
    assert set(catalog) == set(CLASSES)
    assert len(catalog) == 4
    assert catalog["healthy"]["name_ru"] == "Здоровая пшеница"


def test_unknown() -> None:
    assert get_recommendation("missing")["risk"] == "Не определён"


def test_invalid_catalog(tmp_path) -> None:
    data = load_diseases()
    del data["healthy"]["actions"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="обязательные"):
        load_diseases(path)


@pytest.mark.parametrize("value,expected", [(0.44999, "low"), (0.45, "medium"), (0.69999, "medium"), (0.70, "high")])
def test_thresholds(value: float, expected: str) -> None:
    assert confidence_level(value) == expected
