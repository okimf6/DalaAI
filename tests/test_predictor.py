from PIL import Image
import pytest
from utils.model_config import CLASSES, build_model
from utils.predictor import WheatDiseasePredictor, ModelUnavailableError


def test_demo_probabilities() -> None:
    image = Image.new("RGB", (320, 240), "green")
    first = WheatDiseasePredictor().predict(image)
    second = WheatDiseasePredictor().predict(image.copy())
    assert first == second
    assert set(first) == set(CLASSES)
    assert sum(first.values()) == pytest.approx(1)
    assert all(0 <= p <= 1 for p in first.values())


def test_missing_weights(tmp_path) -> None:
    with pytest.raises(ModelUnavailableError, match="отсутствуют"):
        WheatDiseasePredictor(False, tmp_path / "missing.pth")


def test_corrupt_weights(tmp_path) -> None:
    path = tmp_path / "corrupt.pth"
    path.write_bytes(b"invalid checkpoint")
    with pytest.raises(ModelUnavailableError, match="загрузить"):
        WheatDiseasePredictor(False, path)


@pytest.mark.parametrize("wrapped", [False, True])
def test_real_formats(tmp_path, wrapped: bool) -> None:
    # Случайные веса существуют только во временной папке pytest, не в поставке.
    import torch
    torch.set_num_threads(2)
    path = tmp_path / "weights.pth"
    state = build_model().state_dict()
    torch.save({"model_state_dict": state, "classes": list(CLASSES)} if wrapped else state, path)
    probabilities = WheatDiseasePredictor(False, path).predict(Image.new("RGB", (256, 256), "green"))
    assert sum(probabilities.values()) == pytest.approx(1, abs=1e-6)
    assert list(probabilities) == list(CLASSES)


def test_wrong_class_order(tmp_path) -> None:
    import torch
    path = tmp_path / "wrong.pth"
    torch.save({"model_state_dict": {}, "classes": list(reversed(CLASSES))}, path)
    with pytest.raises(ModelUnavailableError):
        WheatDiseasePredictor(False, path)
