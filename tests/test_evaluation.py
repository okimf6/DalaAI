import pytest
from utils.evaluation import classification_metrics, check_split_leakage
from utils.model_config import CLASSES


def test_metrics() -> None:
    report = classification_metrics([0, 1, 2, 3], [0, 1, 1, 2])
    assert report["accuracy"] == .5
    assert report["per_class"]["healthy"]["recall"] == 1
    assert report["per_class"]["yellow_rust"]["recall"] == 0
    assert report["confusion_matrix"][2][1] == 1


def test_leakage_rejected(tmp_path) -> None:
    for split in ("train", "test"):
        folder = tmp_path / split / CLASSES[0]
        folder.mkdir(parents=True)
        (folder / "same.png").write_bytes(b"identical image")
    with pytest.raises(ValueError, match="Утечка"):
        check_split_leakage(tmp_path)
