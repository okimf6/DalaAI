"""Контракт датасета проверяется без скачивания предобученных весов."""
from PIL import Image
import pytest
from scripts.train_model import make_loaders
from utils.model_config import CLASSES


def test_imagefolder_contract(tmp_path) -> None:
    for split in ("train", "val", "test"):
        for class_id in CLASSES:
            folder = tmp_path / split / class_id
            folder.mkdir(parents=True)
            Image.new("RGB", (240, 260), "green").save(folder / "sample.jpg")
    loaders = make_loaders(tmp_path, 2, 42)
    for loader in loaders.values():
        assert tuple(loader.dataset.classes) == CLASSES
        images, targets = next(iter(loader))
        assert images.shape == (2, 3, 224, 224)
    (tmp_path / "train" / "unexpected").mkdir()
    Image.new("RGB", (240, 260)).save(tmp_path / "train" / "unexpected" / "sample.jpg")
    with pytest.raises(ValueError, match="Классы"):
        make_loaders(tmp_path, 2, 42)


@pytest.mark.parametrize("balanced,selection", [(False, "accuracy"), (True, "macro_f1")])
def test_training_checkpoint_and_inference(tmp_path, monkeypatch, balanced, selection) -> None:
    import argparse
    import torch
    import scripts.train_model as training
    from utils.model_config import build_model
    from utils.predictor import WheatDiseasePredictor
    torch.set_num_threads(2)
    for split in ("train", "val", "test"):
        for index, class_id in enumerate(CLASSES):
            folder = tmp_path / "dataset" / split / class_id
            folder.mkdir(parents=True)
            Image.new("RGB", (256, 256), (30 + index * 40, 100 + ("train", "val", "test").index(split) * 20, 20)).save(folder / "sample.jpg")
    # Не скачиваем ImageNet в тестах: проверяем цикл и контракт сохранения.
    monkeypatch.setattr(training, "build_model", lambda pretrained: build_model(False))
    args = argparse.Namespace(data=tmp_path / "dataset", epochs=1, batch_size=2,
                              lr=0.001, output=tmp_path / "trained.pth", patience=1, seed=42, balanced=balanced, selection=selection)
    checkpoint = training.train(args)
    assert checkpoint["classes"] == list(CLASSES)
    assert checkpoint["selection_metric"] == selection
    assert checkpoint["best_selection_score"] == checkpoint["validation_metrics"][selection]
    assert 0 <= checkpoint["test_accuracy"] <= 1
    assert 0 <= checkpoint["best_val_accuracy"] <= 1
    predictor = WheatDiseasePredictor(False, args.output)
    assert sum(predictor.predict(Image.new("RGB", (256, 256))).values()) == pytest.approx(1, abs=1e-6)
