import numpy as np
from PIL import Image
from scripts.prepare_dataset import prepare, PREFIXES
import zipfile
import pytest
from scripts.prepare_dataset import extract_images


def test_extract_flattens_paths_and_ignores_mildew(tmp_path):
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../../Healthy1.png", b"test")
        zipped.writestr("Mildew/Mildew1.png", b"excluded")
    raw = tmp_path / "raw"
    assert extract_images(archive, raw) == 1
    assert (raw / "Healthy1.png").read_bytes() == b"test"
    assert not (tmp_path / "Healthy1.png").exists()
    with pytest.raises(ValueError, match="не пуста"):
        extract_images(archive, raw)


def test_extract_rejects_duplicate_names(tmp_path):
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("one/Healthy1.png", b"one")
        zipped.writestr("two/Healthy1.png", b"two")
    with pytest.raises(ValueError, match="повторяются"):
        extract_images(archive, tmp_path / "raw")


def test_reproducible_grouped_split(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(42)
    for prefix in PREFIXES:
        for index in range(12):
            Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(raw / f"{prefix}{index}.png")
    (raw / "Healthy999.png").write_bytes((raw / "Healthy0.png").read_bytes())
    first = prepare(raw, tmp_path / "first")
    second = prepare(raw, tmp_path / "second")
    record = {"source": "https://www.kaggle.com/example", "archive_sha256": "test"}
    sourced = prepare(raw, tmp_path / "sourced", provenance=record)
    assert sourced["provenance"] == record
    assert sourced["source"] == record["source"]
    assert first == second
    assert first["source"] == "local files; origin not verified"
    assert first["excluded_exact_duplicates"] == ["Healthy999.png"]
    splits = {}
    for row in first["images"]:
        splits.setdefault(row["group"], set()).add(row["split"])
    assert all(len(values) == 1 for values in splits.values())
    assert all(value >= 2 for counts in first["counts"].values() for value in counts.values())
