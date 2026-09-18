import numpy as np
from PIL import Image
from scripts.prepare_dataset import prepare, PREFIXES


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
    assert first == second
    assert first["excluded_exact_duplicates"] == ["Healthy999.png"]
    splits = {}
    for row in first["images"]:
        splits.setdefault(row["group"], set()).add(row["split"])
    assert all(len(values) == 1 for values in splits.values())
    assert all(value >= 2 for counts in first["counts"].values() for value in counts.values())
