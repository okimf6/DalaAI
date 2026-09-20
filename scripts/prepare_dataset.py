"""Загрузка Wheat Disease Dataset с Kaggle и воспроизводимое разделение."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageOps
from utils.model_config import CLASSES

SOURCE_URL = "https://www.kaggle.com/datasets/yasserhessein/wheat-disease-dataset-small"
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/yasserhessein/wheat-disease-dataset-small"
PREFIXES = {"Healthy": "healthy", "BrownRust": "leaf_rust", "Septoria": "septoria", "YellowRust": "yellow_rust"}


def sha256_file(path: Path) -> str:
    """Вычисляет контрольную сумму без загрузки архива в память."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_archive(target: Path) -> dict:
    """Скачивает публичный архив Kaggle; сохраняет подтверждение происхождения."""
    receipt = target.with_suffix(".source.json")
    if target.exists() and receipt.exists():
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if record.get("download_url") == DOWNLOAD_URL and record.get("archive_sha256") == sha256_file(target):
            return record
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    try:
        with urllib.request.urlopen(DOWNLOAD_URL, timeout=120) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("Kaggle вернул не ZIP-архив. Проверьте доступ к странице набора.")
        record = {"source": SOURCE_URL, "download_url": DOWNLOAD_URL,
                  "downloaded_at": datetime.now(timezone.utc).isoformat(),
                  "archive_sha256": sha256_file(temporary)}
        temporary.replace(target)
        receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record
    finally:
        temporary.unlink(missing_ok=True)


def extract_images(archive: Path, raw: Path) -> int:
    """Извлекает только PNG четырёх классов, без использования путей из ZIP."""
    if raw.exists() and any(raw.iterdir()):
        raise ValueError("Папка исходных изображений не пуста. Выберите новую --raw.")
    with zipfile.ZipFile(archive) as zipped:
        selected = []
        names = set()
        for entry in zipped.infolist():
            name = Path(entry.filename).name
            if entry.is_dir() or not name.lower().endswith(".png"):
                continue
            if not any(name.startswith(prefix) for prefix in PREFIXES):
                continue
            if name in names:
                raise ValueError("В архиве повторяются имена изображений.")
            if entry.file_size > 50 * 1024 * 1024:
                raise ValueError("Изображение в архиве превышает 50 МБ.")
            names.add(name)
            selected.append((entry, name))
        if not selected or sum(e.file_size for e, _ in selected) > 3 * 1024**3:
            raise ValueError("Неожиданный состав или размер архива Kaggle.")
        raw.mkdir(parents=True, exist_ok=True)
        for entry, name in selected:
            with zipped.open(entry) as source, (raw / name).open("wb") as destination:
                shutil.copyfileobj(source, destination)
    return len(selected)


def fingerprint(path: Path) -> tuple[str, int]:
    """Точный хеш RGB-пикселей и dHash для группировки близких кадров."""
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        pixels = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        bits = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
        perceptual = sum(int(bit) << i for i, bit in enumerate(bits))
        return digest, perceptual


def prepare(raw: Path, output: Path, seed: int = 42, provenance: dict | None = None) -> dict:
    """Группирует dHash-соседей до разбиения; исходные фотографии остаются в raw."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("Выходная папка не пуста. Выберите новую папку для разделения.")
    rows = []
    exact = {}
    duplicates = []
    for path in sorted(raw.glob("*.png")):
        label = next((value for prefix, value in PREFIXES.items() if path.name.startswith(prefix)), None)
        if label is None:
            continue
        digest, perceptual = fingerprint(path)
        if digest in exact:
            if exact[digest]["class"] != label:
                raise ValueError("Одинаковое изображение имеет конфликтующие метки.")
            duplicates.append(path.name)
            continue
        row = {"source_file": path.name, "class": label, "sha256_rgb": digest, "dhash": perceptual}
        rows.append(row)
        exact[digest] = row
    if set(row["class"] for row in rows) != set(CLASSES):
        raise ValueError("В исходных файлах должны быть все четыре класса.")
    parents = list(range(len(rows)))
    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    # Даже межклассовые похожие изображения остаются в одной выборке.
    for i, row in enumerate(rows):
        for j in range(i):
            if (row["dhash"] ^ rows[j]["dhash"]).bit_count() <= 6:
                parents[find(i)] = find(j)
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(find(index), []).append(row)
    rng = random.Random(seed)
    grouped = list(groups.values())
    rng.shuffle(grouped)
    grouped.sort(key=len, reverse=True)
    totals = Counter(row["class"] for row in rows)
    ratios = {"train": .70, "val": .15, "test": .15}
    counts = {split: Counter() for split in ratios}
    # Largest-first assignment to the split with largest normalized class deficit.
    for group_id, group in enumerate(grouped):
        distribution = Counter(row["class"] for row in group)
        split = max(ratios, key=lambda s: sum(
            distribution[c] * (ratios[s] * totals[c] - counts[s][c]) / max(ratios[s] * totals[c], 1)
            for c in distribution))
        for row in group:
            row["split"] = split
            row["group"] = group_id
            counts[split][row["class"]] += 1
    if any(counts[split][label] < 2 for split in ratios for label in CLASSES):
        raise ValueError("Недостаточно независимых групп для всех классов в train/val/test.")
    for row in rows:
        target = output / row["split"] / row["class"] / row["source_file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        # Lossless rescale caps disk and decode costs; split is decided on originals.
        with Image.open(raw / row["source_file"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            image.save(target)
    manifest = {
        "source": (provenance or {}).get("source", "local files; origin not verified"),
        "provenance": provenance or {},
        "authors": ["Megan Long", "James K. M. Brown", "Richard J. Morris", "Matthew Hartley"],
        "seed": seed, "classes": list(CLASSES), "counts": counts,
        "grouping": "Connected components of 64-bit dHash Hamming distance <=6; exact RGB duplicates removed",
        "limitations": "Image-level internal split only. Plant/field IDs unavailable; independence of plants or locations is NOT proven.",
        "excluded_exact_duplicates": duplicates, "images": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """Скачивает только четыре нужных класса и формирует ImageFolder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("dataset_raw"))
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--archive", type=Path, default=Path("dataset_download/wheat-kaggle.zip"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Выходная папка не пуста. Выберите новую --output.")
    if args.raw.exists() and any(args.raw.iterdir()):
        raise ValueError("Папка исходных изображений не пуста. Выберите новую --raw.")
    print("Загрузка архива Kaggle (около 1 ГБ)...", flush=True)
    record = download_archive(args.archive)
    count = extract_images(args.archive, args.raw)
    (args.raw / "source_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Извлечено {count} изображений. Подготовка разбиения...", flush=True)
    manifest = prepare(args.raw, args.output, args.seed, provenance=record)
    print(json.dumps(manifest["counts"], ensure_ascii=False), flush=True)
    print("Готово. Разбиение внутреннее: отдельная полевая проверка всё ещё нужна.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Ошибка подготовки датасета: {exc}", file=sys.stderr)
        sys.exit(1)
