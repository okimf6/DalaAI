"""Загрузка CC BY 4.0 датасета Zenodo и воспроизводимое разделение без точных дублей."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageOps
from utils.model_config import CLASSES

RECORD_URL = "https://zenodo.org/api/records/7307816"
PREFIXES = {"Healthy": "healthy", "BrownRust": "leaf_rust", "Septoria": "septoria", "YellowRust": "yellow_rust"}


def fetch_json(url: str) -> dict:
    """Читает публичную запись без ключей и учётной записи."""
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download_file(item: dict, destination: Path) -> Path:
    """Возобновляет скачивание по файлам, проверяя контрольную сумму издателя."""
    name = item["key"]
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Недопустимое имя файла в источнике.")
    target = destination / name
    expected = item["checksum"]
    if not expected.startswith("md5:"):
        raise ValueError("Неизвестный формат контрольной суммы.")
    def valid(path: Path) -> bool:
        return path.is_file() and hashlib.md5(path.read_bytes()).hexdigest() == expected[4:]
    if valid(target):
        return target
    url = item["links"]["self"]
    if not url.startswith("https://zenodo.org/"):
        raise ValueError("Неожиданный домен файла.")
    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if hashlib.md5(data).hexdigest() != expected[4:]:
                raise ValueError(f"Не совпала контрольная сумма: {name}")
            target.write_bytes(data)
            return target
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Не удалось скачать {name}: {last_error}")


def fingerprint(path: Path) -> tuple[str, int]:
    """Точный хеш RGB-пикселей и dHash для группировки близких кадров."""
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        pixels = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        bits = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
        perceptual = sum(int(bit) << i for i, bit in enumerate(bits))
        return digest, perceptual


def prepare(raw: Path, output: Path, seed: int = 42) -> dict:
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
        "source": "https://doi.org/10.5281/zenodo.7307816", "license": "CC BY 4.0",
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
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    record = fetch_json(RECORD_URL)
    if record["metadata"].get("license", {}).get("id") != "cc-by-4.0":
        raise ValueError("Лицензия источника изменилась. Проверьте условия перед загрузкой.")
    (args.raw / "source_record.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    files = [f for f in record["files"] if any(f["key"].startswith(prefix) for prefix in PREFIXES)]
    print(f"CC BY 4.0: загрузка {len(files)} изображений ({sum(f['size'] for f in files)/1e6:.0f} МБ)", flush=True)
    with ThreadPoolExecutor(max_workers=min(max(args.workers, 1), 6)) as pool:
        futures = [pool.submit(download_file, item, args.raw) for item in files]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 50 == 0 or done == len(files):
                print(f"Скачано {done}/{len(files)}", flush=True)
    manifest = prepare(args.raw, args.output, args.seed)
    print(json.dumps(manifest["counts"], ensure_ascii=False), flush=True)
    print("Готово. Разбиение внутреннее: отдельная полевая проверка всё ещё нужна.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Ошибка подготовки датасета: {exc}", file=sys.stderr)
        sys.exit(1)
