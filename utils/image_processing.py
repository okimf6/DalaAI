"""Безопасная подготовка снимков. Качество оценивается приблизительно."""
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 25_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(ValueError):
    """Понятная пользователю ошибка снимка."""


def safe_open_image(source: bytes | BinaryIO | Path) -> Image.Image:
    """Проверяет размер, формат и целостность файла до декодирования."""
    try:
        if isinstance(source, Path):
            with source.open("rb") as stream:
                data = stream.read(MAX_BYTES + 1)
        elif isinstance(source, bytes):
            data = source
        else:
            source.seek(0)
            data = source.read(MAX_BYTES + 1)
        if not data:
            raise ImageValidationError("Файл пуст. Выберите фотографию.")
        if len(data) > MAX_BYTES:
            raise ImageValidationError("Файл больше 15 МБ. Уменьшите его размер.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as probe:
                if probe.format not in ALLOWED_FORMATS:
                    raise ImageValidationError("Поддерживаются только JPG, PNG и WEBP.")
                if probe.width * probe.height > MAX_PIXELS:
                    raise ImageValidationError("Снимок больше 25 мегапикселей. Уменьшите разрешение.")
                probe.verify()
            with Image.open(BytesIO(data)) as original:
                original.load()
                return ImageOps.exif_transpose(original).copy()
    except ImageValidationError:
        raise
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError,
            Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ImageValidationError("Не удалось прочитать снимок. Файл повреждён или не является изображением.") from exc


def prepare_image(image: Image.Image, max_size: int = 1280) -> Image.Image:
    """Исправляет ориентацию, сводит прозрачность на белый фон и уменьшает RGB."""
    if max_size < 1:
        raise ValueError("Максимальный размер должен быть положительным.")
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba)
    image = image.convert("RGB")
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image


def estimate_brightness(image: Image.Image) -> float:
    """Средняя яркость от 0 до 255."""
    return float(np.asarray(image.convert("L"), dtype=np.float32).mean())


def estimate_sharpness(image: Image.Image) -> float:
    """Дисперсия дискретного лапласиана; это эвристика, не проверка диагноза."""
    small = image.copy()
    small.thumbnail((512, 512))
    pixels = np.asarray(small.convert("L"), dtype=np.float32)
    if min(pixels.shape) < 3:
        return 0.0
    laplacian = (pixels[:-2, 1:-1] + pixels[2:, 1:-1]
                 + pixels[1:-1, :-2] + pixels[1:-1, 2:]
                 - 4 * pixels[1:-1, 1:-1])
    return float(laplacian.var())


def check_quality(image: Image.Image) -> dict:
    """Возвращает метрики и неблокирующие советы по качеству съёмки."""
    brightness = estimate_brightness(image)
    sharpness = estimate_sharpness(image)
    notes = []
    if brightness < 45:
        notes.append("Снимок слишком тёмный. Снимите лист при естественном освещении.")
    elif brightness > 220:
        notes.append("Снимок слишком светлый. Избегайте пересветов и прямой вспышки.")
    if sharpness < 60:
        notes.append("Снимок может быть размытым. Наведите фокус на поражённый участок.")
    if min(image.size) < 224:
        notes.append("Низкое разрешение. Сделайте более крупный снимок листа.")
    return {"brightness": brightness, "sharpness": sharpness, "warnings": notes}
