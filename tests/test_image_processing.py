"""Проверки форматов, ориентации, лимитов и неблокирующей оценки качества."""
from io import BytesIO
import pytest
from PIL import Image
from utils.image_processing import safe_open_image, prepare_image, check_quality, ImageValidationError


def encode(image: Image.Image, format: str = "PNG") -> bytes:
    stream = BytesIO()
    image.save(stream, format=format)
    return stream.getvalue()


def test_rgb() -> None:
    image = prepare_image(safe_open_image(encode(Image.new("RGB", (1600, 800), "green"))))
    assert image.mode == "RGB"
    assert image.size == (1280, 640)


def test_rgba() -> None:
    image = prepare_image(safe_open_image(encode(Image.new("RGBA", (100, 100), (0, 0, 0, 0)))))
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP"])
def test_supported_formats(format: str) -> None:
    assert safe_open_image(encode(Image.new("RGB", (40, 30)), format)).size == (40, 30)


@pytest.mark.parametrize("payload", [b"", b"not an image", b"\x89PNG\r\n\x1a\ncorrupt"])
def test_invalid(payload: bytes) -> None:
    with pytest.raises(ImageValidationError):
        safe_open_image(payload)


def test_limits(monkeypatch) -> None:
    import utils.image_processing as module
    data = encode(Image.new("RGB", (40, 30)))
    monkeypatch.setattr(module, "MAX_PIXELS", 100)
    with pytest.raises(ImageValidationError, match="мегапикселей"):
        safe_open_image(data)
    monkeypatch.setattr(module, "MAX_BYTES", 10)
    with pytest.raises(ImageValidationError, match="МБ"):
        safe_open_image(data)


def test_exif_orientation() -> None:
    image = Image.new("RGB", (100, 50), "green")
    exif = Image.Exif()
    exif[274] = 6
    stream = BytesIO()
    image.save(stream, format="JPEG", exif=exif)
    prepared = prepare_image(safe_open_image(stream.getvalue()))
    assert prepared.size == (50, 100)


def test_quality_only_warns() -> None:
    for color in ("black", "white"):
        report = check_quality(Image.new("RGB", (224, 224), color))
        assert len(report["warnings"]) >= 2
        assert report["sharpness"] == 0
