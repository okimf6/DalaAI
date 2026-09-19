from io import BytesIO
from pathlib import Path
import pytest
from PIL import Image
from pypdf import PdfReader
from utils.pdf_report import create_pdf_report, register_font
from utils.image_guard import MESSAGES
from reportlab.pdfbase import pdfmetrics


def example_result(demo: bool = True, status: str = "analyzed", confidence: float = .82) -> dict:
    photo = BytesIO()
    Image.new("RGB", (400, 300), "green").save(photo, format="JPEG")
    return {"photo_jpeg": photo.getvalue(), "time": "2026-09-19T14:15:00+05:00",
            "demo": demo, "status": status, "seconds": .32, "image_sha256": "a"*64,
            "model_id": "demo" if demo else "b"*64,
            "guard": {"status": "skipped_demo" if demo else "accepted",
                      "message": MESSAGES["skipped_demo" if demo else "accepted"]},
            "probabilities": {"healthy": confidence, "leaf_rust": (1-confidence)/3,
                              "septoria": (1-confidence)/3, "yellow_rust": (1-confidence)/3} if status == "analyzed" else {},
            "quality": {"warnings": []}}


@pytest.mark.parametrize("language", ["ru", "kk"])
def test_pdf_languages(language: str) -> None:
    result = example_result()
    pdf = create_pdf_report(result, language, 'Поле <А> & 2', 'Ә Ғ Қ Ң Ө Ұ Ү Һ І\nЗаметка')
    reader = PdfReader(BytesIO(pdf))
    assert len(reader.pages) == 2
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert 'Поле <А> & 2' in text
    assert 'Ә Ғ Қ Ң Ө Ұ Ү Һ І' in text
    assert ('СИМУЛЯЦИЯ · НЕ ДИАГНОЗ' if language == "ru" else 'СИМУЛЯЦИЯ · ДИАГНОЗ ЕМЕС') in text
    assert sum(len(page.images) for page in reader.pages) >= 1


def test_kazakh_font_glyphs() -> None:
    register_font()
    mapping = pdfmetrics.getFont("DalaSans").face.charToGlyph
    assert all(ord(char) in mapping for char in "ӘәҒғҚқҢңӨөҰұҮүҺһІі")


def test_rejected_has_no_diagnosis() -> None:
    result = example_result(False, "rejected")
    result["guard"] = {"status": "rejected", "message": MESSAGES["rejected"]}
    reader = PdfReader(BytesIO(create_pdf_report(result)))
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Диагностика не выполнялась" in text
    assert "Здоровая пшеница" not in text
    assert "Три наиболее вероятных" not in text


def test_low_confidence_and_long_notes() -> None:
    result = example_result(False, confidence=.3)
    reader = PdfReader(BytesIO(create_pdf_report(result, notes="длинная заметка "*130)))
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Состояние надёжно определить не удалось" in text
    assert "Не определён" in text
    assert len(reader.pages) >= 2
