"""Проверки Streamlit без браузера и внешних сервисов."""
from streamlit.testing.v1 import AppTest
from utils.model_config import ROOT


def test_app_starts_and_missing_model() -> None:
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "DalaScan AI"
    assert app.button[0].disabled
    app.toggle[0].set_value(False).run(timeout=30)
    assert not app.exception
    if not (ROOT / "model" / "wheat_model.pth").exists():
        assert any("отсутствуют" in item.value for item in app.error)


def test_history_clear() -> None:
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    app.session_state["history"] = [{"probabilities": {"healthy": 0.8, "leaf_rust": 0.1, "septoria": 0.05, "yellow_rust": 0.05},
                                      "seconds": 0.1, "demo": True, "time": "12:00"}]
    app.run()
    next(button for button in app.button if button.label == "Очистить историю").click().run()
    assert app.session_state["history"] == []
    assert not app.exception


def test_upload_diagnosis_history_and_bad_file() -> None:
    from io import BytesIO
    from unittest.mock import patch
    from PIL import Image
    image = BytesIO()
    Image.new("RGB", (640, 480), "green").save(image, format="PNG")
    with patch("streamlit.file_uploader", return_value=image):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        app.toggle[0].set_value(True).run(timeout=30)
        for _ in range(6):
            next(b for b in app.button if b.label == "Провести диагностику").click().run()
        assert not app.exception
        assert len(app.session_state["history"]) == 5
        assert app.session_state["result"]["demo"]
        assert app.session_state["result"]["seconds"] < 5
        assert any("СИМУЛЯЦИЯ" in message.value for message in app.warning)
    with patch("streamlit.file_uploader", return_value=BytesIO(b"broken")):
        app.run()
        assert not app.exception
        assert "result" not in app.session_state
        assert any("повреждён" in message.value for message in app.error)
        assert next(b for b in app.button if b.label == "Провести диагностику").disabled


def test_rejected_upload_pdf_and_no_stale_result() -> None:
    from io import BytesIO
    from unittest.mock import patch
    from PIL import Image
    from pypdf import PdfReader
    from utils.image_guard import MESSAGES, GuardUnavailableError
    from utils.model_config import MODEL_PATH
    import pytest
    if not MODEL_PATH.is_file():
        pytest.skip("Integration requires local disease weights")
    from utils.image_guard import GUARD_PATH
    if not (GUARD_PATH / "pytorch_model.bin").is_file():
        pytest.skip("Integration requires local CLIP weights")
    stream = BytesIO()
    Image.new("RGB", (400, 300), "white").save(stream, format="PNG")
    with patch("streamlit.file_uploader", return_value=stream), \
         patch("utils.image_guard.WheatImageGuard.check", return_value={"status": "rejected", "message": MESSAGES["rejected"]}), \
         patch("utils.predictor.WheatDiseasePredictor.predict") as predict, \
         patch("streamlit.download_button") as download:
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        app.toggle[0].set_value(False).run(timeout=30)
        next(b for b in app.button if b.label == "Провести диагностику").click().run()
        assert not app.exception
        predict.assert_not_called()
        assert app.session_state["result"]["probabilities"] == {}
        assert any(e.value == "Диагностика не выполнялась" for e in app.error)
        assert download.called
        pdf = download.call_args.kwargs["data"]
        text = "\n".join(page.extract_text() for page in PdfReader(BytesIO(pdf)).pages)
        assert "Диагностика не выполнялась" in text
        assert "Три наиболее вероятных" not in text
        with patch("utils.image_guard.WheatImageGuard.check", side_effect=GuardUnavailableError("Сбой проверки")):
            download.reset_mock()
            next(b for b in app.button if b.label == "Провести диагностику").click().run()
            assert "result" not in app.session_state
            download.assert_not_called()
