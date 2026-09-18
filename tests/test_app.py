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
