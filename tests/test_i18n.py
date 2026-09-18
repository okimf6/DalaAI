from io import BytesIO
from unittest.mock import patch
from PIL import Image
from streamlit.testing.v1 import AppTest
from utils.i18n import tr, localize_diseases, kazakh_messages
from utils.model_config import ROOT, CLASSES
from utils.recommendations import load_diseases


def test_catalog_languages() -> None:
    original = load_diseases()
    localized = localize_diseases(original, "kk")
    assert set(localized) == set(CLASSES)
    assert localized["leaf_rust"]["name"] == "Қоңыр тат"
    for info in localized.values():
        assert all(info[key] for key in ("name", "risk", "description", "actions", "signs"))
    assert "name" not in original["healthy"]
    assert tr("Провести диагностику", "kk") == "Диагностика жүргізу"
    assert tr("Провести диагностику", "ru") == "Провести диагностику"
    assert all(k and v for k, v in kazakh_messages().items())


def test_switch_language_preserves_result_and_history() -> None:
    image = BytesIO()
    Image.new("RGB", (320, 240), "green").save(image, format="PNG")
    with patch("streamlit.file_uploader", return_value=image):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        app.toggle[0].set_value(True).run(timeout=30)
        next(b for b in app.button if b.label == "Провести диагностику").click().run()
        probabilities = app.session_state["result"]["probabilities"]
        app.selectbox[0].set_value("kk").run()
        assert not app.exception
        assert app.session_state["result"]["probabilities"] == probabilities
        assert len(app.session_state["history"]) == 1
        assert any(b.label == "Диагностика жүргізу" for b in app.button)
        assert any("СИМУЛЯЦИЯ" in w.value and "сурет" in w.value for w in app.warning)
        assert any("Тексеру тарихы" == h.value for h in app.subheader)
        app.selectbox[0].set_value("ru").run()
        assert app.session_state["result"]["probabilities"] == probabilities


def test_corrupt_upload_in_kazakh() -> None:
    with patch("streamlit.file_uploader", return_value=BytesIO(b"broken")):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        app.selectbox[0].set_value("kk").run()
        assert not app.exception
        assert any("бүлінген" in e.value for e in app.error)
