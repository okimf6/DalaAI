from io import BytesIO
from unittest.mock import Mock
from PIL import Image
import pytest
from utils.image_guard import decide, GuardUnavailableError, WheatImageGuard
from utils.analysis import analyze_photo


def photo() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (320, 240), "green").save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.parametrize("scores,status", [
    ({"wheat": .31, "other_plant": .25, "unrelated": .18}, "accepted"),
    ({"wheat": .15, "other_plant": .16, "unrelated": .32}, "rejected"),
    ({"wheat": .25, "other_plant": .251, "unrelated": .18}, "uncertain"),
    ({"wheat": .10, "other_plant": .05, "unrelated": .05}, "uncertain"),
])
def test_policy(scores: dict, status: str) -> None:
    assert decide(scores)["status"] == status


def test_missing_guard(tmp_path) -> None:
    with pytest.raises(GuardUnavailableError):
        WheatImageGuard(tmp_path)


@pytest.mark.parametrize("status", ["rejected", "uncertain"])
def test_no_disease_prediction_on_rejection(status: str) -> None:
    predictor = Mock(demo_mode=False, model_id="test")
    guard = Mock()
    guard.check.return_value = {"status": status, "message": "test"}
    result = analyze_photo(photo(), predictor, guard)
    predictor.predict.assert_not_called()
    assert result["probabilities"] == {}
    assert result["photo_jpeg"].startswith(b"\xff\xd8")
    assert result["status"] == status


def test_guard_failure_is_not_bypassed() -> None:
    predictor = Mock(demo_mode=False)
    with pytest.raises(GuardUnavailableError):
        analyze_photo(photo(), predictor)
    predictor.predict.assert_not_called()


def test_accepted_calls_predictor_and_demo_is_marked() -> None:
    predictor = Mock(demo_mode=False, model_id="test")
    predictor.predict.return_value = {"healthy": .8, "leaf_rust": .1, "septoria": .05, "yellow_rust": .05}
    guard = Mock()
    guard.check.return_value = {"status": "accepted", "message": "ok"}
    result = analyze_photo(photo(), predictor, guard)
    predictor.predict.assert_called_once()
    assert result["probabilities"]["healthy"] == .8
    predictor.demo_mode = True
    guard.reset_mock()
    result = analyze_photo(photo(), predictor, guard)
    guard.check.assert_not_called()
    assert result["guard"]["status"] == "skipped_demo"
