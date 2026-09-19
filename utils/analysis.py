"""Единый порядок обработки: подготовка, проверка содержимого, затем диагноз."""
from datetime import datetime
from hashlib import sha256
from io import BytesIO
from time import perf_counter
from utils.image_processing import safe_open_image, prepare_image, check_quality
from utils.image_guard import MESSAGES, GuardUnavailableError


def analyze_photo(raw: bytes, predictor, guard=None) -> dict:
    """Отклонённый или сомнительный снимок никогда не попадает в классификатор болезней."""
    started = perf_counter()
    image = prepare_image(safe_open_image(raw))
    quality = check_quality(image)
    if predictor.demo_mode:
        assessment = {"status": "skipped_demo", "message": MESSAGES["skipped_demo"], "policy": "demo"}
    else:
        if guard is None:
            raise GuardUnavailableError("Проверка снимков недоступна. Диагностика не выполнялась.")
        assessment = guard.check(image)
    allowed = assessment["status"] in {"accepted", "skipped_demo"}
    probabilities = predictor.predict(image) if allowed else {}
    elapsed = perf_counter() - started
    photo = BytesIO()
    image.save(photo, format="JPEG", quality=88)
    return {"probabilities": probabilities, "seconds": elapsed, "demo": predictor.demo_mode,
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "quality": quality, "guard": assessment, "photo_jpeg": photo.getvalue(),
            "image_sha256": sha256(raw).hexdigest(),
            "model_id": getattr(predictor, "model_id", "demo" if predictor.demo_mode else "unknown"),
            "status": "analyzed" if allowed else assessment["status"]}
