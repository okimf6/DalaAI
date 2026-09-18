"""Русскоязычный интерфейс DalaScan AI. Запуск: streamlit run app.py."""
from datetime import datetime
from hashlib import sha256
from time import perf_counter

import streamlit as st

from utils.image_processing import safe_open_image, prepare_image, check_quality, ImageValidationError
from utils.model_config import MODEL_PATH
from utils.predictor import WheatDiseasePredictor, ModelUnavailableError
from utils.recommendations import load_diseases
from utils.ui import apply_style, render_result, render_history


@st.cache_resource(show_spinner=False, max_entries=2)
def get_predictor(demo: bool, version: tuple) -> WheatDiseasePredictor:
    """Кэш весов общий для сессий; версия инвалидируется при замене файла."""
    return WheatDiseasePredictor(demo_mode=demo)


def main() -> None:
    """Отрисовывает приложение и обрабатывает один снимок за анализ."""
    st.set_page_config(page_title="DalaScan AI · Здоровье посевов", page_icon="🌾", layout="wide")
    apply_style()
    try:
        diseases = load_diseases()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.session_state.setdefault("history", [])
    with st.sidebar:
        st.markdown("## 🌾 DalaScan AI")
        st.caption("ЗДОРОВЬЕ ПОСЕВОВ · MVP")
        st.write("Помогаем начать обследование пшеницы с одного снимка листа.")
        st.divider()
        demo = st.toggle("Демонстрационный режим", value=True, key="demo_mode")
        st.caption("Режим: демонстрация интерфейса" if demo else "Режим: локальная нейросеть")
        st.markdown("#### Поддерживаемые состояния")
        for info in diseases.values():
            st.write("• " + info["name_ru"])
        st.divider()
        st.caption("Снимки обрабатываются на сервере приложения в памяти. Код не сохраняет их на диск и не отправляет во внешние API.")

    st.markdown('<div class="eyebrow">АГРОТЕХНОЛОГИИ · ОБСЛЕДОВАНИЕ ПОЛЕЙ</div>', unsafe_allow_html=True)
    st.title("DalaScan AI")
    st.markdown("### Диагностика заболеваний пшеницы по фотографии")
    st.write("Загрузите снимок листа, получите предварительную оценку и план дальнейшего осмотра.")
    a, b, c = st.columns(3)
    a.markdown("**01 · Сфотографируйте**  \nОдин лист, крупно и в фокусе")
    b.markdown("**02 · Проанализируйте**  \nЧетыре поддерживаемых состояния")
    c.markdown("**03 · Проверьте в поле**  \nСопоставьте результат с осмотром")
    if demo:
        st.warning("ДЕМОНСТРАЦИОННЫЙ РЕЖИМ. Результат формируется по хешу изображения. Это не настоящий диагноз и не вывод нейросети.", icon="🧪")

    predictor = None
    version = (0, 0)
    try:
        if not demo and MODEL_PATH.exists():
            stat = MODEL_PATH.stat()
            version = (stat.st_mtime_ns, stat.st_size)
        with st.spinner("Подготовка модели…"):
            predictor = get_predictor(demo, version)
    except (ModelUnavailableError, OSError) as exc:
        st.error(str(exc) if isinstance(exc, ModelUnavailableError) else "Нет доступа к весам модели. Включите демонстрационный режим.")
    if predictor and not demo:
        st.caption(f"MobileNetV3 Small · {predictor.device.upper()} · веса загружены")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Снимок растения")
            source = st.radio("Источник снимка", ["Загрузить файл", "Камера"], horizontal=True)
            if source == "Камера":
                upload = st.camera_input("Сфотографируйте лист пшеницы")
                st.caption("Для камеры нужны разрешение браузера и HTTPS либо localhost.")
            else:
                upload = st.file_uploader("Выберите или перетащите фотографию", type=["jpg", "jpeg", "png", "webp"])
            st.caption("JPG, PNG, WEBP · до 15 МБ · до 25 мегапикселей")
            raw = upload.getvalue() if upload is not None else None
            image = None
            if raw is not None:
                try:
                    image = prepare_image(safe_open_image(raw))
                    st.image(image, caption="Ваш снимок · подготовлен для анализа", use_container_width=True)
                except ImageValidationError as exc:
                    st.error(str(exc))
            current_key = (sha256(raw).hexdigest() if raw else None, demo, version)
            if st.session_state.get("result_key") != current_key:
                st.session_state.pop("result", None)
            analyze = st.button("Провести диагностику", type="primary", use_container_width=True,
                                disabled=image is None or predictor is None)
    with right:
        with st.container(border=True):
            st.subheader("Хороший снимок — первый шаг")
            st.markdown("1. Держите лист **в фокусе**.\n2. Используйте **естественное освещение**.\n3. Покажите больной участок **крупным планом**.\n4. Избегайте **сильных теней**.")
            st.caption("Лучше снять отдельный лист, а затем соседние растения. Модель не проверяет, действительно ли в кадре пшеница.")
        st.info("Оценка помогает выбрать следующий шаг обследования. Итоговое решение принимает агроном.", icon="🌱")

    if analyze and raw is not None and predictor is not None:
        try:
            with st.spinner("Анализируем снимок…"):
                started = perf_counter()
                image = prepare_image(safe_open_image(raw))
                quality = check_quality(image)
                probabilities = predictor.predict(image)
                elapsed = perf_counter() - started
                result = {"probabilities": probabilities, "seconds": elapsed, "demo": demo,
                          "time": datetime.now().strftime("%d.%m %H:%M:%S"), "quality": quality}
                st.session_state["result"] = result
                st.session_state["result_key"] = current_key
                st.session_state["history"] = [result, *st.session_state["history"]][:5]
        except (ImageValidationError, ModelUnavailableError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("Не удалось завершить анализ. Повторите попытку с другим снимком.")
    if "result" in st.session_state:
        render_result(st.session_state["result"], diseases)
    render_history(diseases)
    st.divider()
    st.caption("Результат является предварительной оценкой и не заменяет консультацию агронома. Перед применением средств защиты растений необходимо проверить местные регламенты и инструкцию производителя")


if __name__ == "__main__":
    main()
