"""Русско-казахский интерфейс DalaScan AI. Запуск: streamlit run app.py."""
from hashlib import sha256

import streamlit as st

from utils.image_processing import safe_open_image, prepare_image, ImageValidationError
from utils.model_config import MODEL_PATH
from utils.image_guard import WheatImageGuard, GuardUnavailableError, GUARD_PATH, POLICY_VERSION
from utils.analysis import analyze_photo
from utils.predictor import WheatDiseasePredictor, ModelUnavailableError
from utils.i18n import tr, localize_diseases
from utils.recommendations import load_diseases
from utils.ui import apply_style, render_result, render_history, render_model_card, render_pdf_download


@st.cache_resource(show_spinner=False, max_entries=2)
def get_predictor(demo: bool, version: tuple) -> WheatDiseasePredictor:
    """Кэш весов общий для сессий; версия инвалидируется при замене файла."""
    return WheatDiseasePredictor(demo_mode=demo)


@st.cache_resource(show_spinner=False, max_entries=1)
def get_image_guard(version: tuple) -> WheatImageGuard:
    """Одна локальная проверка CLIP для всех сессий, с блокировкой инференса."""
    return WheatImageGuard()


def main() -> None:
    """Отрисовывает приложение и обрабатывает один снимок за анализ."""
    st.set_page_config(page_title="DalaScan AI · RU / KZ", page_icon="🌾", layout="wide")
    language = st.sidebar.selectbox("Язык / Тіл", ["ru", "kk"],
        format_func=lambda value: "Русский" if value == "ru" else "Қазақша", key="language")
    apply_style()
    try:
        diseases = localize_diseases(load_diseases(), language)
    except (ValueError, OSError) as exc:
        st.error(tr("Ошибка справочника. Проверьте data/diseases.json и data/diseases.kk.json.", language))
        st.stop()
    st.session_state.setdefault("history", [])
    with st.sidebar:
        st.markdown("## 🌾 DalaScan AI")
        st.caption(tr("ЗДОРОВЬЕ ПОСЕВОВ · MVP", language))
        st.write(tr("Помогаем начать обследование пшеницы с одного снимка листа.", language))
        st.divider()
        demo = st.toggle("Демонстрационный режим / Демо режимі", value=not MODEL_PATH.is_file(), key="demo_mode")
        st.caption(tr("Режим: демонстрация интерфейса", language) if demo else tr("Режим: локальная нейросеть", language))
        st.markdown(tr("#### Поддерживаемые состояния", language))
        for info in diseases.values():
            st.write("• " + info["name"])
        st.divider()
        st.caption(tr("Снимки обрабатываются на сервере приложения в памяти. Код не сохраняет их на диск и не отправляет во внешние API.", language))

    st.markdown(tr('<div class="eyebrow">АГРОТЕХНОЛОГИИ · ОБСЛЕДОВАНИЕ ПОЛЕЙ</div>', language), unsafe_allow_html=True)
    st.title("DalaScan AI")
    st.markdown(tr("### Диагностика заболеваний пшеницы по фотографии", language))
    st.write(tr("Загрузите снимок листа, получите предварительную оценку и план дальнейшего осмотра.", language))
    a, b, c = st.columns(3)
    a.markdown(tr("**01 · Сфотографируйте**  \nОдин лист, крупно и в фокусе", language))
    b.markdown(tr("**02 · Проанализируйте**  \nЧетыре поддерживаемых состояния", language))
    c.markdown(tr("**03 · Проверьте в поле**  \nСопоставьте результат с осмотром", language))
    if demo:
        st.warning(tr("ДЕМОНСТРАЦИОННЫЙ РЕЖИМ. Результат формируется по хешу изображения. Это не настоящий диагноз и не вывод нейросети.", language), icon="🧪")

    predictor = None
    version = (0, 0)
    try:
        if not demo and MODEL_PATH.exists():
            stat = MODEL_PATH.stat()
            version = (stat.st_mtime_ns, stat.st_size)
        with st.spinner(tr("Подготовка модели…", language)):
            predictor = get_predictor(demo, version)
    except (ModelUnavailableError, OSError) as exc:
        st.error(tr(str(exc), language) if isinstance(exc, ModelUnavailableError) else tr("Нет доступа к весам модели. Включите демонстрационный режим.", language))
    if predictor and not demo:
        st.caption(f"MobileNetV3 Small · {predictor.device.upper()} · {tr('веса загружены', language)}")

        st.info(tr("Исследовательская модель: обучена на открытом датасете. На местных полях качество ещё не проверено.", language))
        render_model_card(predictor, language)

    guard = None
    guard_version = (POLICY_VERSION, 0, 0)
    if predictor and not demo:
        try:
            guard_file = GUARD_PATH / "pytorch_model.bin"
            if guard_file.is_file():
                guard_stat = guard_file.stat()
                guard_version = (POLICY_VERSION, guard_stat.st_mtime_ns, guard_stat.st_size)
            with st.spinner(tr("Подготовка проверки снимков…", language)):
                guard = get_image_guard(guard_version)
        except (GuardUnavailableError, OSError) as exc:
            st.error(tr(str(exc), language) if isinstance(exc, GuardUnavailableError) else tr("Проверка снимков недоступна. Диагностика не выполнялась.", language))
    st.caption(tr("Проверка содержимого экспериментальная: похожие злаки могут быть приняты за пшеницу.", language))

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader(tr("Снимок растения", language))
            source = st.radio("Источник / Дереккөз", ["upload", "camera"], format_func=lambda value: tr("Загрузить файл" if value == "upload" else "Камера", language), horizontal=True, key="image_source")
            if source == "camera":
                upload = st.camera_input("Фото / Сурет", key="camera_photo")
                st.caption(tr("Для камеры нужны разрешение браузера и HTTPS либо localhost.", language))
            else:
                upload = st.file_uploader("Фото / Сурет", type=["jpg", "jpeg", "png", "webp"], key="upload_photo")
            st.caption(tr("JPG, PNG, WEBP · до 15 МБ · до 25 мегапикселей", language))
            raw = upload.getvalue() if upload is not None else None
            image = None
            if raw is not None:
                try:
                    image = prepare_image(safe_open_image(raw))
                    st.image(image, caption=tr("Ваш снимок · подготовлен для анализа", language), use_container_width=True)
                except ImageValidationError as exc:
                    st.error(tr(str(exc), language))
            current_key = (sha256(raw).hexdigest() if raw else None, demo, version, guard_version)
            if st.session_state.get("result_key") != current_key:
                st.session_state.pop("result", None)
            analyze = st.button(tr("Провести диагностику", language), type="primary", use_container_width=True,
                                disabled=image is None or predictor is None or (not demo and guard is None))
    with right:
        with st.container(border=True):
            st.subheader(tr("Хороший снимок — первый шаг", language))
            st.markdown(tr("1. Держите лист **в фокусе**.\n2. Используйте **естественное освещение**.\n3. Покажите больной участок **крупным планом**.\n4. Избегайте **сильных теней**.", language))
            st.caption(tr("Лучше снять отдельный лист, а затем соседние растения. Если содержимое не удалось подтвердить, диагноз не выдаётся.", language))
        st.info(tr("Оценка помогает выбрать следующий шаг обследования. Итоговое решение принимает агроном.", language), icon="🌱")

    if analyze and raw is not None and predictor is not None:
        st.session_state.pop("result", None)
        try:
            with st.spinner(tr("Анализируем снимок…", language)):
                result = analyze_photo(raw, predictor, guard)
                st.session_state["result"] = result
                st.session_state["result_key"] = current_key
                history_entry = {key: value for key, value in result.items() if key != "photo_jpeg"}
                st.session_state["history"] = [history_entry, *st.session_state["history"]][:5]
        except (ImageValidationError, ModelUnavailableError, GuardUnavailableError) as exc:
            st.error(tr(str(exc), language))
        except Exception:
            st.error(tr("Не удалось завершить анализ. Повторите попытку с другим снимком.", language))
    if "result" in st.session_state:
        render_result(st.session_state["result"], diseases, language)
        render_pdf_download(st.session_state["result"], language)
    render_history(diseases, language)
    st.divider()
    st.caption(tr("Результат является предварительной оценкой и не заменяет консультацию агронома. Перед применением средств защиты растений необходимо проверить местные регламенты и инструкцию производителя", language))


if __name__ == "__main__":
    main()
