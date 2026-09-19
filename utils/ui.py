"""Компоненты оформления и результата; логика модели находится отдельно."""
import streamlit as st
from utils.i18n import tr, localize_diseases
from utils.recommendations import confidence_level, get_recommendation


def apply_style() -> None:
    """Светлая тема с адаптивными отступами и контрастными карточками."""
    st.markdown("""<style>
    .stApp {background:#f5f8f2; color:#173d2b;}
    .block-container {max-width:1140px; padding-top:2.5rem; padding-bottom:2rem;}
    h1 {font-size:3.4rem!important; letter-spacing:-.06em; color:#16452e;}
    h2,h3 {letter-spacing:-.025em;}
    .eyebrow {font-size:.72rem; letter-spacing:.16em; font-weight:700; color:#5a7a51;}
    [data-testid="stSidebar"] {background:#eaf0e4;}
    [data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:18px!important;}
    [data-testid="stFileUploader"] {border-radius:14px;}
    [data-testid="stMetric"] {background:#edf4e7; padding:16px; border-radius:14px;}
    .stButton>button {border-radius:12px; min-height:46px;}
    [data-testid="stProgress"] {margin-bottom:12px;}
    @media(max-width:640px) {
      .block-container {padding:1.2rem 1rem;}
      h1 {font-size:2.6rem!important;}
      [data-testid="stMetric"] {padding:10px;}
    }
    </style>""", unsafe_allow_html=True)


def render_result(result: dict, diseases: dict, language: str = "ru") -> None:
    """Показывает вероятности и пороговую интерпретацию без ложной определённости."""
    guard = result.get("guard", {})
    if guard:
        st.info(tr(guard["message"], language))
    if not result.get("probabilities"):
        st.error(tr("Диагностика не выполнялась", language))
        st.caption(tr("Снимите отдельный лист пшеницы крупным планом и повторите анализ.", language))
        st.metric(tr("Время анализа", language), f"{result['seconds']:.2f} с")
        for note in result["quality"]["warnings"]:
            st.warning(tr(note, language))
        return
    ranking = sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True)
    name, confidence = ranking[0]
    info = get_recommendation(name, diseases)
    level = confidence_level(confidence)
    with st.container(border=True):
        st.subheader(tr("Результат анализа", language) if not result["demo"] else tr("Демонстрационный результат", language))
        if result["demo"]:
            st.warning(tr("СИМУЛЯЦИЯ: вероятности, риск и состояние ниже показаны только для проверки интерфейса. Они не характеризуют ваш снимок.", language))
        if level == "high":
            st.success(tr("Вероятное состояние: ", language) + info["name"])
        elif level == "medium":
            st.warning(tr("Предположение: ", language) + info["name"] + tr(". Уверенность недостаточна. Сделайте дополнительную фотографию.", language))
        else:
            st.error(tr("Состояние надёжно определить не удалось. Повторите съёмку и обратитесь к агроному.", language))
            st.caption(tr("Наиболее вероятная, но ненадёжная гипотеза: ", language) + info["name"])
        x, y, z = st.columns(3)
        x.metric(tr("Демо-вероятность", language) if result["demo"] else tr("Уверенность модели", language), f"{confidence:.1%}")
        y.metric(tr("Риск предполагаемого состояния", language), info["risk"] if level == "high" else tr("Не определён", language))
        z.metric(tr("Время анализа", language), f"{result['seconds']:.2f} с")
        st.caption(tr("Время включает проверку файла, подготовку, оценку качества и прогноз; загрузка и прогрев весов выполняются заранее.", language))
        if result["seconds"] >= 5:
            st.warning(tr("Анализ занял больше целевых 5 секунд. Скорость зависит от оборудования и размера снимка.", language))
        for note in result["quality"]["warnings"]:
            st.warning(tr(note, language))
        st.markdown(tr("#### Три наиболее вероятных состояния", language))
        for class_id, probability in ranking[:3]:
            st.progress(float(probability), text=f"{diseases[class_id]['name']} · {probability:.1%}")
        st.caption(tr("Softmax-вероятность не равна доказанной точности диагноза. Неизвестные заболевания и другие растения модель не распознаёт.", language))
        st.markdown(tr("#### Справка по ведущей гипотезе", language))
        st.markdown(f"<span style='color:{info['color']};font-weight:700'>{info['name']}</span>", unsafe_allow_html=True)
        st.write(info["description"])
        st.caption(tr("Справочный риск состояния: ", language) + info["risk"] + tr(". Наличие заболевания на снимке этим не подтверждается.", language))
        signs, actions = st.columns(2)
        with signs:
            st.markdown(tr("**Визуальные признаки**", language))
            for sign in info["signs"]:
                st.write("• " + sign)
        with actions:
            st.markdown(tr("**Что сделать дальше**", language))
            for action in info["actions"]:
                st.write("• " + action)


def render_history(diseases: dict, language: str = "ru") -> None:
    """Пять последних анализов без сохранения пользовательских фотографий."""
    st.subheader(tr("История обследований", language))
    st.caption(tr("Последние пять анализов в текущей сессии. После потери сессии история исчезнет.", language))
    if st.button(tr("Очистить историю", language), disabled=not st.session_state["history"]):
        st.session_state["history"] = []
    if not st.session_state["history"]:
        st.info(tr("Здесь появятся результаты после первого анализа.", language))
    for result in st.session_state["history"]:
        if not result.get("probabilities"):
            st.write(f"{result['time']} · {tr('Диагностика не выполнялась', language)}")
            st.caption(tr(result.get("guard", {}).get("message", ""), language))
            continue
        name = max(result["probabilities"], key=result["probabilities"].get)
        value = result["probabilities"][name]
        mode = tr("ДЕМО · не диагноз", language) if result["demo"] else tr("Модель", language)
        state = diseases[name]["name"] if confidence_level(value) != "low" else tr("Не определено", language)
        st.write(f"{result['time']} · {mode} · {state} · {value:.1%} · {result['seconds']:.2f} с")


def render_model_card(predictor, language: str = "ru") -> None:
    """Показывает метрики именно загруженного checkpoint, если они сохранены."""
    if not predictor.metrics:
        return
    metrics = predictor.metrics
    with st.expander(tr("Карточка модели и результаты проверки", language)):
        a, b = st.columns(2)
        a.metric(tr("Тестовая accuracy", language), f"{metrics['accuracy']:.1%}")
        b.metric(tr("Число тестовых снимков", language), str(metrics["sample_count"]))
        st.caption(tr("Внутренняя тестовая выборка; это не независимая проверка на новых полях.", language))
        st.markdown("[Wheat Disease Dataset — Small · CC BY 4.0](https://doi.org/10.5281/zenodo.7307816)")


def render_pdf_download(result: dict, language: str = "ru") -> None:
    """Создаёт приватный PDF в памяти только для текущего результата."""
    if not result.get("photo_jpeg"):
        return
    with st.expander(tr("Скачать отчёт обследования", language), expanded=True):
        field_name = st.text_input("Поле / Учаске", max_chars=120, key="report_field")
        notes = st.text_area("Заметка / Ескертпе", max_chars=2000, key="report_notes")
        try:
            from utils.pdf_report import create_pdf_report
            pdf = create_pdf_report(result, language, field_name, notes)
            st.download_button(tr("Скачать PDF", language), data=pdf,
                               file_name=f"DalaScan-{result['image_sha256'][:8]}-{language}.pdf",
                               mime="application/pdf", key="download_report")
            st.caption(tr("Отчёт содержит снимок и результат текущего анализа. Проверьте данные перед передачей агроному.", language))
        except Exception:
            st.error(tr("Не удалось создать PDF. Проверьте установку reportlab и наличие шрифта в assets/fonts.", language))
