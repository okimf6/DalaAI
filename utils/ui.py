"""Компоненты оформления и результата; логика модели находится отдельно."""
import streamlit as st
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


def render_result(result: dict, diseases: dict) -> None:
    """Показывает вероятности и пороговую интерпретацию без ложной определённости."""
    ranking = sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True)
    name, confidence = ranking[0]
    info = get_recommendation(name, diseases)
    level = confidence_level(confidence)
    with st.container(border=True):
        st.subheader("Результат анализа" if not result["demo"] else "Демонстрационный результат")
        if result["demo"]:
            st.warning("СИМУЛЯЦИЯ: вероятности, риск и состояние ниже показаны только для проверки интерфейса. Они не характеризуют ваш снимок.")
        if level == "high":
            st.success("Вероятное состояние: " + info["name_ru"])
        elif level == "medium":
            st.warning("Предположение: " + info["name_ru"] + ". Уверенность недостаточна. Сделайте дополнительную фотографию.")
        else:
            st.error("Состояние надёжно определить не удалось. Повторите съёмку и обратитесь к агроному.")
            st.caption("Наиболее вероятная, но ненадёжная гипотеза: " + info["name_ru"])
        x, y, z = st.columns(3)
        x.metric("Демо-вероятность" if result["demo"] else "Уверенность модели", f"{confidence:.1%}")
        y.metric("Риск предполагаемого состояния", info["risk"] if level == "high" else "Не определён")
        z.metric("Время анализа", f"{result['seconds']:.2f} с")
        st.caption("Время включает проверку файла, подготовку, оценку качества и прогноз; загрузка и прогрев весов выполняются заранее.")
        if result["seconds"] >= 5:
            st.warning("Анализ занял больше целевых 5 секунд. Скорость зависит от оборудования и размера снимка.")
        for note in result["quality"]["warnings"]:
            st.warning(note)
        st.markdown("#### Три наиболее вероятных состояния")
        for class_id, probability in ranking[:3]:
            st.progress(float(probability), text=f"{diseases[class_id]['name_ru']} · {probability:.1%}")
        st.caption("Softmax-вероятность не равна доказанной точности диагноза. Неизвестные заболевания и другие растения модель не распознаёт.")
        st.markdown("#### Справка по ведущей гипотезе")
        st.markdown(f"<span style='color:{info['color']};font-weight:700'>{info['name_ru']}</span>", unsafe_allow_html=True)
        st.write(info["description"])
        st.caption("Справочный риск состояния: " + info["risk"] + ". Наличие заболевания на снимке этим не подтверждается.")
        signs, actions = st.columns(2)
        with signs:
            st.markdown("**Визуальные признаки**")
            for sign in info["signs"]:
                st.write("• " + sign)
        with actions:
            st.markdown("**Что сделать дальше**")
            for action in info["actions"]:
                st.write("• " + action)


def render_history(diseases: dict) -> None:
    """Пять последних анализов без сохранения пользовательских фотографий."""
    st.subheader("История обследований")
    st.caption("Последние пять анализов в текущей сессии. После потери сессии история исчезнет.")
    if st.button("Очистить историю", disabled=not st.session_state["history"]):
        st.session_state["history"] = []
    if not st.session_state["history"]:
        st.info("Здесь появятся результаты после первого анализа.")
    for result in st.session_state["history"]:
        name = max(result["probabilities"], key=result["probabilities"].get)
        value = result["probabilities"][name]
        mode = "ДЕМО · не диагноз" if result["demo"] else "Модель"
        state = diseases[name]["name_ru"] if confidence_level(value) != "low" else "Не определено"
        st.write(f"{result['time']} · {mode} · {state} · {value:.1%} · {result['seconds']:.2f} с")
