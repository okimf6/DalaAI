"""PDF-отчёт в памяти, с переносом текста и встроенным шрифтом RU/KK."""
from io import BytesIO
from functools import lru_cache
from threading import RLock
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from utils.i18n import tr, localize_diseases
from utils.model_config import ROOT
from utils.recommendations import load_diseases, confidence_level

FONT = "DalaSans"
FONT_LOCK = RLock()
DISCLAIMER = "Результат является предварительной оценкой и не заменяет консультацию агронома. Перед применением средств защиты растений необходимо проверить местные регламенты и инструкцию производителя"


@lru_cache(maxsize=1)
def register_font() -> None:
    """Встроенный DejaVu Sans поддерживает в том числе Ә Ғ Қ Ң Ө Ұ Ү Һ І."""
    with FONT_LOCK:
        pdfmetrics.registerFont(TTFont(FONT, str(ROOT / "assets" / "fonts" / "DejaVuSans.ttf")))


def create_pdf_report(result: dict, language: str = "ru", field_name: str = "", notes: str = "") -> bytes:
    """Создаёт PDF по неизменяемому результату, не читая текущий upload или внешние URL."""
    if not result.get("photo_jpeg"):
        raise ValueError("Для отчёта нет снимка. Выполните анализ ещё раз.")
    register_font()
    field_name, notes = field_name.strip()[:120], notes.strip()[:2000]
    catalog = localize_diseases(load_diseases(), language)
    green, ink, muted = colors.HexColor("#236348"), colors.HexColor("#183e30"), colors.HexColor("#60746a")
    style = ParagraphStyle("body", fontName=FONT, fontSize=9.5, leading=14, textColor=ink, spaceAfter=6)
    small = ParagraphStyle("small", parent=style, fontSize=8, leading=11, textColor=muted)
    heading = ParagraphStyle("heading", parent=style, fontSize=14, leading=19, spaceBefore=12, spaceAfter=8)
    title = ParagraphStyle("title", parent=style, fontSize=25, leading=30, textColor=green, spaceAfter=8)
    def para(text: str, selected=style) -> Paragraph:
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), selected)
    def label(text: str) -> str:
        return tr(text, language)
    demo = bool(result["demo"])
    mode = label("СИМУЛЯЦИЯ · НЕ ДИАГНОЗ") if demo else label("ПРЕДВАРИТЕЛЬНАЯ ОЦЕНКА")
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=(210*mm, 297*mm), rightMargin=18*mm, leftMargin=18*mm,
                                 topMargin=17*mm, bottomMargin=21*mm, title="DalaScan AI", author="DalaScan AI")
    content = [para("DalaScan AI", title), para(label("Отчёт обследования"), heading)]
    banner = Table([[para(mode)]], colWidths=[174*mm])
    banner.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#fff0cf") if demo else colors.HexColor("#eaf4ee")),
                               ("LEFTPADDING", (0,0),(-1,-1),10), ("TOPPADDING", (0,0),(-1,-1),8), ("BOTTOMPADDING", (0,0),(-1,-1),5)]))
    content += [banner, Spacer(1, 9*mm)]
    content += [para(label("Дата анализа") + ": " + result["time"]),
                para(label("Поле / участок") + ": " + (field_name or label("Не указано"))),
                para(label("Время анализа") + f": {result['seconds']:.2f} " + label("сек."))]
    guard = result.get("guard", {})
    if guard:
        content.append(para(label(guard.get("message", ""))))
    content.append(para(label("Проверка содержимого экспериментальная: похожие злаки могут быть приняты за пшеницу."), small))
    photo = Image(BytesIO(result["photo_jpeg"]))
    scale = min(174*mm / photo.imageWidth, 64*mm / photo.imageHeight)
    photo.drawWidth, photo.drawHeight = photo.imageWidth*scale, photo.imageHeight*scale
    photo.hAlign = "LEFT"
    content += [Spacer(1, 3*mm), photo, Spacer(1, 3*mm)]
    content.append(para(label("Снимок анализа") + " · SHA-256: " + result.get("image_sha256", "")[:16], small))
    probabilities = result.get("probabilities", {})
    if not probabilities:
        content += [para(label("Диагностика не выполнялась"), heading),
                    para(label("Снимите отдельный лист пшеницы крупным планом и повторите анализ."))]
    else:
        ranking = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        class_id, confidence = ranking[0]
        info = catalog[class_id]
        level = confidence_level(confidence)
        if level == "high":
            message = label("Вероятное состояние: ") + info["name"]
        elif level == "medium":
            message = label("Предположение: ") + info["name"] + label(". Уверенность недостаточна. Сделайте дополнительную фотографию.")
        else:
            message = label("Состояние надёжно определить не удалось. Повторите съёмку и обратитесь к агроному.")
        content += [para(message, heading), para(label("Риск предполагаемого состояния") + ": " + (info["risk"] if level == "high" and not demo else label("Не определён")))]
        rows = [[para(label("Три наиболее вероятных состояния"), small), para(label("Демо-вероятность") if demo else label("Уверенность модели"), small)]]
        rows += [[para(catalog[key]["name"]), para(f"{value:.1%}")] for key, value in ranking[:3]]
        table = Table(rows, colWidths=[122*mm, 52*mm], hAlign="LEFT")
        table.setStyle(TableStyle([("VALIGN", (0,0),(-1,-1),"TOP"), ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eaf4ee")),
                                  ("LINEBELOW",(0,0),(-1,-1),.3,colors.HexColor("#d8e5dc")),
                                  ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        content += [table, para(label("Вероятности модели не являются доказанной точностью диагноза."), small), PageBreak(),
                    para(label("Справка и рекомендуемые действия"), heading), para(info["name"], heading), para(info["description"]),
                    para(label("Справочные признаки, а не обнаруженные на снимке симптомы."), small)]
        content.append(para(label("Визуальные признаки"), heading))
        content.extend(para("• " + text) for text in info["signs"])
        content.append(para(label("Рекомендуемые действия"), heading))
        content.extend(para(f"{index}. {text}") for index, text in enumerate(info["actions"], 1))
        content.append(para(label("Идентификатор модели") + ": " + result.get("model_id", "unknown")[:20], small))
    if result["quality"].get("warnings"):
        content.append(para(label("Качество снимка"), heading))
        content.extend(para(label(text)) for text in result["quality"]["warnings"])
    if notes:
        content += [para(label("Заметка пользователя"), heading), para(notes)]
    content += [Spacer(1, 7*mm), para(label(DISCLAIMER), small)]
    if demo:
        content.append(para(label("В демонстрационном режиме содержимое снимка не проверяется. Результат не является диагнозом."), small))
    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d8e5dc"))
        canvas.line(18*mm, 16*mm, 192*mm, 16*mm)
        canvas.setFont(FONT, 7)
        canvas.setFillColor(muted)
        canvas.drawString(18*mm, 11*mm, "DalaScan AI · " + mode)
        canvas.drawRightString(192*mm, 11*mm, str(doc.page))
        canvas.restoreState()
    document.build(content, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
