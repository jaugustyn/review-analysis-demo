"""Polish labels, number formatting and chart colors shared by the app and the report."""

import re

METHOD_LABELS = {
    "vader": "Słownik (VADER)",
    "tfidf": "Klasyczne ML (TF-IDF)",
    "transformer": "Gotowy model (Hugging Face)",
    "llm": "LLM",
}
SENTIMENT_LABELS = {"negative": "negatywna", "neutral": "neutralna", "positive": "pozytywna"}
SOURCE_LABELS = {"keywords": "Słownik słów kluczowych", "llm": "LLM"}

# Aqua for praise and positive reviews, orange for complaints and negative ones (a pair checked for color
# blindness; the bars carry numbers, as the light aqua is low-contrast). Gray for neutral, teal for the LLM.
POSITIVE = "#1BAF7A"
NEGATIVE = "#EB6834"
NEUTRAL = "#C3C2B7"
TEAL = "#0F766E"
INK = "#13292D"
GRAY = "#9DAAA7"


def fmt_int(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def fmt_pct(share: float, digits: int = 0) -> str:
    return f"{share * 100:.{digits}f}%".replace(".", ",")


def fmt_num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def fmt_pln(value: float) -> str:
    """Small amounts keep grosze: 0,04 zł; larger ones are whole: 1 250 zł."""
    return f"{fmt_num(value, 2 if value < 100 else 0)} zł"


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def md_escape(text: str) -> str:
    """Customer text shown as Markdown: no formatting, and "$" must not start a formula."""
    return re.sub(r"([\\`*_{}\[\]()#+\-!|>~<$])", r"\\\1", text)


def fmt_seconds(seconds: float) -> str:
    if seconds < 1:
        return "poniżej 1 s"
    if seconds < 120:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min"
