"""Topics with a keyword dictionary: the cheap alternative to the LLM.

A sentence that contains a topic keyword is one mention of that topic. Its polarity comes from the Polish
sentiment dictionary for that sentence, and the sentence itself is the quote.
"""

import json
import re

import pandas as pd

TOPIC_LABELS = {
    "quality": "Jakość", "description": "Zgodność z opisem", "price": "Cena", "delivery": "Dostawa",
    "service": "Obsługa klienta",
}

# Beginnings of words, so that one entry covers all forms: "zepsu" matches "zepsuł", "zepsuta", "zepsuło".
# In a phrase every word is such a beginning: "obsług klient" matches "obsługa klienta" and "obsługi klienta".
TOPIC_KEYWORDS = {
    "quality": [
        "jakoś", "wykonani", "solidn", "trwał", "wytrzymał", "porządn", "zepsu", "popsu", "psuj", "awari", "usterk",
        "wadliw", "uszkodz", "pękn", "przestał działa", "rozpad", "tandet", "bubel", "badziew", "przecieka", "rdz",
    ],
    "description": [
        "zgodn z opis", "niezgodn", "opis", "oryginaln", "podrób", "rozmiar", "pasuj", "kompatybiln",
        "jak na zdjęci", "używan",
    ],
    "price": [
        "cen", "tani", "tanio", "drog", "droż", "opłacaln", "pieniądz", "pieniędz", "promocj", "okazj", "koszt",
        "przepłac",
    ],
    "delivery": [
        "dostaw", "dostarcz", "przesyłk", "paczk", "kurier", "wysyłk", "wysłan", "doszł", "doszed", "dotarł",
        "opakowani", "zapakowan", "paczkomat",
    ],
    "service": [
        "obsług klient", "obsług sklep", "obsług zamówieni", "sprzedawc", "sprzedając", "sklep", "reklamac",
        "gwarancj", "zwrot", "zwróci", "serwis", "kontakt", "faktur", "infolini", "konsultant",
    ],
}
_PATTERNS = {
    topic: re.compile(r"\b(?:" + "|".join(k.replace(" ", r"\w*\s+") for k in keywords) + ")", re.IGNORECASE)
    for topic, keywords in TOPIC_KEYWORDS.items()
}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def keyword_topics(texts: list[str]) -> list[list[dict]]:
    """For each review: [{"topic", "polarity", "quote"}], one per topic mentioned (first matching sentence)."""
    from src.sentiment import lexicon_score

    results = []
    for text in texts:
        found = {}
        for sentence in _sentences(text):
            for topic, pattern in _PATTERNS.items():
                if topic in found or not pattern.search(sentence):
                    continue
                score = lexicon_score(sentence)
                if score:  # a neutral sentence says nothing good or bad about the topic
                    found[topic] = {"topic": topic, "polarity": "positive" if score > 0 else "negative",
                                    "quote": sentence}
        results.append(list(found.values()))
    return results


def topic_summary(reviews: pd.DataFrame, source: str) -> pd.DataFrame:
    """Per topic and polarity: how many reviews mention it, with quotes. source: "keywords" or "llm".
    LLM quotes that are not word for word in the review are left out."""
    rows = [item for raw in reviews[f"topics_{source}"] for item in json.loads(raw) if item.get("quote_ok", True)]
    mentions = pd.DataFrame(rows, columns=["topic", "polarity", "quote"])
    summary = []
    for topic in TOPIC_LABELS:
        for polarity in ("positive", "negative"):
            quotes = mentions.loc[(mentions["topic"] == topic) & (mentions["polarity"] == polarity), "quote"]
            summary.append({"topic": topic, "polarity": polarity, "reviews": len(quotes), "quotes": quotes.tolist()})
    return pd.DataFrame(summary)
