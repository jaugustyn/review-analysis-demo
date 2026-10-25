"""Topics with a keyword dictionary: the cheap alternative to the LLM.

A sentence that contains a topic keyword is one mention of that topic. Its polarity comes from VADER
for that sentence, and the sentence itself is the quote.
"""

import json
import re

import pandas as pd

TOPIC_LABELS = {"quality": "Jakość", "price": "Cena", "delivery": "Dostawa", "service": "Obsługa"}

TOPIC_KEYWORDS = {
    "quality": [
        "quality", "durable", "sturdy", "flimsy", "well made", "well-made", "poorly made", "cheaply made",
        "broke", "broken", "defective", "stopped working", "quit working", "died", "leak", "leaks", "leaked",
        "leaking", "malfunction", "falling apart", "fell apart", "lasted",
    ],
    "price": [
        "price", "priced", "expensive", "inexpensive", "affordable", "cheaper", "worth", "value", "money",
        "overpriced", "cost", "bargain", "deal",
    ],
    "delivery": [
        "shipping", "shipped", "delivery", "delivered", "arrived", "arrive", "package", "packaging", "packaged",
        "on time", "dented", "damaged in shipping", "came damaged",
    ],
    "service": [
        "customer service", "customer support", "seller", "refund", "refunded", "return", "returned",
        "returning", "warranty", "support", "contacted", "responded", "responsive", "sent a replacement",
        "sent me a replacement",
    ],
}
_PATTERNS = {
    topic: re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b", re.IGNORECASE)
    for topic, keywords in TOPIC_KEYWORDS.items()
}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def keyword_topics(texts: list[str]) -> list[list[dict]]:
    """For each review: [{"topic", "polarity", "quote"}], one per topic mentioned (first matching sentence)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    results = []
    for text in texts:
        found = {}
        for sentence in _sentences(text):
            for topic, pattern in _PATTERNS.items():
                if topic in found or not pattern.search(sentence):
                    continue
                score = analyzer.polarity_scores(sentence)["compound"]
                if abs(score) >= 0.05:  # a neutral sentence says nothing good or bad about the topic
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
