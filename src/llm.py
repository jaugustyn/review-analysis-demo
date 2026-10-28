"""LLM through a local Ollama server: sentiment and topics of each review, and a report for one product.

Answers are forced into a JSON schema, so they can be read without guessing. Quotes are checked
against the review text, because a model can "quote" words that the customer never wrote.
"""

import json
import re
import time
from dataclasses import dataclass
from functools import cache

from src.config import LLM_MODEL, OLLAMA_URL
from src.topics import TOPIC_LABELS

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"enum": ["negative", "neutral", "positive"]},
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"enum": list(TOPIC_LABELS)},
                    "polarity": {"enum": ["positive", "negative"]},
                    "quote": {"type": "string"},
                },
                "required": ["topic", "polarity", "quote"],
            },
        },
    },
    "required": ["sentiment", "topics"],
}

REVIEW_PROMPT = """Analyse this customer review of a product bought online. The review is in Polish.

<review>
{text}
</review>

1. sentiment: the overall sentiment of the review: negative (unhappy), neutral (mixed or lukewarm)
   or positive (happy).
2. topics: what the customer explicitly praises or criticises, only among these topics:
   - quality: build quality and durability: well or poorly made, still works after a long time, broke,
     stopped working, arrived defective
   - description: whether the product matches its description or photos: original or fake, new or used,
     the right size or model, fits or is compatible
   - price: price and value for money
   - delivery: delivery speed, courier, packaging, arrived damaged
   - service: customer service of the shop or the manufacturer: contact, returns, complaints, warranty, invoices
   Not topics: whether the product works well or badly, is easy to use or is useful.
   Many reviews mention no topic at all; then return an empty list.
   For each topic mentioned give the polarity (positive or negative) and a short quote copied word for word
   from the review, in Polish (at most 20 words)."""

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "improvements": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["summary", "strengths", "improvements"],
}

REPORT_PROMPT = """Jesteś analitykiem opinii klientów w sklepie internetowym.
Produkt: {title}
Liczba przeanalizowanych opinii: {n_reviews}

Ile opinii chwali lub krytykuje dany temat, z przykładowymi cytatami z opinii klientów:
{evidence}

Napisz po polsku krótki raport dla menedżera:
- summary: jedno zdanie o ogólnym odbiorze produktu,
- strengths: do 3 rzeczy, które klienci chwalą (pusta lista, jeśli nikt niczego nie chwali),
- improvements: do 3 rzeczy do poprawy (pusta lista, jeśli nikt niczego nie krytykuje).
Każdy punkt to jedno konkretne zdanie. Opieraj się wyłącznie na powyższych danych, nie dodawaj faktów spoza nich.
Porównuj liczby: temat, który więcej opinii krytykuje, niż chwali, nie jest mocną stroną produktu."""


@dataclass
class LLMCall:
    result: dict
    input_tokens: int
    output_tokens: int
    seconds: float


@cache
def _client():
    """One client for all requests, so connections are reused (the client is thread-safe)."""
    import ollama

    return ollama.Client(host=OLLAMA_URL)


def _chat(prompt: str, schema: dict, model: str) -> LLMCall:
    import ollama

    request = {"model": model, "messages": [{"role": "user", "content": prompt}], "format": schema,
               "options": {"temperature": 0}, "think": False}
    try:
        response = _client().chat(**request)
    except ollama.ResponseError:
        # Starting the model on the GPU occasionally fails the first time (a CUDA error); the next try works.
        time.sleep(5)
        response = _client().chat(**request)
    return LLMCall(
        result=json.loads(response.message.content),
        input_tokens=response.prompt_eval_count or 0,
        output_tokens=response.eval_count or 0,
        seconds=(response.total_duration or 0) / 1e9,
    )


def _normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _quote_in_review(quote: str, review: str) -> bool:
    """Each fragment of the quote (fragments may be joined with "...") must appear in the review."""
    fragments = [_normalise(part) for part in re.split(r"\.\.\.|…", quote)]
    fragments = [part for part in fragments if part]
    return bool(fragments) and all(part in review for part in fragments)


def analyse_review(text: str, model: str = LLM_MODEL) -> LLMCall:
    """Sentiment and topics of one review. Each topic gets "quote_ok": whether the quote really is in the review."""
    call = _chat(REVIEW_PROMPT.format(text=text), REVIEW_SCHEMA, model)
    review = _normalise(text)
    seen = set()
    topics = []
    for item in call.result.get("topics", []):
        if item["topic"] in seen:  # one mention per topic, like the keyword method
            continue
        seen.add(item["topic"])
        topics.append({**item, "quote_ok": _quote_in_review(item["quote"], review)})
    call.result["topics"] = topics
    return call


def product_report(title: str, n_reviews: int, evidence: str, model: str = LLM_MODEL) -> LLMCall:
    return _chat(REPORT_PROMPT.format(title=title, n_reviews=n_reviews, evidence=evidence), REPORT_SCHEMA, model)
