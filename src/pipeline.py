"""All four methods on a set of reviews, and the product summary. Used by analyze.py and the live analysis."""

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import TRANSFORMER_MODEL
from src.llm import analyse_review, product_report
from src.sentiment import lexicon, transformer
from src.topics import TOPIC_LABELS, keyword_topics, topic_summary

LLM_WORKERS = 4  # with OLLAMA_NUM_PARALLEL=4 about 1.6x faster than one at a time
METHODS = ["lexicon", "tfidf", "transformer", "llm"]


def run_methods(
    reviews: pd.DataFrame, tfidf: Pipeline, on_progress: Callable[[int, int], None] | None = None
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Adds a sentiment column per method (sent_<method>), topics from keywords and from the LLM (JSON lists)
    and LLM token counts. Returns the results and the time each step took, in seconds."""
    texts = reviews["text"].tolist()
    out = reviews.copy()
    seconds = {}

    steps = {
        "lexicon": lambda: lexicon(texts),
        "tfidf": lambda: tfidf.predict(texts).tolist(),
        "transformer": lambda: transformer(texts, TRANSFORMER_MODEL),
    }
    for method, run in steps.items():
        start = time.perf_counter()
        out[f"sent_{method}"] = run()
        seconds[method] = time.perf_counter() - start

    start = time.perf_counter()
    out["topics_keywords"] = [json.dumps(topics) for topics in keyword_topics(texts)]
    seconds["keywords"] = time.perf_counter() - start

    start = time.perf_counter()
    calls = []
    with ThreadPoolExecutor(LLM_WORKERS) as pool:
        for done, call in enumerate(pool.map(analyse_review, texts), 1):
            calls.append(call)
            if on_progress:
                on_progress(done, len(texts))
    seconds["llm"] = time.perf_counter() - start
    out["sent_llm"] = [call.result["sentiment"] for call in calls]
    out["topics_llm"] = [json.dumps(call.result["topics"]) for call in calls]
    out["llm_input_tokens"] = [call.input_tokens for call in calls]
    out["llm_output_tokens"] = [call.output_tokens for call in calls]
    return out, seconds


def write_report(title: str, reviews: pd.DataFrame) -> dict:
    """Manager report from the LLM topics: counts per topic with up to 5 quotes each as evidence."""
    lines = []
    for row in topic_summary(reviews, "llm").itertuples():
        if row.reviews:
            verb = "chwali" if row.polarity == "positive" else "krytykuje"
            quotes = "; ".join(f'"{quote}"' for quote in row.quotes[:5])
            lines.append(f"- {TOPIC_LABELS[row.topic]}: {verb} {row.reviews} opinii, np. {quotes}")
    call = product_report(title, len(reviews), "\n".join(lines))
    return {**call.result, "input_tokens": call.input_tokens, "output_tokens": call.output_tokens}
