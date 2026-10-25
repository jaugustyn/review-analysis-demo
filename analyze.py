"""Runs the four methods once: on the evaluation set (to compare them) and on the demo products (for the app).

Needs an NVIDIA GPU for the transformer model and a running Ollama server with LLM_MODEL from src/config.py.
Run after prepare_data.py: uv run python analyze.py
"""

import json
import time
from datetime import date

import joblib
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.config import (
    DEMO_INPUT, DEMO_PRODUCT_IDS, DEMO_RESULTS, EVAL_INPUT, EVAL_RESULTS, LLM_MODEL, METRICS_PATH, MODELS_DIR,
    PRECOMPUTED_DIR, PRODUCTS_RESULTS, TFIDF_PATH, TRAIN_INPUT, TRANSFORMER_MODEL,
)
from src.data import SENTIMENTS, product_titles, review_text
from src.pipeline import LLM_WORKERS, METHODS, run_methods, write_report
from src.sentiment import train_tfidf

COLUMNS = [
    "parent_asin", "rating", "sentiment", "title", "text", "timestamp",
    *(f"sent_{method}" for method in METHODS), "topics_keywords", "topics_llm", "llm_input_tokens", "llm_output_tokens",
]


def show_progress(done: int, total: int) -> None:
    if done % 100 == 0 or done == total:
        print(f"  LLM: {done}/{total}", flush=True)


def compare(evaluated: pd.DataFrame, seconds: dict[str, float]) -> dict:
    n = len(evaluated)
    truth = evaluated["sentiment"]
    quotes = [item["quote_ok"] for raw in evaluated["topics_llm"] for item in json.loads(raw)]
    return {
        "eval_reviews": n,
        "class_shares": truth.value_counts(normalize=True).round(4).to_dict(),
        "methods": {
            method: {
                "accuracy": accuracy_score(truth, evaluated[f"sent_{method}"]),
                "f1_macro": f1_score(truth, evaluated[f"sent_{method}"], average="macro"),
                "f1": dict(zip(SENTIMENTS, f1_score(truth, evaluated[f"sent_{method}"], labels=SENTIMENTS, average=None))),
                "seconds_per_1000": seconds[method] / n * 1000,
            }
            for method in METHODS
        },
        "keywords_seconds_per_1000": seconds["keywords"] / n * 1000,
        "llm_tokens_per_1000": {
            "input": int(evaluated["llm_input_tokens"].sum() / n * 1000),
            "output": int(evaluated["llm_output_tokens"].sum() / n * 1000),
        },
        "llm_quotes_verbatim": sum(quotes) / len(quotes),
    }


def main() -> None:
    demo, evaluation, train = (pd.read_parquet(path) for path in (DEMO_INPUT, EVAL_INPUT, TRAIN_INPUT))

    print(f"Uczę TF-IDF + regresję logistyczną na {len(train)} opiniach...")
    start = time.perf_counter()
    tfidf = train_tfidf([review_text(a, b) for a, b in zip(train["title"], train["text"])], train["sentiment"].tolist())
    train_seconds = time.perf_counter() - start
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(tfidf, TFIDF_PATH)

    print(f"Porównanie metod na {len(evaluation)} opiniach...")
    evaluated, seconds = run_methods(evaluation, tfidf, show_progress)
    metrics = {
        "date": date.today().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "llm_model": LLM_MODEL,
        "llm_workers": LLM_WORKERS,
        "transformer_model": TRANSFORMER_MODEL,
        "tfidf_train_reviews": len(train),
        "tfidf_train_seconds": train_seconds,
        **compare(evaluated, seconds),
    }

    print(f"Analiza produktów demo ({len(demo)} opinii)...")
    demo_done, _ = run_methods(demo, tfidf, show_progress)
    titles = product_titles(DEMO_PRODUCT_IDS)
    products = {}
    for product_id in DEMO_PRODUCT_IDS:
        print(f"  Raport: {titles[product_id][:60]}")
        reviews = demo_done[demo_done["parent_asin"] == product_id]
        products[product_id] = {"title": titles[product_id], "report": write_report(titles[product_id], reviews)}

    PRECOMPUTED_DIR.mkdir(exist_ok=True)
    evaluated[COLUMNS].to_parquet(EVAL_RESULTS, index=False)
    demo_done[COLUMNS].to_parquet(DEMO_RESULTS, index=False)
    PRODUCTS_RESULTS.write_text(json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["methods"], indent=2))


if __name__ == "__main__":
    main()
