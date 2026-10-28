"""Runs the four methods once: on the Allegro Reviews test split (to compare them) and on the demo products
from Ceneo (for the app).

Needs an NVIDIA GPU for the transformer model and a running Ollama server with LLM_MODEL from src/config.py.
Run after prepare_data.py: uv run python analyze.py
"""

import json
from datetime import date

import joblib
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.config import (
    DEMO_INPUT, DEMO_RESULTS, EVAL_RESULTS, LLM_MODEL, METRICS_PATH, MODELS_DIR, PRECOMPUTED_DIR, PRODUCTS_RESULTS,
    TFIDF_PATH, TRANSFORMER_MODEL,
)
from src.data import SENTIMENTS, load_allegro
from src.pipeline import LLM_WORKERS, METHODS, run_methods, write_report
from src.sentiment import train_tfidf

RESULT_COLUMNS = [
    "rating", "sentiment", "text", *(f"sent_{method}" for method in METHODS), "topics_keywords", "topics_llm",
    "llm_input_tokens", "llm_output_tokens",
]
DEMO_COLUMNS = ["product_id", "product_title", "date", *RESULT_COLUMNS]


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
        "llm_tokens_per_1000": {
            "input": int(evaluated["llm_input_tokens"].sum() / n * 1000),
            "output": int(evaluated["llm_output_tokens"].sum() / n * 1000),
        },
        "llm_quotes_verbatim": sum(quotes) / len(quotes),
    }


def main() -> None:
    train, evaluation, demo = load_allegro("train"), load_allegro("test"), pd.read_parquet(DEMO_INPUT)

    print(f"Uczę TF-IDF + regresję logistyczną na {len(train)} opiniach z Allegro Reviews...")
    tfidf = train_tfidf(train["text"].tolist(), train["sentiment"].tolist())
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(tfidf, TFIDF_PATH)

    print(f"Porównanie metod na {len(evaluation)} opiniach z Allegro Reviews...")
    evaluated, seconds = run_methods(evaluation, tfidf, show_progress)
    metrics = {
        "date": date.today().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "llm_model": LLM_MODEL,
        "llm_workers": LLM_WORKERS,
        "transformer_model": TRANSFORMER_MODEL,
        "tfidf_train_reviews": len(train),
        **compare(evaluated, seconds),
    }

    print(f"Analiza produktów demo z Ceneo ({len(demo)} opinii)...")
    demo_done, _ = run_methods(demo, tfidf, show_progress)
    products = {}
    for product_id, reviews in demo_done.groupby("product_id", sort=False):
        first = reviews.iloc[0]
        print(f"  Raport: {first['product_title'][:60]}")
        products[product_id] = {
            "title": first["product_title"], "ceneo_rating": first["ceneo_rating"],
            "ceneo_reviews": int(first["ceneo_reviews"]), "report": write_report(first["product_title"], reviews),
        }

    PRECOMPUTED_DIR.mkdir(exist_ok=True)
    evaluated[RESULT_COLUMNS].to_parquet(EVAL_RESULTS, index=False)
    demo_done[DEMO_COLUMNS].to_parquet(DEMO_RESULTS, index=False)
    PRODUCTS_RESULTS.write_text(json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["methods"], indent=2))


if __name__ == "__main__":
    main()
