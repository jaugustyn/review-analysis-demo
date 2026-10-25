"""Shared settings: paths, data sizes and model names."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Raw data (downloaded by prepare_data.py, not in git)
DATA_DIR = ROOT / "data"
DATA_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
RAW_REVIEWS = DATA_DIR / "Appliances.jsonl"
RAW_META = DATA_DIR / "meta_Appliances.jsonl"
REVIEWS_PATH = DATA_DIR / "reviews.parquet"  # all reviews, sorted by product, for the live analysis
PRODUCTS_PATH = DATA_DIR / "products.parquet"
DEMO_INPUT = DATA_DIR / "demo.parquet"
EVAL_INPUT = DATA_DIR / "eval.parquet"
TRAIN_INPUT = DATA_DIR / "train.parquet"

# Results computed once by analyze.py (in git, read by the app)
PRECOMPUTED_DIR = ROOT / "precomputed"
DEMO_RESULTS = PRECOMPUTED_DIR / "demo_reviews.parquet"
EVAL_RESULTS = PRECOMPUTED_DIR / "eval_reviews.parquet"
PRODUCTS_RESULTS = PRECOMPUTED_DIR / "products.json"
METRICS_PATH = PRECOMPUTED_DIR / "metrics.json"

MODELS_DIR = ROOT / "models"  # TF-IDF model for the live analysis (not in git)
TFIDF_PATH = MODELS_DIR / "tfidf.joblib"
RESULTS_DIR = ROOT / "results"

SEED = 42
# Popular products of different types (the top 10 by review count are mostly filters).
DEMO_PRODUCT_IDS = [
    "B09YC8YCV6",  # Cuisinart SS-10P1 coffee maker
    "B01ALBMIEI",  # Giantex portable washing machine
    "B07YF9SGBW",  # GE Profile Opal nugget ice maker
    "B07NVJP82G",  # AGLUCKY countertop ice maker
    "B0BW9B4YZL",  # Broan-NuTone range hood insert
    "B0C35WFNZF",  # COSMO 5U30 range hood
    "B0B3DB5HTC",  # K&J Keurig water filters, 12 pack
    "B08YBGCNHP",  # SAMSUNG genuine refrigerator filter
    "B07WTXWC32",  # Linda's Essentials stove gap covers
    "B081K4Q6KQ",  # Cimkiz dishwasher clean/dirty magnet
]
DEMO_REVIEWS_PER_PRODUCT = 200  # newest reviews of each demo product
LIVE_REVIEWS = 100  # newest reviews of a product analysed live
EVAL_SIZE = 1000  # the same reviews for every method
TRAIN_SIZE = 100_000  # training data for TF-IDF + logistic regression

# Trained on product reviews (1–5 stars); best of three candidates on the evaluation set.
TRANSFORMER_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen3.5:9b"  # to be replaced after comparing newer models

# What the LLM part would cost through an API: the cheapest current-generation model of three providers,
# USD per 1M tokens, standard tier, checked on API_PRICES_CHECKED.
API_PRICES = [
    {"model": "Claude Haiku 5.5 (Anthropic)", "input": 0.10, "output": 0.50,
     "source": "https://platform.claude.com/docs/en/about-claude/pricing"},
    {"model": "GPT-6 Luna (OpenAI)", "input": 0.10, "output": 0.50,
     "source": "https://developers.openai.com/api/docs/pricing"},
    {"model": "Gemini 3.5 Flash-Lite (Google)", "input": 0.30, "output": 2.50,
     "source": "https://ai.google.dev/gemini-api/docs/pricing"},
]
API_PRICES_CHECKED = "25.10.2026"
USD_PLN = 3.9247
USD_PLN_SOURCE = "kurs średni NBP z 23.10.2026 (tabela 207/A/NBP/2026)"
