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
# Products of brands well known in Poland, of different types and ratings. The data has few whole appliances
# of such brands, so some are accessories.
DEMO_PRODUCT_IDS = [
    "B01CBWDY0C",  # refillable capsules for Nespresso machines
    "B0C69HCVRM",  # BLACK+DECKER mini fridge
    "B002UYSHMM",  # Haier portable washer
    "B08FDBWTDB",  # BLACK+DECKER countertop dishwasher
    "B08YBGCNHP",  # SAMSUNG refrigerator water filter
    "B002Y2S536",  # BOSCH refrigerator water filter
    "B0BMPQXGMG",  # LG washing machine drain pump
    "B07JMNYK7X",  # De'Longhi milk frothing pitcher
    "B01251BJX8",  # KRUPS permanent coffee filter
    "B00WA6TGDE",  # Melitta cone coffee filters
]
DEMO_REVIEWS_PER_PRODUCT = 200  # newest reviews of each demo product
LIVE_REVIEWS = 100  # newest reviews of a product analysed live
LIVE_EXAMPLE_LINK = "https://www.amazon.com/dp/B0C1ZXTW1T"  # Electrolux water filter, not a demo product
EVAL_SIZE = 1000  # the same reviews for every method
TRAIN_SIZE = 100_000  # training data for TF-IDF + logistic regression

# Trained on product reviews (1–5 stars); best of three candidates on the evaluation set.
TRANSFORMER_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
# Not "localhost": on Windows it tries IPv6 first and every new connection waits about 2 s.
OLLAMA_URL = "http://127.0.0.1:11434"
# Best of five local models that fit a 12 GB GPU (sentiment F1, speed, sensible topics); needs Ollama 0.40+.
LLM_MODEL = "gemma4:12b"

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
