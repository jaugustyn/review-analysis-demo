"""Shared settings: paths, data sources, demo products and model names."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Downloaded by prepare_data.py (not in git)
DATA_DIR = ROOT / "data"
# Allegro Reviews (KLEJ benchmark, CC BY-SA 4.0): Polish product reviews with 1–5 stars,
# to train TF-IDF and to compare the methods.
ALLEGRO_URL = "https://huggingface.co/api/datasets/allegro/klej-allegro-reviews/parquet/default/{split}/0.parquet"
ALLEGRO_TRAIN = DATA_DIR / "allegro_train.parquet"
ALLEGRO_TEST = DATA_DIR / "allegro_test.parquet"
# Polish sentiment dictionary by A. Wawer (IPI PAN): base forms of words with scores from -2 to 2.
LEXICON_URL = "https://zil.ipipan.waw.pl/SlownikWydzwieku?action=AttachFile&do=get&target=slownikWydzwieku01.csv"
LEXICON_PATH = DATA_DIR / "slownik_wydzwieku.csv"
DEMO_INPUT = DATA_DIR / "demo.parquet"  # reviews of the demo products, fetched from Ceneo

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
# Ceneo product IDs (https://www.ceneo.pl/<ID>). The first page of reviews of each is fetched by prepare_data.py.
DEMO_PRODUCT_IDS = [
    "120252432",  # fridge
    "50562241",   # coffee machine
    "170194155",  # washing machine
    "182519828",  # vacuum cleaner
    "167406299",  # air fryer
    "45033962",   # electric kettle
    "101400465",  # dishwasher
    "99493041",   # dishwasher
    "99047227",   # fridge
    "130004065",  # hair dryer
]
CENEO_PAUSE = 3  # seconds between product pages fetched one after another
LIVE_EXAMPLE_LINK = "https://www.ceneo.pl/157145031"  # Apple iPhone 15 256GB, not a demo product

# Multilingual, trained on product reviews (1–5 stars). A Polish model (Voicelab HerBERT) scored much higher,
# but it was most likely trained on Allegro Reviews, so it cannot be compared fairly on that data.
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
