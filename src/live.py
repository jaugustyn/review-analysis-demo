"""Live analysis of one product from a pasted Amazon link. Reviews come from the downloaded data set,
not from Amazon. Works only locally: it needs the analysis libraries, the data, a GPU and Ollama."""

from collections.abc import Callable
from importlib.util import find_spec

from src.config import LIVE_REVIEWS, LLM_MODEL, OLLAMA_URL, REVIEWS_PATH, TFIDF_PATH


class LiveError(Exception):
    """A problem to show to the user, in Polish."""


def unavailable_reason() -> str | None:
    """None when the live analysis can run here, otherwise the reason why not."""
    # Only check that the libraries are installed: importing PyTorch would slow down the first page load.
    if not all(find_spec(name) for name in ("ollama", "torch", "transformers", "vaderSentiment", "sklearn")):
        return "brak bibliotek do analizy – uruchom aplikację lokalnie po `uv sync`"
    if not REVIEWS_PATH.exists() or not TFIDF_PATH.exists():
        return "brak pobranych opinii lub modelu TF-IDF – uruchom `prepare_data.py` i `analyze.py`"
    try:
        import ollama

        models = [m.model for m in ollama.Client(host=OLLAMA_URL).list().models]
    except Exception:
        return "Ollama nie działa – uruchom ją (`ollama serve`)"
    if LLM_MODEL not in models:
        return f"w Ollamie brakuje modelu {LLM_MODEL} – pobierz go (`ollama pull {LLM_MODEL}`)"
    return None


def analyse_link(link: str, on_progress: Callable[[int, int], None]) -> dict:
    import joblib

    from src.data import load_product_reviews, product_id_from_link, product_titles
    from src.pipeline import run_methods, write_report

    product_id = product_id_from_link(link)
    if not product_id:
        raise LiveError("Nie rozpoznaję linku. Wklej link do strony produktu na Amazonie (z fragmentem /dp/…).")
    found = load_product_reviews(product_id, LIVE_REVIEWS)
    if not found:
        raise LiveError("Tego produktu nie ma w zbiorze. Analiza działa dla produktów AGD z Amazon Reviews 2023.")
    product_id, reviews = found
    title = product_titles([product_id]).get(product_id, product_id)
    analysed, seconds = run_methods(reviews, joblib.load(TFIDF_PATH), on_progress)
    return {"product_id": product_id, "title": title, "reviews": analysed,
            "report": write_report(title, analysed), "seconds": sum(seconds.values())}
