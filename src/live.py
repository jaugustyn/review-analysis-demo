"""Live analysis of one product from a pasted Ceneo link: the reviews on its page, the four methods and a report.
Works only locally: it needs the analysis libraries, the downloaded dictionary, a GPU and Ollama."""

from collections.abc import Callable
from importlib.util import find_spec

from src.config import LEXICON_PATH, LLM_MODEL, OLLAMA_URL, TFIDF_PATH


class LiveError(Exception):
    """A problem to show to the user, in Polish."""


def unavailable_reason() -> str | None:
    """None when the live analysis can run here, otherwise the reason why not."""
    # Only check that the libraries are installed: importing PyTorch would slow down the first page load.
    if not all(find_spec(name) for name in ("ollama", "torch", "transformers", "sklearn", "simplemma", "bs4", "httpx")):
        return "brak bibliotek do analizy – uruchom aplikację lokalnie po `uv sync`"
    if not LEXICON_PATH.exists() or not TFIDF_PATH.exists():
        return "brak słownika lub modelu TF-IDF – uruchom `prepare_data.py` i `analyze.py`"
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

    from src.ceneo import CeneoError, fetch_product, product_id_from_link
    from src.data import sentiment_from_rating
    from src.pipeline import run_methods, write_report

    product_id = product_id_from_link(link)
    if not product_id:
        raise LiveError("Nie rozpoznaję linku. Wklej link do strony produktu na Ceneo, np. https://www.ceneo.pl/157145031.")
    try:
        product = fetch_product(product_id)
    except CeneoError as error:
        raise LiveError(str(error)) from error
    reviews = product["reviews"]
    if reviews.empty:
        raise LiveError("Ten produkt nie ma na Ceneo opinii z treścią.")
    reviews["sentiment"] = sentiment_from_rating(reviews["rating"])
    analysed, seconds = run_methods(reviews, joblib.load(TFIDF_PATH), on_progress)
    return {"product_id": product_id, "title": product["title"], "reviews": analysed,
            "ceneo_rating": product["ceneo_rating"], "ceneo_reviews": product["ceneo_reviews"],
            "report": write_report(product["title"], analysed), "seconds": sum(seconds.values())}
