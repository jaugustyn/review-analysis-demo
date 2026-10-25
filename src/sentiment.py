"""Sentiment without an LLM: a dictionary (VADER), TF-IDF + logistic regression and a ready transformer model.

Heavy libraries are imported inside the functions: the app imports this module only for the live analysis.
"""

from sklearn.pipeline import Pipeline

from src.data import SENTIMENTS


def vader(texts: list[str]) -> list[str]:
    """VADER compound score with its authors' thresholds: >= 0.05 positive, <= -0.05 negative."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(text)["compound"] for text in texts]
    return ["positive" if s >= 0.05 else "negative" if s <= -0.05 else "neutral" for s in scores]


def train_tfidf(texts: list[str], labels: list[str]) -> Pipeline:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True),
        # Balanced weights, otherwise the rare neutral and negative reviews are mostly called positive.
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    return model.fit(texts, labels)


def _label_to_sentiment(label: str) -> str:
    """Maps model labels ("negative", "Very Positive", "4 stars", ...) to our three classes."""
    label = label.lower()
    if "star" in label:
        stars = int(label[0])
        return "negative" if stars <= 2 else "neutral" if stars == 3 else "positive"
    return next(s for s in SENTIMENTS if s[:3] in label)


def transformer(texts: list[str], model_name: str) -> list[str]:
    import torch
    from transformers import pipeline

    classify = pipeline(
        "text-classification", model=model_name, device=0 if torch.cuda.is_available() else -1,
        truncation=True, max_length=512,
    )
    return [_label_to_sentiment(result["label"]) for result in classify(texts, batch_size=32)]
