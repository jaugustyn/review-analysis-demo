"""Sentiment without an LLM: a Polish sentiment dictionary, TF-IDF + logistic regression and a ready model.

Heavy libraries are imported inside the functions: the app imports this module only for the live analysis.
"""

import re
from functools import cache

from sklearn.pipeline import Pipeline

from src.config import LEXICON_PATH
from src.data import SENTIMENTS


@cache
def _lexicon() -> dict[str, int]:
    """Base forms of words with scores from -2 (very negative) to 2 (very positive)."""
    import pandas as pd

    df = pd.read_csv(LEXICON_PATH, sep="\t", header=None, usecols=[0, 4], names=["word", "score"], quoting=3)
    return dict(zip(df["word"].str.lower(), df["score"]))


def lexicon_score(text: str) -> float:
    """Sum of the scores of the words in the text. Each word is reduced to its base form first
    ("zepsuła" -> "zepsuć"), and "nie" reverses the score of the next word."""
    import simplemma

    lexicon = _lexicon()
    total, negate = 0, False
    for word in re.findall(r"\w+", text.lower()):
        if word == "nie":
            negate = True
            continue
        score = lexicon.get(simplemma.lemmatize(word, lang="pl"), lexicon.get(word, 0))
        total += -score if negate else score
        negate = False
    return total


def lexicon(texts: list[str]) -> list[str]:
    """Positive when the scores add up above zero, negative below zero, otherwise neutral."""
    scores = [lexicon_score(text) for text in texts]
    return ["positive" if s > 0 else "negative" if s < 0 else "neutral" for s in scores]


def train_tfidf(texts: list[str], labels: list[str]) -> Pipeline:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
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
