"""Data: Allegro Reviews to compare the methods, reviews of the demo products from Ceneo, sentiment from stars."""

import numpy as np
import pandas as pd

from src.config import ALLEGRO_TEST, ALLEGRO_TRAIN

SENTIMENTS = ["negative", "neutral", "positive"]


def sentiment_from_rating(rating: pd.Series) -> pd.Series:
    """Below 3 stars negative, 3 neutral, above 3 positive (Ceneo also has half stars)."""
    return pd.Series(np.select([rating < 3, rating == 3], ["negative", "neutral"], "positive"), index=rating.index)


def load_allegro(split: str) -> pd.DataFrame:
    """"train" (to train TF-IDF) or "test" (to compare the methods): text, rating, sentiment."""
    df = pd.read_parquet(ALLEGRO_TRAIN if split == "train" else ALLEGRO_TEST)
    # Some texts are padded with spaces to a fixed width
    return df.assign(text=df["text"].str.strip(), sentiment=sentiment_from_rating(df["rating"]))
