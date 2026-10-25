"""Amazon Reviews 2023 (Appliances): conversion, demo / evaluation / training sets and product lookup."""

import html
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    DEMO_PRODUCT_IDS, DEMO_REVIEWS_PER_PRODUCT, EVAL_SIZE, PRODUCTS_PATH, RAW_META, RAW_REVIEWS, REVIEWS_PATH,
    SEED, TRAIN_SIZE,
)

SENTIMENTS = ["negative", "neutral", "positive"]
REVIEW_COLUMNS = ["parent_asin", "asin", "rating", "title", "text", "timestamp"]


def review_text(title: str, text: str) -> str:
    """Title and body together: titles such as "Broke after a week" carry a lot of the sentiment."""
    return f"{title}. {text}" if title and text else title or text


def sentiment_from_rating(rating: pd.Series) -> pd.Series:
    """1–2 stars: negative, 3: neutral, 4–5: positive."""
    return pd.cut(rating, bins=[0, 2, 3, 5], labels=SENTIMENTS).astype(str)


def _clean(text: str) -> str:
    """Removes HTML line breaks and video placeholders such as [[VIDEOID:...]]; a link to another product
    ([[ASIN:B00SPYDSFC Whirlpool Adjuster Kit]]) becomes the product name."""
    text = re.sub(r"<br\s*/?>|\[\[VIDEOID:[^\]]*\]\]", " ", html.unescape(text or ""))
    text = re.sub(r"\[\[ASIN:\w+\s*([^\]]*)\]\]", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _read_jsonl(path: Path, fields: list[str]) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            rows.append([record.get(field) for field in fields])
    return pd.DataFrame(rows, columns=fields)


def convert_reviews() -> None:
    """All reviews in one Parquet file, sorted by product so that the live lookup reads little."""
    df = _read_jsonl(RAW_REVIEWS, [*REVIEW_COLUMNS, "user_id"])
    df = df.drop_duplicates(["user_id", "parent_asin", "timestamp"]).drop(columns="user_id")
    df["title"] = df["title"].map(_clean)
    df["text"] = df["text"].map(_clean)
    df = df[(df["title"] != "") | (df["text"] != "")]
    df["rating"] = df["rating"].astype("int8")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["sentiment"] = sentiment_from_rating(df["rating"])
    df.sort_values(["parent_asin", "timestamp"]).to_parquet(REVIEWS_PATH, index=False, row_group_size=50_000)


def convert_products() -> None:
    df = _read_jsonl(RAW_META, ["parent_asin", "title", "store", "average_rating", "rating_number"])
    df["title"] = df["title"].map(_clean)
    df.to_parquet(PRODUCTS_PATH, index=False)


def make_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Demo: newest reviews of the demo products. Evaluation and training: random reviews
    of the other products, so no method is tested on reviews it learned from.
    Demo and evaluation reviews get the product name, shown in the app."""
    reviews = pd.read_parquet(REVIEWS_PATH)
    is_demo = reviews["parent_asin"].isin(DEMO_PRODUCT_IDS)
    demo = (
        reviews[is_demo]
        .sort_values("timestamp", ascending=False)
        .groupby("parent_asin")
        .head(DEMO_REVIEWS_PER_PRODUCT)
        .reset_index(drop=True)
    )
    rest = reviews[~is_demo]
    evaluation = rest.sample(EVAL_SIZE, random_state=SEED)
    train = rest.drop(evaluation.index).sample(TRAIN_SIZE, random_state=SEED)
    evaluation = evaluation.reset_index(drop=True)
    titles = product_titles([*demo["parent_asin"].unique(), *evaluation["parent_asin"].unique()])
    for df in (demo, evaluation):
        df["product_title"] = df["parent_asin"].map(titles).fillna(df["parent_asin"])
    return demo, evaluation, train.reset_index(drop=True)


def product_id_from_link(link: str) -> str | None:
    """Amazon product ID (ASIN) from a product link such as amazon.com/.../dp/B07XYZ1234, or a bare ID."""
    match = re.search(r"(?:/dp/|/gp/product/|/product-reviews/|^)([A-Z0-9]{10})(?:[/?#]|$)", link.strip())
    return match.group(1) if match else None


def amazon_link(product_id: str) -> str:
    return f"https://www.amazon.com/dp/{product_id}"


def load_product_reviews(product_id: str, limit: int) -> tuple[str, pd.DataFrame] | None:
    """Newest reviews of a product from the downloaded data. The ID may be a product or one of its variants."""
    table = pq.read_table(REVIEWS_PATH, filters=[("parent_asin", "==", product_id)])
    if table.num_rows == 0:
        variant = pq.read_table(REVIEWS_PATH, columns=["parent_asin"], filters=[("asin", "==", product_id)])
        if variant.num_rows == 0:
            return None
        product_id = variant.column("parent_asin")[0].as_py()
        table = pq.read_table(REVIEWS_PATH, filters=[("parent_asin", "==", product_id)])
    reviews = table.to_pandas().sort_values("timestamp", ascending=False).head(limit).reset_index(drop=True)
    return product_id, reviews


def product_titles(product_ids: list[str]) -> dict[str, str]:
    table = pq.read_table(PRODUCTS_PATH, columns=["parent_asin", "title"], filters=[("parent_asin", "in", product_ids)])
    return dict(zip(table.column("parent_asin").to_pylist(), table.column("title").to_pylist()))
