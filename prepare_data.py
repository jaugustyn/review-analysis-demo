"""Downloads the data: Allegro Reviews, the Polish sentiment dictionary and the reviews of the demo products
from Ceneo (the first page of reviews of each product, one request per product).

Run: uv run python prepare_data.py
"""

import shutil
import time
import urllib.request

import pandas as pd

from src.ceneo import fetch_product
from src.config import (
    ALLEGRO_TEST, ALLEGRO_TRAIN, ALLEGRO_URL, CENEO_PAUSE, DATA_DIR, DEMO_INPUT, DEMO_PRODUCT_IDS, LEXICON_PATH,
    LEXICON_URL,
)
from src.data import sentiment_from_rating

DOWNLOADS = {
    ALLEGRO_TRAIN: ALLEGRO_URL.format(split="train"),
    ALLEGRO_TEST: ALLEGRO_URL.format(split="test"),
    LEXICON_PATH: LEXICON_URL,
}


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for path, url in DOWNLOADS.items():
        if not path.exists():
            print(f"Pobieram {path.name}...")
            with urllib.request.urlopen(url) as response, open(path, "wb") as f:
                shutil.copyfileobj(response, f)

    print(f"Pobieram opinie {len(DEMO_PRODUCT_IDS)} produktów z Ceneo...")
    products = []
    for i, product_id in enumerate(DEMO_PRODUCT_IDS):
        if i:
            time.sleep(CENEO_PAUSE)
        product = fetch_product(product_id)
        print(f"  {product_id}  {len(product['reviews'])} z {product['ceneo_reviews']} opinii  {product['title'][:70]}")
        products.append(product["reviews"].assign(
            product_id=product_id, product_title=product["title"],
            ceneo_rating=product["ceneo_rating"], ceneo_reviews=product["ceneo_reviews"],
        ))
    demo = pd.concat(products, ignore_index=True)
    demo["sentiment"] = sentiment_from_rating(demo["rating"])
    demo.to_parquet(DEMO_INPUT, index=False)
    print(f"Produkty demo: {len(demo)} opinii, rozkład: {demo['sentiment'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
