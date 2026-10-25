"""Downloads Amazon Reviews 2023 (Appliances) and prepares the demo, evaluation and training sets.

Run: uv run python prepare_data.py
"""

import shutil
import urllib.request

from src.config import DATA_DIR, DATA_URL, DEMO_INPUT, EVAL_INPUT, RAW_META, RAW_REVIEWS, TRAIN_INPUT
from src.data import convert_products, convert_reviews, make_splits, product_titles

DOWNLOADS = {
    RAW_REVIEWS: f"{DATA_URL}/review_categories/Appliances.jsonl",
    RAW_META: f"{DATA_URL}/meta_categories/meta_Appliances.jsonl",
}


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for path, url in DOWNLOADS.items():
        if not path.exists():
            print(f"Pobieram {path.name} (to może potrwać kilka minut)...")
            with urllib.request.urlopen(url) as response, open(path, "wb") as f:
                shutil.copyfileobj(response, f)
    print("Konwertuję opinie i opisy produktów do Parquet...")
    convert_reviews()
    convert_products()

    demo, evaluation, train = make_splits()
    demo.to_parquet(DEMO_INPUT, index=False)
    evaluation.to_parquet(EVAL_INPUT, index=False)
    train.to_parquet(TRAIN_INPUT, index=False)

    print(f"Demo: {len(demo)} opinii, {demo['parent_asin'].nunique()} produktów")
    for product_id, title in product_titles(demo["parent_asin"].unique().tolist()).items():
        print(f"  {product_id}  {title[:90]}")
    print(f"Porównanie metod: {len(evaluation)} opinii, rozkład: "
          f"{evaluation['sentiment'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"Nauka TF-IDF: {len(train)} opinii")


if __name__ == "__main__":
    main()
