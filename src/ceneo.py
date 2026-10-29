"""Reviews of one product from its page on Ceneo.pl.

Only the product page is fetched, with one request. It shows the first page of reviews (about 10, in the order
Ceneo chooses). Further pages (/<ID>/opinie-2 and so on) open only in a browser, after a script checks that it is
not a bot; we do not get around that.
Reviewer names are not kept. httpx and BeautifulSoup are imported inside fetch_product: the app itself (and Docker)
uses only product_link and does not install them.
"""

import re

import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "pl-PL,pl;q=0.9",
}


class CeneoError(Exception):
    """A problem to show to the user, in Polish."""


def product_id_from_link(link: str) -> str | None:
    """Product ID from a link such as https://www.ceneo.pl/157145031#tab=reviews, or a bare ID."""
    match = re.search(r"(?:ceneo\.pl/|^)(\d{4,})(?:\D|$)", link.strip())
    return match.group(1) if match else None


def product_link(product_id: str) -> str:
    return f"https://www.ceneo.pl/{product_id}"


def _number(text: str) -> float:
    return float(re.search(r"\d+(?:[.,]\d+)?", text.replace("\xa0", "").replace(" ", "")).group().replace(",", "."))


def fetch_product(product_id: str) -> dict:
    """{"title", "ceneo_rating" (average of all reviews), "ceneo_reviews" (number of all reviews),
    "reviews" (DataFrame: rating, text, date)}."""
    import httpx
    from bs4 import BeautifulSoup

    try:
        response = httpx.get(product_link(product_id), headers=HEADERS, follow_redirects=True, timeout=20)
    except httpx.HTTPError as error:
        raise CeneoError(f"Nie udało się połączyć z Ceneo ({error}).") from error
    page = BeautifulSoup(response.text, "html.parser")
    title = page.select_one("h1.product-top__product-info__name")
    if response.status_code != 200 or not title:
        raise CeneoError("Ceneo nie pokazało strony produktu. Sprawdź link albo spróbuj za chwilę.")

    reviews = []
    for post in page.select("div.js_product-review"):
        score = post.select_one(".user-post__score-count")  # a shop's answer has no score
        text = post.select_one(".user-post__text")
        if not score or not text or not text.get_text(strip=True):
            continue
        date = post.select_one("time[datetime]")
        reviews.append({
            "rating": _number(score.get_text()),
            "text": re.sub(r"\s+", " ", text.get_text(" ")).strip(),
            "date": pd.Timestamp(date["datetime"]) if date else pd.NaT,
        })
    rating = page.select_one(".product-review__score")
    count = page.select_one(".product-review__qo")
    return {
        "title": title.get_text(" ", strip=True),
        "ceneo_rating": _number(rating.get("content") or rating.get_text()) if rating else None,
        "ceneo_reviews": int(_number(count.get_text())) if count else len(reviews),
        "reviews": pd.DataFrame(reviews, columns=["rating", "text", "date"]),
    }
