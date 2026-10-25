"""One demo product: what customers praise and criticise, by topic, with quotes and a report for the manager."""

import streamlit as st

from src.app_data import load_demo, load_metrics, load_products
from src.formatting import shorten
from src.product_view import show_product

products = load_products()
demo = load_demo()

st.title("Produkt")
st.caption("Co klienci chwalą i na co narzekają, według tematów, z cytatami z opinii.")

labels = {pid: shorten(p["title"], 90) for pid, p in products.items()}
product_id = st.selectbox("Produkt", list(labels), format_func=labels.get)
product = products[product_id]
show_product(product_id, product["title"], demo[demo["parent_asin"] == product_id], product["report"],
             load_metrics()["llm_model"])
