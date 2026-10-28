"""Głos Klienta – Streamlit entry point: page navigation.

Run: uv run streamlit run app.py
"""

import streamlit as st

from src.config import METRICS_PATH

st.set_page_config(page_title="Głos Klienta", page_icon=":material/reviews:", layout="wide")

if not METRICS_PATH.exists():
    st.error("Brak policzonych wyników. Najpierw uruchom: `uv run python prepare_data.py` i `uv run python analyze.py`")
    st.stop()

page = st.navigation([
    st.Page("views/product.py", title="Produkt", icon=":material/inventory_2:", default=True),
    st.Page("views/live.py", title="Analiza na żywo", icon=":material/bolt:"),
    st.Page("views/comparison.py", title="Porównanie metod", icon=":material/compare_arrows:"),
    st.Page("views/reviews.py", title="Opinie", icon=":material/chat:"),
])
with st.sidebar:
    st.caption("Opinie o produktach: Ceneo.pl. Porównanie metod: zbiór Allegro Reviews.")
page.run()
