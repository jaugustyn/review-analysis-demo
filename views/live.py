"""Live analysis: a pasted Ceneo link, the four methods on the reviews from the product's page."""

import streamlit as st

from src.app_data import live_unavailable_reason
from src.config import LIVE_EXAMPLE_LINK, LLM_MODEL
from src.formatting import fmt_seconds
from src.product_view import show_product

st.title("Analiza na żywo")
st.caption("Wklej link do dowolnego produktu na Ceneo, a aplikacja przeanalizuje jego opinie.")

reason = live_unavailable_reason()
if reason:
    st.info(f"Analiza na żywo działa tylko przy lokalnym uruchomieniu. Teraz jest niedostępna: {reason}.",
            icon=":material/info:")
    st.stop()

with st.form("live"):
    link = st.text_input(
        "Link do produktu na Ceneo", placeholder=LIVE_EXAMPLE_LINK,
        help="Bierzemy opinie z pierwszej strony produktu na Ceneo (zwykle 10, w kolejności wybranej przez Ceneo). "
             "Kolejnych stron Ceneo nie udostępnia bez weryfikacji, że jesteś człowiekiem.",
    )
    submitted = st.form_submit_button("Analizuj", type="primary")
st.caption(f"Przykład: `{LIVE_EXAMPLE_LINK}`")

if submitted and link:
    from src.live import LiveError, analyse_link

    bar = st.progress(0.0, "Pobieranie opinii z Ceneo...")
    try:
        st.session_state.live_result = analyse_link(
            link, lambda done, total: bar.progress(done / total, f"LLM: {done} z {total} opinii"))
    except LiveError as error:
        st.error(str(error))
    bar.empty()

result = st.session_state.get("live_result")
if result:
    st.caption(f"Analiza zajęła {fmt_seconds(result['seconds'])}.")
    show_product(result["product_id"], result, result["reviews"], LLM_MODEL)
