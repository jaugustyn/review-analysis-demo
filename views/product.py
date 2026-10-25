"""One product: what customers praise and criticise, by topic, with quotes and a report for the manager."""

import altair as alt
import pandas as pd
import streamlit as st

from src.app_data import live_unavailable_reason, load_demo, load_metrics, load_products
from src.config import LIVE_REVIEWS
from src.formatting import BLUE, ORANGE, SOURCE_LABELS, fmt_int, fmt_num, fmt_pct, fmt_seconds, md_escape, shorten
from src.topics import TOPIC_LABELS, topic_summary

products = load_products()
demo = load_demo()
metrics = load_metrics()
live = st.session_state.setdefault("live_products", {})
if "select_next" in st.session_state:  # a product just analysed live; set before the select box exists
    st.session_state.product = st.session_state.pop("select_next")

st.title("Produkt")
st.caption("Co klienci chwalą i na co narzekają, według tematów, z cytatami z opinii.")

labels = {pid: shorten(p["title"], 90) for pid, p in products.items()}
labels |= {pid: shorten(p["title"], 70) + " · analiza na żywo" for pid, p in live.items()}
product_id = st.selectbox("Produkt", list(labels), format_func=labels.get, key="product")

with st.expander("Analiza na żywo: wklej link do produktu AGD z Amazona", icon=":material/link:"):
    reason = live_unavailable_reason()
    if reason:
        st.info(f"Analiza na żywo działa tylko przy lokalnym uruchomieniu. Teraz jest niedostępna: {reason}.")
    else:
        st.caption(f"Bierzemy {LIVE_REVIEWS} najnowszych opinii produktu z pobranego zbioru (opinie do 2023 roku), "
                   "nie z Amazona. Analiza trwa około 2 minut.")
        link = st.text_input("Link do produktu", placeholder="https://www.amazon.com/dp/B07NVJP82G")
        if st.button("Analizuj", type="primary", disabled=not link):
            from src.live import LiveError, analyse_link

            bar = st.progress(0.0, "Analiza opinii...")
            try:
                result = analyse_link(link, lambda done, total: bar.progress(done / total, f"LLM: {done} z {total} opinii"))
            except LiveError as error:
                st.error(str(error))
            else:
                live[result["product_id"]] = result
                st.session_state.select_next = result["product_id"]
                st.rerun()

if product_id in live:
    entry = live[product_id]
    reviews, report, title = entry["reviews"], entry["report"], entry["title"]
    st.caption(f"Analiza na żywo zajęła {fmt_seconds(entry['seconds'])}.")
else:
    reviews = demo[demo["parent_asin"] == product_id]
    report, title = products[product_id]["report"], products[product_id]["title"]

st.markdown(f"**{md_escape(title)}**")
c1, c2, c3, c4 = st.columns(4)
period = f"{reviews['timestamp'].min():%m.%Y}–{reviews['timestamp'].max():%m.%Y}"
c1.metric("Przeanalizowane opinie", fmt_int(len(reviews)), border=True, help=f"Najnowsze opinie z okresu {period}.")
c2.metric("Średnia ocena", f"{fmt_num(reviews['rating'].mean(), 1)} ★", border=True)
c3.metric("Opinie pozytywne", fmt_pct((reviews["rating"] >= 4).mean()), border=True, help="Oceny 4–5★.")
c4.metric("Opinie negatywne", fmt_pct((reviews["rating"] <= 2).mean()), border=True, help="Oceny 1–2★.")

st.subheader("Raport dla menedżera")
st.markdown(md_escape(report["summary"]))
col_good, col_bad = st.columns(2, gap="large")
col_good.markdown("**Co klienci chwalą**\n" + "\n".join(f"- {md_escape(item)}" for item in report["strengths"]))
col_bad.markdown("**Co poprawić**\n" + "\n".join(f"- {md_escape(item)}" for item in report["improvements"]))
st.caption(f"Raport napisał LLM ({metrics['llm_model']}) na podstawie tematów i cytatów z opinii poniżej.")

st.subheader("Tematy")
source = st.segmented_control("Źródło tematów", ["llm", "keywords"], default="llm", required=True,
                              format_func=SOURCE_LABELS.get, key="topic_source")
summary = topic_summary(reviews, source)
chart_data = pd.DataFrame({
    "Temat": summary["topic"].map(TOPIC_LABELS),
    "Kierunek": summary["polarity"].map({"positive": "Chwalą", "negative": "Narzekają"}),
    "Opinie": summary["reviews"],
    "Wartość": summary["reviews"].where(summary["polarity"] == "positive", -summary["reviews"]),
})
bars = alt.Chart(chart_data).mark_bar(size=22).encode(
    x=alt.X("Wartość:Q", title="Liczba opinii (narzekają ← → chwalą)", axis=alt.Axis(labelExpr="abs(datum.value)")),
    y=alt.Y("Temat:N", sort=list(TOPIC_LABELS.values()), title=None),
    color=alt.Color("Kierunek:N", title=None, legend=alt.Legend(orient="top"),
                    scale=alt.Scale(domain=["Chwalą", "Narzekają"], range=[BLUE, ORANGE])),
    tooltip=["Temat:N", "Kierunek:N", "Opinie:Q"],
)
right = bars.transform_filter("datum.Wartość > 0").mark_text(align="left", dx=4).encode(
    text="Opinie:Q", color=alt.value("#172033"))
left = bars.transform_filter("datum.Wartość < 0").mark_text(align="right", dx=-4).encode(
    text="Opinie:Q", color=alt.value("#172033"))
st.altair_chart((bars + right + left).properties(height=210), width="stretch")
st.caption("Ile opinii chwali lub krytykuje dany temat. Jedna opinia może dotyczyć kilku tematów.")

for topic, label in TOPIC_LABELS.items():
    rows = summary[summary["topic"] == topic].set_index("polarity")
    praise, complaints = rows.loc["positive"], rows.loc["negative"]
    with st.expander(f"{label}: chwalą {praise['reviews']}, narzekają {complaints['reviews']}"):
        for column, heading, row in zip(st.columns(2, gap="large"), ["Chwalą", "Narzekają"], [praise, complaints]):
            column.markdown(f"**{heading}**")
            for quote in row["quotes"][:3]:
                column.markdown(f"> {md_escape(shorten(quote, 300))}")
            if not row["quotes"]:
                column.caption("Brak opinii.")
