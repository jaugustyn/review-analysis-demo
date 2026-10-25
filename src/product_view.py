"""One product on screen: tiles, the report for the manager and topics with quotes.
Shared by the Produkt page and the live analysis."""

import altair as alt
import pandas as pd
import streamlit as st

from src.data import amazon_link
from src.formatting import INK, NEGATIVE, POSITIVE, SOURCE_LABELS, fmt_int, fmt_num, fmt_pct, md_escape, shorten
from src.topics import TOPIC_LABELS, topic_summary


def show_product(product_id: str, title: str, reviews: pd.DataFrame, report: dict, llm_model: str) -> None:
    st.markdown(f"**{md_escape(title)}** · [strona produktu]({amazon_link(product_id)})")
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
    st.caption(f"Raport napisał LLM ({llm_model}) na podstawie tematów i cytatów z opinii.")

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
    # The same scale on both sides, with room for the numbers next to the longest bar
    limit = max(chart_data["Opinie"].max(), 1) * 1.15
    bars = alt.Chart(chart_data).mark_bar(size=22, cornerRadius=3).encode(
        x=alt.X("Wartość:Q", title="Liczba opinii (narzekają ← → chwalą)", scale=alt.Scale(domain=[-limit, limit]),
                axis=alt.Axis(labelExpr="abs(datum.value)")),
        y=alt.Y("Temat:N", sort=list(TOPIC_LABELS.values()), title=None),
        color=alt.Color("Kierunek:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["Chwalą", "Narzekają"], range=[POSITIVE, NEGATIVE])),
        tooltip=["Temat:N", "Kierunek:N", "Opinie:Q"],
    )
    right = bars.transform_filter("datum.Wartość > 0").mark_text(align="left", dx=4).encode(
        text="Opinie:Q", color=alt.value(INK))
    left = bars.transform_filter("datum.Wartość < 0").mark_text(align="right", dx=-4).encode(
        text="Opinie:Q", color=alt.value(INK))
    st.altair_chart((bars + right + left).properties(height=250), width="stretch")
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
