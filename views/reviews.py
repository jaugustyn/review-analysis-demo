"""Reviews of the demo products with the verdict of each method: where does which method go wrong."""

import json

import pandas as pd
import streamlit as st

from src.app_data import load_demo
from src.ceneo import product_link
from src.formatting import METHOD_LABELS, SENTIMENT_LABELS, SOURCE_LABELS, md_escape, shorten
from src.topics import TOPIC_LABELS

MISS = "background-color: #FCE3D8"
VIEWS = ["Wszystkie", "Metody się nie zgadzają", "Słownik się myli, LLM nie"]
SHORT_LABELS = {"lexicon": "Słownik", "tfidf": "Klasyczne ML", "transformer": "Gotowy model", "llm": "LLM"}

evaluated = load_demo()


def stars(rating: float) -> str:
    return "★" * int(rating) + ("½" if rating % 1 else "")


st.title("Opinie")
st.caption("Opinie produktów demo z Ceneo i werdykt każdej metody; pomyłki są zaznaczone kolorem. "
           "Kliknij wiersz, aby zobaczyć całą opinię.")

view = st.segmented_control("Pokaż", VIEWS, default=VIEWS[1], required=True)
verdicts = evaluated[[f"sent_{m}" for m in METHOD_LABELS]]
if view == VIEWS[1]:
    shown = evaluated[verdicts.nunique(axis=1) > 1]
elif view == VIEWS[2]:
    shown = evaluated[(evaluated["sent_lexicon"] != evaluated["sentiment"]) & (evaluated["sent_llm"] == evaluated["sentiment"])]
else:
    shown = evaluated
shown = shown.reset_index(drop=True)
st.caption(f"{len(shown)} opinii.")

table = pd.DataFrame({
    "Ocena": shown["rating"].map(stars),
    "Opinia": shown["text"].map(lambda t: shorten(t, 140)),
    "Produkt": shown["product_title"].map(lambda t: shorten(t, 60)),
    "Link": shown["product_id"].map(product_link),
    **{label: shown[f"sent_{m}"].map(SENTIMENT_LABELS) for m, label in SHORT_LABELS.items()},
})
misses = pd.DataFrame("", index=table.index, columns=table.columns)
for method, label in SHORT_LABELS.items():
    misses[label] = (shown[f"sent_{method}"] != shown["sentiment"]).map({True: MISS, False: ""})
event = st.dataframe(table.style.apply(lambda _: misses, axis=None), hide_index=True, width="stretch", height=460,
                     on_select="rerun", selection_mode="single-row", key=f"reviews-{view}",
                     column_config={
                         "Ocena": st.column_config.TextColumn(
                             width=70, help="Ocena klienta: poniżej 3★ negatywna, 3★ neutralna, powyżej 3★ pozytywna."),
                         "Opinia": st.column_config.TextColumn(width=240),
                         "Produkt": st.column_config.TextColumn(width=150),
                         "Link": st.column_config.LinkColumn(
                             width=40, display_text=":material/open_in_new:",
                             help="Strona produktu na Ceneo."),
                         **{c: st.column_config.TextColumn(width=100) for c in SHORT_LABELS.values()},
                     })

if not event.selection.rows:
    st.stop()
row = shown.iloc[event.selection.rows[0]]
st.subheader(stars(row["rating"]))
st.markdown(md_escape(row["text"]))
st.caption(f"{row['date']:%d.%m.%Y} · według gwiazdek: {SENTIMENT_LABELS[row['sentiment']]} · "
           f"[{md_escape(shorten(row['product_title'], 70))}]({product_link(row['product_id'])})")
for column, (source, label) in zip(st.columns(2, gap="large"), SOURCE_LABELS.items()):
    column.markdown(f"**Tematy – {label}**")
    topics = json.loads(row[f"topics_{source}"])
    for item in topics:
        sign = "chwali" if item["polarity"] == "positive" else "krytykuje"
        note = "" if item.get("quote_ok", True) else " (cytatu nie ma w opinii)"
        column.markdown(f"- {TOPIC_LABELS[item['topic']]}: {sign} – „{md_escape(item['quote'])}”{note}")
    if not topics:
        column.caption("Nie znaleziono tematów.")
