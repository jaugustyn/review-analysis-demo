"""Test reviews with the verdict of each method: where does which method go wrong."""

import json

import pandas as pd
import streamlit as st

from src.app_data import load_eval
from src.data import review_text
from src.formatting import METHOD_LABELS, SENTIMENT_LABELS, SOURCE_LABELS, md_escape, shorten
from src.topics import TOPIC_LABELS

MISS = "background-color: #FBE3D6"
VIEWS = ["Wszystkie", "Metody się nie zgadzają", "Słownik się myli, LLM nie"]
SHORT_LABELS = {"vader": "Słownik", "tfidf": "Klasyczne ML", "transformer": "Gotowy model", "llm": "LLM"}

evaluated = load_eval()
st.title("Opinie")
st.caption("Opinie testowe z oceną klienta i werdyktem każdej metody. Pomyłki są zaznaczone kolorem. "
           "Zaznacz wiersz, aby zobaczyć całą opinię i wskazane tematy.")

view = st.segmented_control("Pokaż", VIEWS, default=VIEWS[1], required=True)
verdicts = evaluated[[f"sent_{m}" for m in METHOD_LABELS]]
if view == VIEWS[1]:
    shown = evaluated[verdicts.nunique(axis=1) > 1]
elif view == VIEWS[2]:
    shown = evaluated[(evaluated["sent_vader"] != evaluated["sentiment"]) & (evaluated["sent_llm"] == evaluated["sentiment"])]
else:
    shown = evaluated
shown = shown.reset_index(drop=True)
st.caption(f"{len(shown)} opinii.")

table = pd.DataFrame({
    "Ocena": shown["rating"].map(lambda r: "★" * r),
    "Opinia": [shorten(review_text(a, b), 140) for a, b in zip(shown["title"], shown["text"])],
    "Wg gwiazdek": shown["sentiment"].map(SENTIMENT_LABELS),
    **{label: shown[f"sent_{m}"].map(SENTIMENT_LABELS) for m, label in SHORT_LABELS.items()},
})
misses = pd.DataFrame("", index=table.index, columns=table.columns)
for method, label in SHORT_LABELS.items():
    misses[label] = (shown[f"sent_{method}"] != shown["sentiment"]).map({True: MISS, False: ""})
event = st.dataframe(table.style.apply(lambda _: misses, axis=None), hide_index=True, width="stretch", height=420,
                     on_select="rerun", selection_mode="single-row", key=f"reviews-{view}",
                     column_config={
                         "Ocena": st.column_config.TextColumn(width=80),
                         "Opinia": st.column_config.TextColumn(width=300),
                         **{c: st.column_config.TextColumn(width=100)
                            for c in ["Wg gwiazdek", *SHORT_LABELS.values()]},
                     })

if not event.selection.rows:
    st.stop()
row = shown.iloc[event.selection.rows[0]]
st.subheader(f"{'★' * row['rating']} {md_escape(row['title'])}")
st.markdown(md_escape(row["text"]))
st.caption(f"{row['timestamp']:%d.%m.%Y} · według gwiazdek: {SENTIMENT_LABELS[row['sentiment']]}")
for column, (source, label) in zip(st.columns(2, gap="large"), SOURCE_LABELS.items()):
    column.markdown(f"**Tematy – {label}**")
    topics = json.loads(row[f"topics_{source}"])
    for item in topics:
        sign = "chwali" if item["polarity"] == "positive" else "krytykuje"
        note = "" if item.get("quote_ok", True) else " (cytatu nie ma w opinii)"
        column.markdown(f"- {TOPIC_LABELS[item['topic']]}: {sign} – „{md_escape(item['quote'])}”{note}")
    if not topics:
        column.caption("Nie znaleziono tematów.")
