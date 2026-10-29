"""Comparison of the four methods: quality, time and what the LLM would cost through an API."""

import json

import altair as alt
import pandas as pd
import streamlit as st

from src.app_data import load_eval, load_metrics
from src.config import API_PRICES_CHECKED
from src.formatting import (
    GRAY, INK, METHOD_LABELS, NEGATIVE, NEUTRAL, POSITIVE, SENTIMENT_LABELS, SOURCE_LABELS, TEAL, fmt_int, fmt_num,
    fmt_pct, fmt_pln, fmt_seconds,
)
from src.llm import api_costs
from src.topics import TOPIC_LABELS, manual_check

metrics = load_metrics()
evaluated = load_eval()
methods = metrics["methods"]
n = metrics["eval_reviews"]
positive = metrics["class_shares"].get("positive", 0)

st.title("Porównanie metod")
st.caption(f"Te same {fmt_int(n)} polskich opinii ze zbioru Allegro Reviews dla każdej metody; "
           "prawdą jest ocena klienta.")

best = max(methods, key=lambda m: methods[m]["f1_macro"])
best_negative = max(methods, key=lambda m: methods[m]["f1"]["negative"])
st.markdown(f"Najlepiej rozpoznaje sentyment **{METHOD_LABELS[best]}**, a opinie negatywne najlepiej wyłapuje "
            f"**{METHOD_LABELS[best_negative]}**. Słownik jest najprostszy, ale wyraźnie słabszy.")

table = pd.DataFrame({
    "Metoda": [METHOD_LABELS[m] for m in methods],
    "Trafność": [fmt_pct(v["accuracy"], 1) for v in methods.values()],
    "F1 makro": [fmt_num(v["f1_macro"]) for v in methods.values()],
    "Czas na 1000 opinii": [fmt_seconds(v["seconds_per_1000"]) for v in methods.values()],
})
st.dataframe(table, hide_index=True, width="stretch", column_config={
    "Trafność": st.column_config.TextColumn(
        alignment="right", help="Jaka część opinii dostała dobrą ocenę. Ocena klienta: poniżej 3★ negatywna, "
                                "3★ neutralna, powyżej 3★ pozytywna."),
    "F1 makro": st.column_config.TextColumn(
        alignment="right", help=f"Główna miara: średnia F1 z trzech klas, każda liczy się tak samo. Metoda, która zawsze "
                                f"mówi „pozytywna”, ma trafność {fmt_pct(positive)}, ale F1 makro tylko "
                                f"{fmt_num(2 * positive / (1 + positive) / 3)}."),
    "Czas na 1000 opinii": st.column_config.TextColumn(
        alignment="right", help=f"Zmierzony na {metrics['gpu']}. LLM: {metrics['llm_model']} przez Ollamę."),
})
costs = api_costs(metrics["llm_tokens_per_1000"])
links = ", ".join(f"[{c['model'].split(' (')[0]}]({c['source']})" for c in costs)
st.caption(f"Wszystkie metody działały lokalnie, bez opłat. Przez płatne API LLM kosztowałby "
           f"{fmt_num(min(c['pln'] for c in costs))}–{fmt_pln(max(c['pln'] for c in costs))} za 1000 opinii "
           f"(ceny z {API_PRICES_CHECKED}: {links}).")

st.subheader("Które opinie sprawiają kłopot")
per_class = pd.DataFrame([
    {"Metoda": METHOD_LABELS[m], "Klasa": SENTIMENT_LABELS[c], "F1": v["f1"][c]}
    for m, v in methods.items() for c in SENTIMENT_LABELS
])
per_class["Etykieta"] = per_class["F1"].map(fmt_num)
classes = list(SENTIMENT_LABELS.values())
chart = alt.Chart(per_class).mark_bar(height=12, cornerRadius=3).encode(
    x=alt.X("F1:Q", scale=alt.Scale(domain=[0, 1]), title="F1 (1 = bez błędów)",
            axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1], labelExpr="replace(format(datum.value, '.1f'), '.', ',')")),
    y=alt.Y("Metoda:N", sort=[METHOD_LABELS[m] for m in methods], title=None, axis=alt.Axis(labelLimit=240)),
    yOffset=alt.YOffset("Klasa:N", sort=classes),
    color=alt.Color("Klasa:N", sort=classes, title=None, legend=alt.Legend(orient="top"),
                    scale=alt.Scale(domain=classes, range=[NEGATIVE, NEUTRAL, POSITIVE])),
    tooltip=["Metoda:N", "Klasa:N", alt.Tooltip("F1:Q", format=".2f")],
)
labels = chart.mark_text(align="left", dx=4, fontSize=11).encode(text="Etykieta:N", color=alt.value(INK))
st.altair_chart((chart + labels).properties(height=320), width="stretch")
st.caption("Najtrudniejsze są opinie neutralne (3★): klient pisze o zaletach i wadach naraz.")

st.subheader("Tematy: słownik czy LLM?")
check = manual_check()
if check:
    st.markdown(f"Tematy są poprawne w **{fmt_pct(check['keywords'])}** opinii dla słownika i w "
                f"**{fmt_pct(check['llm'])}** dla LLM (ręczna ocena {check['reviews']} losowych opinii).",
                help="Opinia liczy się jako poprawna, gdy tematy i ich wydźwięk zgadzają się z jej treścią.")
counts = pd.DataFrame([
    {"Temat": TOPIC_LABELS[item["topic"]], "Źródło": SOURCE_LABELS[source]}
    for source in SOURCE_LABELS for raw in evaluated[f"topics_{source}"] for item in json.loads(raw)
]).value_counts().rename("Wzmianki").reset_index()
sources = list(SOURCE_LABELS.values())
chart = alt.Chart(counts).mark_bar(height=14, cornerRadius=3).encode(
    x=alt.X("Wzmianki:Q", title=f"Wzmianki o temacie w {fmt_int(n)} opiniach", axis=alt.Axis(format="d")),
    y=alt.Y("Temat:N", sort=list(TOPIC_LABELS.values()), title=None, axis=alt.Axis(labelLimit=200)),
    yOffset=alt.YOffset("Źródło:N", sort=sources),
    color=alt.Color("Źródło:N", sort=sources, title=None, legend=alt.Legend(orient="top"),
                    scale=alt.Scale(domain=sources, range=[GRAY, TEAL])),
    tooltip=["Temat:N", "Źródło:N", "Wzmianki:Q"],
)
st.altair_chart(chart.properties(height=320), width="stretch")
st.caption(f"Słownik widzi temat tylko wtedy, gdy padnie słowo z listy; LLM rozumie sens. "
           f"{fmt_pct(metrics['llm_quotes_verbatim'])} cytatów LLM jest dosłownie w opinii, pozostałe aplikacja pomija.")
