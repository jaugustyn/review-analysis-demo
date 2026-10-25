"""Comparison of the four methods: quality, time and cost of analysing 1000 reviews."""

import json

import altair as alt
import pandas as pd
import streamlit as st

from src.app_data import load_eval, load_metrics
from src.config import API_PRICES, API_PRICES_CHECKED, USD_PLN, USD_PLN_SOURCE
from src.formatting import (
    GRAY, INK, METHOD_LABELS, NEGATIVE, NEUTRAL, POSITIVE, SENTIMENT_LABELS, SOURCE_LABELS, TEAL, fmt_int, fmt_num,
    fmt_pct, fmt_pln, fmt_seconds,
)
from src.topics import TOPIC_LABELS

metrics = load_metrics()
evaluated = load_eval()
methods = metrics["methods"]
n = metrics["eval_reviews"]
right = st.column_config.TextColumn(alignment="right")

st.title("Porównanie metod")
st.caption(f"Te same {fmt_int(n)} opinii dla każdej metody. Prawdą jest ocena klienta: "
           "1–2★ to opinia negatywna, 3★ neutralna, 4–5★ pozytywna.")

best = max(methods, key=lambda m: methods[m]["f1_macro"])
best_negative = max(methods, key=lambda m: methods[m]["f1"]["negative"])
st.markdown(
    f"Najlepiej rozpoznaje sentyment **{METHOD_LABELS[best]}** (F1 makro {fmt_num(methods[best]['f1_macro'])}), "
    f"a opinie negatywne najlepiej wyłapuje **{METHOD_LABELS[best_negative]}** "
    f"(F1 {fmt_num(methods[best_negative]['f1']['negative'])}). "
    f"Słownik jest najtańszy i najszybszy, ale wyraźnie słabszy (F1 makro {fmt_num(methods['vader']['f1_macro'])})."
)
table = pd.DataFrame({
    "Metoda": [METHOD_LABELS[m] for m in methods],
    "Trafność": [fmt_pct(v["accuracy"], 1) for v in methods.values()],
    "F1 makro": [fmt_num(v["f1_macro"]) for v in methods.values()],
    "Czas na 1000 opinii": [fmt_seconds(v["seconds_per_1000"]) for v in methods.values()],
})
st.dataframe(table, hide_index=True, width="stretch", column_config={
    "Trafność": right,
    "F1 makro": st.column_config.TextColumn(
        alignment="right", help="Średnia F1 z trzech klas: każda liczy się tak samo, także rzadkie opinie neutralne."),
    "Czas na 1000 opinii": right,
})
st.caption(f"Wszystkie metody działały lokalnie, bez opłat. Czas zmierzony na {metrics['gpu']}. "
           f"LLM: {metrics['llm_model']} przez Ollamę, "
           f"{metrics['llm_workers']} zapytania naraz. Klasyczne ML uczyło się "
           f"{fmt_seconds(metrics['tfidf_train_seconds'])} na {fmt_int(metrics['tfidf_train_reviews'])} innych opiniach.")

st.subheader("Które opinie sprawiają kłopot")
per_class = pd.DataFrame([
    {"Metoda": METHOD_LABELS[m], "Klasa": SENTIMENT_LABELS[c], "F1": v["f1"][c]}
    for m, v in methods.items() for c in SENTIMENT_LABELS
])
classes = list(SENTIMENT_LABELS.values())
per_class["Etykieta"] = per_class["F1"].map(fmt_num)
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
shares = metrics["class_shares"]
st.caption(f"Opinii neutralnych jest tylko {fmt_pct(shares.get('neutral', 0), 1)}, więc ich wynik jest najmniej pewny. "
           "Wszystkie metody radzą sobie z nimi najgorzej.")

st.subheader("Ile kosztowałby LLM przez API")
tokens = metrics["llm_tokens_per_1000"]
costs = []
for price in API_PRICES:
    pln = (tokens["input"] * price["input"] + tokens["output"] * price["output"]) / 1e6 * USD_PLN
    costs.append({"Model": price["model"], "Koszt na 1000 opinii": fmt_pln(pln),
                  "Koszt na 100 000 opinii": fmt_pln(pln * 100), "Cennik": price["source"]})
st.dataframe(pd.DataFrame(costs), hide_index=True, width="stretch", column_config={
    "Koszt na 1000 opinii": right, "Koszt na 100 000 opinii": right,
    "Cennik": st.column_config.LinkColumn(display_text="cennik"),
})
st.caption(f"Ceny standardowe z cenników z {API_PRICES_CHECKED}; {USD_PLN_SOURCE}. Liczba tokenów pochodzi z modelu "
           f"lokalnego ({fmt_int(tokens['input'])} wejściowych i {fmt_int(tokens['output'])} wyjściowych na 1000 opinii). "
           "Modele w API liczą tokeny trochę inaczej, więc to szacunek.")

st.subheader("Tematy: słownik czy LLM?")
counts = pd.DataFrame([
    {"Temat": TOPIC_LABELS[item["topic"]], "Źródło": SOURCE_LABELS[source]}
    for source in SOURCE_LABELS for raw in evaluated[f"topics_{source}"] for item in json.loads(raw)
]).value_counts().rename("Wzmianki").reset_index()
sources = list(SOURCE_LABELS.values())
chart = alt.Chart(counts).mark_bar(height=14, cornerRadius=3).encode(
    x=alt.X("Wzmianki:Q", title=f"Wzmianki o temacie w {fmt_int(n)} opiniach", axis=alt.Axis(format="d")),
    y=alt.Y("Temat:N", sort=list(TOPIC_LABELS.values()), title=None),
    yOffset=alt.YOffset("Źródło:N", sort=sources),
    color=alt.Color("Źródło:N", sort=sources, title=None, legend=alt.Legend(orient="top"),
                    scale=alt.Scale(domain=sources, range=[GRAY, TEAL])),
    tooltip=["Temat:N", "Źródło:N", "Wzmianki:Q"],
)
st.altair_chart(chart.properties(height=320), width="stretch")
st.markdown(
    f"Słownik znajduje temat tylko wtedy, gdy padnie jedno ze słów z listy, a wydźwięk ocenia po słowach w zdaniu. "
    f"LLM rozumie sens, ale czasem cytuje słowa, których klient nie napisał: dosłownych cytatów jest "
    f"**{fmt_pct(metrics['llm_quotes_verbatim'])}**. Pozostałe aplikacja pomija."
)

positive = shares.get("positive", 0)
st.info(
    f"**Dlaczego nie wystarczy trafność?** {fmt_pct(positive)} opinii jest pozytywnych. Metoda, która zawsze mówi "
    f"„pozytywna”, ma trafność {fmt_pct(positive)}, ale F1 makro tylko {fmt_num(2 * positive / (1 + positive) / 3)}.",
    icon=":material/info:",
)
