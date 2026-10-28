"""Writes a summary of the results to results/: Markdown tables, two charts and a sample for the manual
check of topics. The sample is created once; when its "poprawne_*" columns are filled in, the summary
reports the share of correct topics.

Run after analyze.py: uv run python report.py
"""

import json
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    API_PRICES_CHECKED, DEMO_RESULTS, EVAL_RESULTS, METRICS_PATH, PRODUCTS_RESULTS, RESULTS_DIR, SEED, USD_PLN_SOURCE,
)
from src.formatting import (
    INK, METHOD_LABELS, NEGATIVE, NEUTRAL, POSITIVE, SENTIMENT_LABELS, SOURCE_LABELS, TEAL, fmt_int, fmt_num,
    fmt_pct, fmt_pln, fmt_seconds,
)
from src.llm import api_costs
from src.topics import TOPIC_LABELS

TOPIC_SAMPLE = RESULTS_DIR / "topic_review_sample.csv"
SAMPLE_SIZE = 50


def md_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C3C2B7")
    ax.tick_params(colors=INK)
    ax.grid(color="#E1E0D9", linewidth=0.8)
    ax.set_axisbelow(True)


def quality_chart(methods: dict) -> None:
    """F1 per class for each method: everyone struggles with neutral reviews."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    names = [METHOD_LABELS[m] for m in methods][::-1]
    y = np.arange(len(names))
    for i, (cls, color) in enumerate(zip(SENTIMENT_LABELS, [NEGATIVE, NEUTRAL, POSITIVE])):
        values = [methods[m]["f1"][cls] for m in methods][::-1]
        bars = ax.barh(y + (1 - i) * 0.25, values, height=0.23, color=color, label=SENTIMENT_LABELS[cls])
        ax.bar_label(bars, labels=[fmt_num(v) for v in values], padding=3, fontsize=8, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(lambda v, _: fmt_num(v, 1))
    ax.set_xlabel("F1 (1 = bez błędów)", color=INK)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3)
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "f1_by_class.png")
    plt.close(fig)


def speed_chart(methods: dict) -> None:
    """Quality against time: one point per method, time on a log scale."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    for method, values in methods.items():
        seconds = max(values["seconds_per_1000"], 0.01)
        ax.scatter([seconds], [values["f1_macro"]], s=70, color=TEAL if method == "llm" else INK, zorder=3)
        ax.annotate(METHOD_LABELS[method], (seconds, values["f1_macro"]), xytext=(8, -4),
                    textcoords="offset points", color=INK)
    ax.set_xscale("log")
    ax.set_xlim(right=max(v["seconds_per_1000"] for v in methods.values()) * 6)  # room for the last label
    ax.set_xlabel("Czas analizy 1000 opinii (s, skala logarytmiczna)", color=INK)
    ax.set_ylabel("F1 makro", color=INK)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: fmt_num(v, 0 if v >= 1 else 2))
    ax.yaxis.set_major_formatter(lambda v, _: fmt_num(v, 1))
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "quality_vs_time.png")
    plt.close(fig)


def topics_text(raw: str) -> str:
    items = json.loads(raw)
    return "; ".join(f"{TOPIC_LABELS[i['topic']]} {'+' if i['polarity'] == 'positive' else '−'}" for i in items) or "-"


def topic_check(evaluated: pd.DataFrame) -> str:
    """Creates the sample for the manual check, or reports the check when it has been filled in."""
    if not TOPIC_SAMPLE.exists():
        sample = evaluated.sample(SAMPLE_SIZE, random_state=SEED)
        pd.DataFrame({
            "opinia": sample["text"],
            "tematy_slownik": sample["topics_keywords"].map(topics_text),
            "tematy_llm": sample["topics_llm"].map(topics_text),
            "poprawne_slownik": "",
            "poprawne_llm": "",
        }).to_csv(TOPIC_SAMPLE, index=False, encoding="utf-8-sig")
    checked = pd.read_csv(TOPIC_SAMPLE, encoding="utf-8-sig", dtype=str).fillna("")
    answers = {source: checked[f"poprawne_{source}"].str.strip().str.lower() for source in ("slownik", "llm")}
    if not all(answers[source].isin(["tak", "nie"]).all() for source in answers):
        return (f"Ręczna ocena jeszcze niewypełniona: w `{TOPIC_SAMPLE.name}` dla każdej z {SAMPLE_SIZE} opinii wpisz "
                "„tak” albo „nie” w kolumnach `poprawne_slownik` i `poprawne_llm` (czy tematy i ich wydźwięk "
                "zgadzają się z treścią opinii), potem uruchom `report.py` ponownie.")
    return (f"Ręczna ocena {SAMPLE_SIZE} losowych opinii: tematy poprawne w "
            f"{fmt_pct((answers['slownik'] == 'tak').mean())} opinii dla słownika i "
            f"{fmt_pct((answers['llm'] == 'tak').mean())} dla LLM.")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    products = json.loads(PRODUCTS_RESULTS.read_text(encoding="utf-8"))
    evaluated = pd.read_parquet(EVAL_RESULTS)
    methods = metrics["methods"]
    quality_chart(methods)
    speed_chart(methods)

    comparison = pd.DataFrame({
        "Metoda": [METHOD_LABELS[m] for m in methods],
        "Trafność": [fmt_pct(v["accuracy"], 1) for v in methods.values()],
        "F1 makro": [fmt_num(v["f1_macro"]) for v in methods.values()],
        **{f"F1 {label}": [fmt_num(v["f1"][cls]) for v in methods.values()] for cls, label in SENTIMENT_LABELS.items()},
        "Czas na 1000 opinii": [fmt_seconds(v["seconds_per_1000"]) for v in methods.values()],
    })
    tokens = metrics["llm_tokens_per_1000"]
    api = pd.DataFrame([{
        "Model": c["model"],
        "Cena USD za 1 mln tokenów (wejście / wyjście)": f"{fmt_num(c['input'])} / {fmt_num(c['output'])}",
        "Koszt na 1000 opinii": fmt_pln(c["pln"]),
        "Źródło": c["source"],
    } for c in api_costs(tokens)])
    mentions = pd.DataFrame([
        {"Temat": TOPIC_LABELS[item["topic"]], "Źródło": SOURCE_LABELS[source]}
        for source in SOURCE_LABELS for raw in evaluated[f"topics_{source}"] for item in json.loads(raw)
    ]).value_counts().unstack(fill_value=0).reindex(list(TOPIC_LABELS.values())).reset_index()
    misses = evaluated[(evaluated["sent_lexicon"] != evaluated["sentiment"]) & (evaluated["sent_llm"] == evaluated["sentiment"])]
    examples = misses[misses["text"].str.len().between(5, 300)].head(4)
    demo = pd.read_parquet(DEMO_RESULTS)
    shares = metrics["class_shares"]

    report = f"""# Wyniki – Głos Klienta

Wygenerowane przez `report.py` ({date.today():%d.%m.%Y}) z wyników `analyze.py` z {metrics['date']}.

## Dane

- Porównanie metod: część testowa zbioru Allegro Reviews (benchmark KLEJ, licencja CC BY-SA 4.0,
  https://huggingface.co/datasets/allegro/klej-allegro-reviews) – {fmt_int(metrics['eval_reviews'])} polskich opinii
  z Allegro.pl z oceną 1–5. Rozkład: pozytywne {fmt_pct(shares.get('positive', 0), 1)},
  negatywne {fmt_pct(shares.get('negative', 0), 1)}, neutralne {fmt_pct(shares.get('neutral', 0), 1)}.
- Prawda: ocena klienta – poniżej 3★ negatywna, 3★ neutralna, powyżej 3★ pozytywna.
- Produkty demo: {len(products)} produktów z Ceneo.pl, po pierwszej stronie opinii (zwykle 10, w kolejności
  wybranej przez Ceneo; kolejnych stron Ceneo nie udostępnia bez weryfikacji, że jest się człowiekiem).
  Razem {fmt_int(len(demo))} opinii, pozytywne {fmt_pct((demo['sentiment'] == 'positive').mean())}.

## Metody

1. **Słownik wydźwięku** – polski słownik A. Wawera (IPI PAN, https://zil.ipipan.waw.pl/SlownikWydzwieku,
   3 703 słowa z oceną od −2 do 2). Słowa z opinii sprowadzamy do formy podstawowej (biblioteka simplemma),
   „nie” odwraca ocenę następnego słowa; suma ocen powyżej zera to opinia pozytywna, poniżej – negatywna.
2. **Klasyczne ML** – TF-IDF (słowa i pary słów) + regresja logistyczna, uczona na
   {fmt_int(metrics['tfidf_train_reviews'])} opiniach z części treningowej Allegro Reviews.
3. **Gotowy model** – `{metrics['transformer_model']}` z Hugging Face (wielojęzyczny, uczony na opiniach
   o produktach). Polski model Voicelab HerBERT miał na części walidacyjnej Allegro Reviews 96% trafności, ale był najpewniej uczony
   na Allegro Reviews, więc nie dałoby się go uczciwie porównać.
4. **LLM** – `{metrics['llm_model']}` lokalnie przez Ollamę; dla każdej opinii zwraca sentyment, tematy i cytaty.

## Porównanie na {fmt_int(metrics['eval_reviews'])} opiniach (wykresy: `f1_by_class.png`, `quality_vs_time.png`)

{md_table(comparison)}

Czas zmierzony na {metrics['gpu']}; LLM: {metrics['llm_workers']} zapytania naraz. Wszystkie metody działały
lokalnie, więc ich koszt to 0 zł (poza prądem i sprzętem). Metoda, która zawsze mówi „pozytywna”, miałaby
trafność {fmt_pct(shares.get('positive', 0))} i F1 makro {fmt_num(2 * shares.get('positive', 0) / (1 + shares.get('positive', 0)) / 3)} –
dlatego główną miarą jest F1 makro.

## Ile kosztowałby LLM przez API

LLM zużył średnio {fmt_int(tokens['input'])} tokenów wejściowych i {fmt_int(tokens['output'])} wyjściowych na 1000 opinii
(liczone przez model lokalny; modele w API liczą tokeny trochę inaczej). Ceny standardowe z {API_PRICES_CHECKED},
{USD_PLN_SOURCE}.

{md_table(api)}

## Tematy: słownik czy LLM

Wzmianki o tematach w {fmt_int(metrics['eval_reviews'])} opiniach:

{md_table(mentions)}

Cytaty LLM dosłownie obecne w opinii: {fmt_pct(metrics['llm_quotes_verbatim'])} (pozostałe aplikacja pomija).

{topic_check(evaluated)}

## Przykłady, w których słownik się myli, a LLM nie

{chr(10).join(f'- „{row.text}” ({row.rating:g}★) – słownik: {SENTIMENT_LABELS[row.sent_lexicon]}, LLM: {SENTIMENT_LABELS[row.sent_llm]}' for row in examples.itertuples())}

## Produkty demo

{chr(10).join(f'- **{p["title"][:80]}** ({p["ceneo_reviews"]} opinii na Ceneo) – {p["report"]["summary"]}' for p in products.values())}

## Zastrzeżenia

- Produkty demo mają tylko po ok. 10 opinii z pierwszej strony Ceneo, w większości pozytywnych – liczby
  w tematach są małe, a raport opiera się na kilku opiniach.
- Ocena w gwiazdkach nie zawsze zgadza się z treścią opinii, więc część „błędów” metod to w rzeczywistości
  niespójność klienta.
- Słownik wydźwięku nie ma podanej licencji, dlatego pobieramy go przy przygotowaniu danych i nie trzymamy w repozytorium.
"""
    (RESULTS_DIR / "summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
