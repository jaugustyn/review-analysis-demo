# Głos Klienta – analiza opinii klientów: od słownika do LLM

Demo narzędzia, które z opinii o produkcie wyciąga najczęstsze zalety i skargi według tematów (jakość, cena,
dostawa, obsługa), z cytatami i krótkim raportem dla menedżera. Pokazuje też, jak cztery metody – od prostego
słownika po model językowy (LLM) – radzą sobie z tymi samymi opiniami i ile kosztuje każda z nich.

> „Z setek opinii – jedna jasna odpowiedź.”

## Ekrany

- **Produkt** – zalety i skargi wybranego produktu według tematów, z cytatami, do wyboru ze słownika albo z LLM,
  oraz raport dla menedżera. Przy lokalnym uruchomieniu można też wkleić link do produktu AGD z Amazona
  i przeanalizować go na żywo.
- **Porównanie metod** – trafność, F1, czas i koszt analizy 1000 opinii dla czterech metod oraz szacunek kosztu
  LLM w płatnych API.
- **Opinie** – opinie testowe z werdyktem każdej metody; widać, gdzie która się myli.

Podsumowanie wyników (tabele i wykresy): [results/summary.md](results/summary.md).

## Dane

Opinie z kategorii Appliances (AGD) zbioru [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
(McAuley Lab): Hou i in., *Bridging Language and Items for Retrieval and Recommendation*, 2024,
[arXiv:2403.03952](https://arxiv.org/abs/2403.03952). Opinie są po angielsku, interfejs po polsku.

W repozytorium są tylko opinie potrzebne aplikacji (produkty demo i opinie testowe, folder `precomputed/`).
Pełny zbiór (ok. 1,2 GB) pobiera `prepare_data.py`.

## Uruchomienie w Dockerze (najprościej)

```bash
docker compose up --build
```

Aplikacja: http://localhost:8502 (także przy uruchomieniu lokalnym). Wyniki są policzone wcześniej,
więc nie trzeba karty graficznej ani LLM.
Analiza na żywo nie działa w Dockerze.

## Uruchomienie lokalne (z analizą na żywo)

Wymagania: Python 3.12, [uv](https://docs.astral.sh/uv/), karta NVIDIA ze sterownikiem dla CUDA 13
i [Ollama](https://ollama.com) z modelem z `src/config.py` (`LLM_MODEL`).

```bash
uv sync
ollama pull qwen3.5:9b               # model z LLM_MODEL w src/config.py
uv run python prepare_data.py        # pobiera dane (ok. 1,2 GB) i wybiera opinie
uv run python analyze.py             # liczy 4 metody od nowa (ok. godziny)
uv run streamlit run app.py          # aplikacja
uv run python report.py              # opcjonalnie: odświeża results/
```

Sama aplikacja działa od razu po `uv sync`, bo wyniki są w repozytorium. `prepare_data.py` i `analyze.py`
są potrzebne do analizy na żywo i do przeliczenia wyników.

Windows ze Smart App Control: ustaw `UV_PYTHON_PREFERENCE=only-system`, żeby uv używał Pythona z python.org
zamiast pobierać własny (system blokuje jego niepodpisane pliki).

## Jak to działa

- **Sentyment** w trzech klasach według gwiazdek: 1–2★ negatywna, 3★ neutralna, 4–5★ pozytywna.
- **Cztery metody** na tych samych 1000 opiniach testowych (spoza produktów demo):
  1. słownik VADER,
  2. TF-IDF + regresja logistyczna uczona na 100 000 innych opinii,
  3. gotowy model z Hugging Face uczony na opiniach o produktach,
  4. LLM lokalnie przez Ollamę – zwraca też tematy i cytaty.
- **Tematy** na dwa sposoby: słownik słów kluczowych (zdanie ze słowem z listy, wydźwięk z VADER) albo LLM.
  Cytat z LLM, którego nie ma dosłownie w opinii, jest pomijany.
- **Raport dla menedżera** pisze LLM na podstawie policzonych tematów i cytatów.
- **Koszt API** to szacunek: liczba tokenów zużytych przez model lokalny razy ceny tanich modeli trzech firm
  (cenniki i data w `src/config.py`).
- **Analiza na żywo** bierze 100 najnowszych opinii produktu z pobranego zbioru, nie z Amazona.

## Struktura

```
app.py              aplikacja Streamlit: nawigacja
views/              trzy ekrany aplikacji
prepare_data.py     pobranie danych, produkty demo, opinie testowe i do nauki
analyze.py          cztery metody na opiniach testowych i produktach demo → precomputed/
report.py           podsumowanie wyników: tabele i wykresy (results/)
precomputed/        policzone wyniki, które czyta aplikacja
results/            summary.md, wykresy i próbka do ręcznej oceny tematów
src/
  config.py         ścieżki, produkty demo, modele, ceny API
  data.py           dane: konwersja, podział, wyszukiwanie produktu po linku
  sentiment.py      słownik, TF-IDF, gotowy model
  topics.py         tematy ze słownika słów kluczowych
  llm.py            LLM: sentyment, tematy i cytaty, raport
  pipeline.py       cztery metody na zbiorze opinii
  live.py           analiza na żywo
  app_data.py       wczytywanie wyników w aplikacji
  formatting.py     polskie etykiety, formatowanie liczb, kolory
```
