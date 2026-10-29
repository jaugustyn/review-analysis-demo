# Głos Klienta – analiza opinii klientów: od słownika do LLM

Demo narzędzia, które z polskich opinii o produkcie wyciąga zalety i skargi według tematów (jakość, zgodność
z opisem, cena, dostawa, obsługa klienta), z cytatami i krótkim raportem dla menedżera. Pokazuje też, jak cztery
metody – od prostego słownika po model językowy (LLM) – radzą sobie z tymi samymi opiniami i ile kosztuje każda
z nich.

> „Z setek opinii – jedna jasna odpowiedź.”

## Ekrany

- **Produkt** – zalety i skargi jednego z 10 produktów demo z Ceneo.pl według tematów, z cytatami, do wyboru
  ze słownika albo z LLM, oraz raport dla menedżera.
- **Analiza na żywo** – wklejasz link do dowolnego produktu na Ceneo, a aplikacja pobiera jego opinie i liczy
  to samo (tylko przy lokalnym uruchomieniu).
- **Porównanie metod** – trafność, F1 i czas analizy 1000 opinii dla czterech metod, szacunek kosztu
  LLM w płatnych API oraz ręczna ocena tematów ze słownika i z LLM.
- **Opinie** – opinie produktów demo z werdyktem każdej metody; widać, gdzie która metoda się myli.

Jak korzystać z aplikacji, krok po kroku: [docs/tutorial.md](docs/tutorial.md).

Podsumowanie wyników (tabele i wykresy): [results/summary.md](results/summary.md).

## Dane

- **Opinie o produktach: [Ceneo.pl](https://www.ceneo.pl).** Aplikacja pobiera tylko stronę produktu (jedno
  zapytanie) i bierze z niej pierwszą stronę opinii – zwykle 10, w kolejności wybranej przez Ceneo. Kolejne
  strony Ceneo otwiera tylko w przeglądarce, po automatycznym sprawdzeniu, czy to nie bot – tego nie obchodzimy.
  Zapisujemy tylko ocenę, treść i datę opinii, bez nazw autorów.
- **Porównanie metod i nauka TF-IDF: [Allegro Reviews](https://huggingface.co/datasets/allegro/klej-allegro-reviews)**
  z benchmarku [KLEJ](https://klejbenchmark.com) (licencja CC BY-SA 4.0): ok. 11,6 tys. opinii z Allegro.pl
  z oceną 1–5.
- **Słownik wydźwięku: [Słownik Wydźwięku](https://zil.ipipan.waw.pl/SlownikWydzwieku)** A. Wawera (IPI PAN).
  Nie ma podanej licencji, więc nie jest w repozytorium – pobiera go `prepare_data.py`.

W repozytorium są tylko opinie potrzebne aplikacji (folder `precomputed/`).

## Uruchomienie w Dockerze (najprościej)

```bash
docker compose up --build
```

Aplikacja: http://localhost:8502 (także przy uruchomieniu lokalnym). Wyniki są policzone wcześniej,
więc nie trzeba karty graficznej ani LLM.
Analiza na żywo nie działa w Dockerze.

## Uruchomienie lokalne (z analizą na żywo)

Wymagania: Python 3.12, [uv](https://docs.astral.sh/uv/), karta NVIDIA z co najmniej 12 GB pamięci
i sterownikiem dla CUDA 13 oraz [Ollama](https://ollama.com) w wersji 0.40 lub nowszej.

```bash
uv sync
ollama pull gemma4:12b               # model z LLM_MODEL w src/config.py
uv run python prepare_data.py        # pobiera dane i opinie produktów demo z Ceneo
uv run python analyze.py             # liczy 4 metody od nowa (ok. 20 min)
uv run streamlit run app.py          # aplikacja
uv run python report.py              # opcjonalnie: odświeża results/
```

Sama aplikacja działa od razu po `uv sync`, bo wyniki są w repozytorium. `prepare_data.py` i `analyze.py`
są potrzebne do analizy na żywo i do przeliczenia wyników. Produkty demo to lista identyfikatorów z Ceneo
w `src/config.py` (`DEMO_PRODUCT_IDS`) – po jej zmianie uruchom oba skrypty ponownie.

Ollama domyślnie odpowiada na jedno zapytanie naraz. Aplikacja wysyła cztery, więc z ustawioną zmienną
środowiskową `OLLAMA_NUM_PARALLEL=4` (przed startem Ollamy) analiza LLM jest ok. 1,6 raza szybsza.

Windows ze Smart App Control: ustaw `UV_PYTHON_PREFERENCE=only-system`, żeby uv używał Pythona z python.org
zamiast pobierać własny (system blokuje jego niepodpisane pliki).

## Jak to działa

- **Sentyment** w trzech klasach według gwiazdek: poniżej 3★ negatywna, 3★ neutralna, powyżej 3★ pozytywna.
- **Cztery metody** na tych samych opiniach z części testowej Allegro Reviews:
  1. słownik wydźwięku – słowa sprowadzone do formy podstawowej (simplemma), „nie” odwraca ocenę następnego słowa,
  2. TF-IDF + regresja logistyczna uczona na części treningowej Allegro Reviews,
  3. gotowy wielojęzyczny model z Hugging Face uczony na opiniach o produktach,
  4. LLM lokalnie przez Ollamę – zwraca też tematy i cytaty.
- **Tematy** na dwa sposoby: słownik słów kluczowych (zdanie z początkiem słowa z listy, wydźwięk ze słownika
  wydźwięku) albo LLM. Cytat z LLM, którego nie ma dosłownie w opinii, jest pomijany.
- **Raport dla menedżera** pisze LLM na podstawie policzonych tematów i cytatów.
- **Koszt API** to szacunek: liczba tokenów zużytych przez model lokalny razy ceny tanich modeli trzech firm
  (cenniki i data w `src/config.py`).

## Struktura

```
app.py              aplikacja Streamlit: nawigacja
views/              cztery ekrany aplikacji
prepare_data.py     pobranie Allegro Reviews, słownika i opinii produktów demo z Ceneo
analyze.py          cztery metody na opiniach testowych i produktach demo → precomputed/
report.py           podsumowanie wyników: tabele i wykresy (results/)
precomputed/        policzone wyniki, które czyta aplikacja
results/            summary.md, wykresy i próbka do ręcznej oceny tematów
docs/               tutorial ze zrzutami ekranu
src/
  config.py         ścieżki, źródła danych, produkty demo, modele, ceny API
  ceneo.py          pobranie opinii produktu ze strony Ceneo
  data.py           Allegro Reviews i sentyment z gwiazdek
  sentiment.py      słownik wydźwięku, TF-IDF, gotowy model
  topics.py         tematy ze słownika słów kluczowych
  llm.py            LLM: sentyment, tematy i cytaty, raport
  pipeline.py       cztery metody na zbiorze opinii
  live.py           analiza na żywo
  app_data.py       wczytywanie wyników w aplikacji
  product_view.py   widok produktu wspólny dla ekranów Produkt i Analiza na żywo
  formatting.py     polskie etykiety, formatowanie liczb, kolory
```
