# Wyniki – Głos Klienta

Wygenerowane przez `report.py` (25.10.2026) z wyników `analyze.py` z 2026-10-25.

## Dane

- Opinie: Amazon Reviews 2023 (McAuley Lab, https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023),
  kategoria Appliances (AGD), po angielsku.
- Porównanie metod: 1 000 losowych opinii spoza produktów demo. Rozkład według gwiazdek:
  pozytywne 81,1%, negatywne 15,4%,
  neutralne 3,5%.
- Prawda: ocena klienta – 1–2★ negatywna, 3★ neutralna, 4–5★ pozytywna.
- Produkty demo: 10 popularnych produktów różnych typów, po 200 najnowszych opinii.

## Metody

1. **Słownik (VADER)** – gotowa lista słów z wydźwiękiem; próg ±0,05 dla oceny całej opinii.
2. **Klasyczne ML** – TF-IDF (słowa i pary słów) + regresja logistyczna, uczona na
   100 000 innych opiniach z tej kategorii.
3. **Gotowy model** – `nlptown/bert-base-multilingual-uncased-sentiment` z Hugging Face (uczony na opiniach o produktach,
   wybrany spośród trzech kandydatów).
4. **LLM** – `qwen3.5:9b` lokalnie przez Ollamę; dla każdej opinii zwraca sentyment, tematy i cytaty.

## Porównanie na 1 000 opiniach (wykresy: `f1_by_class.png`, `quality_vs_time.png`)

| Metoda | Trafność | F1 makro | F1 negatywna | F1 neutralna | F1 pozytywna | Czas na 1000 opinii |
|---|---|---|---|---|---|---|
| Słownik (VADER) | 80,5% | 0,53 | 0,64 | 0,05 | 0,89 | poniżej 1 s |
| Klasyczne ML (TF-IDF) | 90,1% | 0,70 | 0,83 | 0,31 | 0,96 | poniżej 1 s |
| Gotowy model (Hugging Face) | 92,2% | 0,76 | 0,83 | 0,47 | 0,96 | 8 s |
| LLM | 87,1% | 0,65 | 0,85 | 0,18 | 0,93 | 18 min |

Czas zmierzony na NVIDIA GeForce RTX 5070; LLM: 4 zapytania naraz. Wszystkie metody działały
lokalnie, więc ich koszt to 0 zł (poza prądem i sprzętem). Metoda, która zawsze mówi „pozytywna”, miałaby
trafność 81% i F1 makro 0,30 –
dlatego główną miarą jest F1 makro.

## Ile kosztowałby LLM przez API

LLM zużył średnio 208 325 tokenów wejściowych i 86 336 wyjściowych na 1000 opinii
(liczone przez model lokalny; modele w API liczą tokeny trochę inaczej). Ceny standardowe z 25.10.2026,
kurs średni NBP z 23.10.2026 (tabela 207/A/NBP/2026).

| Model | Cena USD za 1 mln tokenów (wejście / wyjście) | Koszt na 1000 opinii | Źródło |
|---|---|---|---|
| Claude Haiku 5.5 (Anthropic) | 0,10 / 0,50 | 0,25 zł | https://platform.claude.com/docs/en/about-claude/pricing |
| GPT-6 Luna (OpenAI) | 0,10 / 0,50 | 0,25 zł | https://developers.openai.com/api/docs/pricing |
| Gemini 3.5 Flash-Lite (Google) | 0,30 / 2,50 | 1,09 zł | https://ai.google.dev/gemini-api/docs/pricing |

## Tematy: słownik czy LLM

Wzmianki o tematach w 1 000 opiniach:

| Temat | Słownik słów kluczowych | LLM |
|---|---|---|
| Jakość | 131 | 902 |
| Cena | 216 | 268 |
| Dostawa | 68 | 126 |
| Obsługa | 34 | 93 |

Cytaty LLM dosłownie obecne w opinii: 96% (pozostałe aplikacja pomija).

Ręczna ocena jeszcze niewypełniona: w `topic_review_sample.csv` dla każdej z 50 opinii wpisz „tak” albo „nie” w kolumnach `poprawne_slownik` i `poprawne_llm` (czy tematy i ich wydźwięk zgadzają się z treścią opinii), potem uruchom `report.py` ponownie.

## Przykłady, w których słownik się myli, a LLM nie

- „Fixed. I had a leak in my refrigerator water valve and bought this one to replace. It was easy to install and I have had no issues since...” (5★) – słownik: negatywna, LLM: pozytywna
- „Just What Was Needed. The icemaker quit working and replacing the filters fixed the problem. Quick delivery, too.” (5★) – słownik: negatywna, LLM: pozytywna
- „Junk. Pulley hub dissolved into metal shavings in 9 months” (1★) – słownik: neutralna, LLM: negatywna
- „Didn’t fit. Did not fit my fridge regardless of it saying it did” (1★) – słownik: pozytywna, LLM: negatywna

## Produkty demo

- **Cuisinart SS-10P1 Premium 72-Ounce Single-Serve Coffemaker, Stainless Steel & Cu** – Ogólny odbiór produktu jest podzielony: klienci chwalą jego wydajność i wygląd, ale krytykują niezadowalającą trwałość oraz problemy z obsługą klienta.
- **Giantex Portable Mini Compact Twin Tub Washing Machine 20lbs Washer Spain Spinne** – Ogólny odbiór produktu jest mieszany, gdzie klienci doceniają niską cenę i funkcjonalność urządzenia, ale narzekają na problemy z jakością towaru oraz kosztowne i trudne warunki zwrotów.
- **GE Profile Opal | Countertop Nugget Ice Maker with Side Tank | Portable Ice Mach** – Ogólny odbiór produktu jest podzielony: klienci chwalą jego cichą pracę i szybkość produkcji lodu, jednak większość negatywnych opinii dotyczy nagłych awarii po krótkim czasie użytkowania oraz trudności z uzyskaniem wsparcia serwisowego.
- **AGLUCKY Countertop Ice Maker Machine, Portable Ice Makers Countertop, Make 26 lb** – Producent AGLUCKY oferuje maszynę do lodu o wysokiej wydajności i atrakcyjnej cenie, jednak jej długoterminowa niezawodność oraz obsługa klienta wymagają pilnych popraw.
- **Broan-NuTone 414204 Non-Ducted Under-Cabinet Ductless Range Hood Insert, 42-Inch** – Ogólny odbiór produktu Broan-NuTone 414204 jest mieszany, gdzie klienci doceniają łatwą instalację i stosunek ceny do jakości, ale narzekają na problemy z korozją, uszkodzeniami przy dostawie oraz brakujące elementy montażowe.
- **COSMO 5U30 30 in. Under Cabinet Range Hood with Ducted/Ductless Convertible (Kit** – Ogólny odbiór produktu COSMO 5U30 jest pozytywny ze względu na nowoczesny wygląd i stosunek jakości do ceny, jednak klienci zgłaszają istotne zastrzeżenia dotyczące hałasu pracy, braku niektórych akcesoriów w zestawie oraz uszkodzeń powstałych podczas transportu.
- **12 Pack Keurig Filter Replacement by K&J - Compatible with Keurig Coffee Machine** – Ogólny odbiór produktu jest bardzo pozytywny, przy czym większość klientów chwali jego skuteczność i stosunek jakości do ceny.
- **SAMSUNG Genuine Filters for Refrigerator Water and Ice, Carbon Block Filtration ** – Ogólna ocena produktu jest mieszana, z wyraźną przewagą pozytywnych opinii dotyczących jakości filtracji i dopasowania, ale znacznym odsetkiem negatywnych komentarzy związanych z kompatybilnością modeli oraz problemami logistycznymi.
- **Linda's Essentials Silicone Stove Gap Covers (2 Pack), Heat Resistant Oven Gap F** – Produkt jest generalnie dobrze oceniany ze względu na skuteczność w wypełnianiu szczelin i łatwą instalację, jednak klienci zgłaszają obawy dotyczące estetyki, trwałości oraz uszkodzeń podczas transportu.
- **Cimkiz Dishwasher Magnet Clean Dirty Sign Shutter Only Changes When You Push It ** – Ogólny odbiór produktu jest pozytywny, z przewagą opinii chwalących jego solidną konstrukcję i skuteczność w oznaczaniu brudu, choć występują pojedyncze przypadki awarii magnetycznej.

## Zastrzeżenia

- Opinii neutralnych jest mało (3,5%), więc wyniki dla tej klasy są niepewne.
- Ocena w gwiazdkach nie zawsze zgadza się z treścią opinii, więc część „błędów” metod to w rzeczywistości
  niespójność klienta.
- Gotowy model z Hugging Face był uczony na opiniach o produktach – nie wiemy, czy nie było wśród nich opinii z Amazona.
- Model LLM może jeszcze zostać zmieniony; wtedy trzeba ponownie uruchomić `analyze.py` i `report.py`.
