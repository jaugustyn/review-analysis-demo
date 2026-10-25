# Wyniki – Głos Klienta

Wygenerowane przez `report.py` (26.10.2026) z wyników `analyze.py` z 2026-10-25.

## Dane

- Opinie: Amazon Reviews 2023 (McAuley Lab, https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023),
  kategoria Appliances (AGD), po angielsku.
- Porównanie metod: 1 000 losowych opinii spoza produktów demo. Rozkład według gwiazdek:
  pozytywne 80,3%, negatywne 15,0%,
  neutralne 4,7%.
- Prawda: ocena klienta – 1–2★ negatywna, 3★ neutralna, 4–5★ pozytywna.
- Produkty demo: 10 popularnych produktów różnych typów, po 200 najnowszych opinii.

## Metody

1. **Słownik (VADER)** – gotowa lista słów z wydźwiękiem; próg ±0,05 dla oceny całej opinii.
2. **Klasyczne ML** – TF-IDF (słowa i pary słów) + regresja logistyczna, uczona na
   100 000 innych opiniach z tej kategorii.
3. **Gotowy model** – `nlptown/bert-base-multilingual-uncased-sentiment` z Hugging Face (uczony na opiniach o produktach,
   wybrany spośród trzech kandydatów).
4. **LLM** – `gemma4:12b` lokalnie przez Ollamę; dla każdej opinii zwraca sentyment, tematy i cytaty.

## Porównanie na 1 000 opiniach (wykresy: `f1_by_class.png`, `quality_vs_time.png`)

| Metoda | Trafność | F1 makro | F1 negatywna | F1 neutralna | F1 pozytywna | Czas na 1000 opinii |
|---|---|---|---|---|---|---|
| Słownik (VADER) | 80,0% | 0,52 | 0,61 | 0,05 | 0,89 | poniżej 1 s |
| Klasyczne ML (TF-IDF) | 88,1% | 0,70 | 0,80 | 0,34 | 0,94 | poniżej 1 s |
| Gotowy model (Hugging Face) | 91,7% | 0,75 | 0,84 | 0,45 | 0,96 | 10 s |
| LLM | 88,3% | 0,70 | 0,88 | 0,27 | 0,94 | 7 min |

Czas zmierzony na NVIDIA GeForce RTX 5070; LLM: 4 zapytania naraz. Wszystkie metody działały
lokalnie, więc ich koszt to 0 zł (poza prądem i sprzętem). Metoda, która zawsze mówi „pozytywna”, miałaby
trafność 80% i F1 makro 0,30 –
dlatego główną miarą jest F1 makro.

## Ile kosztowałby LLM przez API

LLM zużył średnio 290 816 tokenów wejściowych i 60 220 wyjściowych na 1000 opinii
(liczone przez model lokalny; modele w API liczą tokeny trochę inaczej). Ceny standardowe z 25.10.2026,
kurs średni NBP z 23.10.2026 (tabela 207/A/NBP/2026).

| Model | Cena USD za 1 mln tokenów (wejście / wyjście) | Koszt na 1000 opinii | Źródło |
|---|---|---|---|
| Claude Haiku 5.5 (Anthropic) | 0,10 / 0,50 | 0,23 zł | https://platform.claude.com/docs/en/about-claude/pricing |
| GPT-6 Luna (OpenAI) | 0,10 / 0,50 | 0,23 zł | https://developers.openai.com/api/docs/pricing |
| Gemini 3.5 Flash-Lite (Google) | 0,30 / 2,50 | 0,93 zł | https://ai.google.dev/gemini-api/docs/pricing |

## Tematy: słownik czy LLM

Wzmianki o tematach w 1 000 opiniach:

| Temat | Słownik słów kluczowych | LLM |
|---|---|---|
| Jakość | 139 | 301 |
| Dopasowanie | 193 | 304 |
| Cena | 199 | 263 |
| Dostawa | 52 | 116 |
| Obsługa | 43 | 65 |

Cytaty LLM dosłownie obecne w opinii: 98% (pozostałe aplikacja pomija).

Ręczna ocena jeszcze niewypełniona: w `topic_review_sample.csv` dla każdej z 50 opinii wpisz „tak” albo „nie” w kolumnach `poprawne_slownik` i `poprawne_llm` (czy tematy i ich wydźwięk zgadzają się z treścią opinii), potem uruchom `report.py` ponownie.

## Przykłady, w których słownik się myli, a LLM nie

- „Dryer lint catcher. Fast delivery and as described.” (5★) – słownik: neutralna, LLM: pozytywna
- „Five Stars. Quick and just what I needed to get things running again....moto scoot” (5★) – słownik: neutralna, LLM: pozytywna
- „Fixed myself!. Found out the problem, researched the part number and had it working within an hour. For less than the cost of a repair estimate.” (5★) – słownik: negatywna, LLM: pozytywna
- „Cut to size. Wondering where these have been my whole life!!” (5★) – słownik: negatywna, LLM: pozytywna

## Produkty demo

- **Nespresso Capsules Refillable - Reusable Coffee Pods For Nespresso Cups - Origin** – Produkt ma mieszany odbiór, w którym głównym atutem jest oszczędność pieniędzy, natomiast poważnym problemem jest niedopasowanie wielkości kapsułek do maszyn.
- **BLACK+DECKER BCRK32B Compact Refrigerator Energy Star Single Door Mini Fridge wi** – Produkt ma mieszany odbiór, w którym pozytywnie oceniane są wymiary i cena, natomiast poważnym problemem jest niska niezawodność oraz trudności z obsługą klienta.
- **Haier HLP21N Pulsator 1-Cubic-Foot Portable Washer** – Produkt cieszy się ogólnie pozytywnym odbiorem, głównie dzięki dobremu stosunkowi jakości do ceny oraz dopasowaniu do małych przestrzeni, choć klienci zgłaszają problemy z trwałością i obsługą serwisową.
- **BLACK+DECKER BCD6W Compact Countertop Dishwasher, 6 Place Settings, White** – Produkt ma mieszany odbiór, przy czym klienci doceniają jego wymiary i stosunek jakości do ceny, ale zgłaszają poważne problemy z trwałością oraz obsługą posprzedową.
- **SAMSUNG Genuine Filters for Refrigerator Water and Ice, Carbon Block Filtration ** – Produkt ma mieszany odbiór, w którym pozytywnie oceniana jest cena i dopasowanie, jednak istotnym problemem jest jakość wykonania oraz trudności z procesem zwrotów.
- **BOSCH / Cuno 9000 077104 UltraClarity REPLFLTR10 Refrigerator Water Filter** – Produkt ma mieszany odbiór, w którym pozytywnie oceniane są głównie dopasowanie i cena, natomiast istotnym problemem jest niska jakość niektórych egzemplarzy oraz trudności z procesem zwrotów.
- **LG 4681EA2001T Genuine OEM Drain Pump Assembly for LG Washing Machines** – Produkt cieszy się ogólnie pozytywnym odbiorem, głównie dzięki wysokiej kompatybilności z urządzeniami LG oraz korzystnej cenie.
- **De'Longhi Stainless Steel Milk Frothing Pitcher, 12 ounce (350 ml), Barista Tool** – Produkt cieszy się ogólnie pozytywnym odbiorem, głównie dzięki wysokiej jakości wykonania i dopasowaniu do konkretnych modeli ekspresów.
- **KRUPS F0494210 Gold Tone Permanent Filter, Black** – Produkt cieszy się ogólnie pozytywnym odbiorem, głównie dzięki dobremu dopasowaniu do ekspresów oraz korzystnej cenie.
- **Melitta Cone Coffee Filters, Natural Brown #4, 100 Count (Pack Of 3)** – Produkt cieszy się ogólnie pozytywnym odbiorem, głównie dzięki wysokiej jakości wykonania i korzystnej cenie.

## Zastrzeżenia

- Opinii neutralnych jest mało (4,7%), więc wyniki dla tej klasy są niepewne.
- Ocena w gwiazdkach nie zawsze zgadza się z treścią opinii, więc część „błędów” metod to w rzeczywistości
  niespójność klienta.
- Gotowy model z Hugging Face był uczony na opiniach o produktach – nie wiemy, czy nie było wśród nich opinii z Amazona.
