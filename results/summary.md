# Wyniki – Głos Klienta

Wygenerowane przez `report.py` (29.10.2026) z wyników `analyze.py` z 2026-10-28.

## Dane

- Porównanie metod: część testowa zbioru Allegro Reviews (benchmark KLEJ, licencja CC BY-SA 4.0,
  https://huggingface.co/datasets/allegro/klej-allegro-reviews) – 1 006 polskich opinii
  z Allegro.pl z oceną 1–5. Rozkład: pozytywne 53,1%,
  negatywne 32,6%, neutralne 14,3%.
- Prawda: ocena klienta – poniżej 3★ negatywna, 3★ neutralna, powyżej 3★ pozytywna.
- Produkty demo: 10 produktów z Ceneo.pl, po pierwszej stronie opinii (zwykle 10, w kolejności
  wybranej przez Ceneo; kolejne strony Ceneo otwiera tylko w przeglądarce, po automatycznym sprawdzeniu,
  czy to nie bot).
  Razem 100 opinii, pozytywne 94%.

## Metody

1. **Słownik wydźwięku** – polski słownik A. Wawera (IPI PAN, https://zil.ipipan.waw.pl/SlownikWydzwieku,
   3 703 słowa z oceną od −2 do 2). Słowa z opinii sprowadzamy do formy podstawowej (biblioteka simplemma),
   „nie” odwraca ocenę następnego słowa; suma ocen powyżej zera to opinia pozytywna, poniżej – negatywna.
2. **Klasyczne ML** – TF-IDF (słowa i pary słów) + regresja logistyczna, uczona na
   9 577 opiniach z części treningowej Allegro Reviews.
3. **Gotowy model** – `nlptown/bert-base-multilingual-uncased-sentiment` z Hugging Face (wielojęzyczny, uczony na opiniach
   o produktach). Polski model Voicelab HerBERT miał na części walidacyjnej Allegro Reviews 96% trafności, ale był najpewniej uczony
   na Allegro Reviews, więc nie dałoby się go uczciwie porównać.
4. **LLM** – `gemma4:12b` lokalnie przez Ollamę; dla każdej opinii zwraca sentyment, tematy i cytaty.

## Porównanie na 1 006 opiniach (wykresy: `f1_by_class.png`, `quality_vs_time.png`)

| Metoda | Trafność | F1 makro | F1 negatywna | F1 neutralna | F1 pozytywna | Czas na 1000 opinii |
|---|---|---|---|---|---|---|
| Słownik wydźwięku | 49,8% | 0,37 | 0,41 | 0,08 | 0,63 | 1 s |
| Klasyczne ML (TF-IDF) | 73,2% | 0,63 | 0,76 | 0,29 | 0,84 | poniżej 1 s |
| Gotowy model (Hugging Face) | 67,4% | 0,59 | 0,74 | 0,29 | 0,74 | 9 s |
| LLM | 72,2% | 0,65 | 0,84 | 0,31 | 0,79 | 18 min |

Czas zmierzony na NVIDIA GeForce RTX 5070. Wszystkie metody działały lokalnie, więc ich koszt to 0 zł
(poza prądem i sprzętem). Metoda, która zawsze mówi „pozytywna”, miałaby trafność 53% i F1 makro 0,23 –
dlatego główną miarą jest F1 makro.

## Ile kosztowałby LLM przez API

LLM zużył średnio 415 726 tokenów wejściowych i 99 843 wyjściowych na 1000 opinii
(liczone przez model lokalny; modele w API liczą tokeny trochę inaczej). Ceny standardowe z 25.10.2026,
kurs średni NBP z 23.10.2026 (tabela 207/A/NBP/2026).

| Model | Cena USD za 1 mln tokenów (wejście / wyjście) | Koszt na 1000 opinii | Źródło |
|---|---|---|---|
| Claude Haiku 5.5 (Anthropic) | 0,10 / 0,50 | 0,36 zł | https://platform.claude.com/docs/en/about-claude/pricing |
| GPT-6 Luna (OpenAI) | 0,10 / 0,50 | 0,36 zł | https://developers.openai.com/api/docs/pricing |
| Gemini 3.5 Flash-Lite (Google) | 0,30 / 2,50 | 1,47 zł | https://ai.google.dev/gemini-api/docs/pricing |

## Tematy: słownik czy LLM

Wzmianki o tematach w 1 006 opiniach:

| Temat | Słownik słów kluczowych | LLM |
|---|---|---|
| Jakość | 328 | 667 |
| Zgodność z opisem | 218 | 361 |
| Cena | 221 | 326 |
| Dostawa | 42 | 54 |
| Obsługa klienta | 59 | 75 |

Cytaty LLM dosłownie obecne w opinii: 94% (pozostałe aplikacja pomija).

Ręczna ocena 50 losowych opinii: tematy poprawne w 30% opinii dla słownika i 70% dla LLM.

## Przykłady, w których słownik się myli, a LLM nie

- „klawiatura wbrew opisowi wcale nie pasuje do Lenovo G505. Nie wiem jak w innych modelach, ale w G505 po zamontowaniu zostają szpary! I to na tyle duże że nie tylko kurz może przedostać się do wnętrza układu. Miałem używaną do tego laptopa i tych szpary nie było, więc nie są one normą!” (1★) – słownik: pozytywna, LLM: negatywna
- „dobry suwaczek, praktyczny” (4★) – słownik: neutralna, LLM: pozytywna
- „Spacja działa  tylko w gdy naciśnie się  ją  po środku. Przyciski nie   zawsze  dobrze   kontaktują   .  Pisze  tą klawiaturą  nie   wiem czy  zostanie  to  wyczyszczone z formularza    spacje, ale klawiatura sama z siebie robi 2-3 spacje  zamiast jednej. Do komputera nie polecam!” (2★) – słownik: pozytywna, LLM: negatywna
- „Jakość jest tak beznadziejna że brak słów. Już szkło za 5 zł z innej aukcji jest lepsze niż to. Po 10 godzinach od nałożenia całkiem pokruszone po bokach, całe porysowane i to głęboko, niewiadomo z jakiego powodu gdyż telefon przez ten cały czas był w kieszeni, bez żadnych kluczy razem itp.” (1★) – słownik: neutralna, LLM: negatywna

## Produkty demo

- **Lodówka Gorenje NRK620EABXL4 z zamrażalnikiem dolnym 200 cm Czarna** (34 opinii na Ceneo) – Produkt ma mieszany odbiór, w którym pozytywne opinie dotyczące wyglądu i wymiarów są przyćmione przez liczne problemy z jakością wykonania oraz serwisem.
- **Ekspres De'Longhi Dedica Style EC685.BK** (203 opinii na Ceneo) – Produkt cieszy się ogólnie pozytywnym odbiorem, głównie dzięki wysokiej jakości wykonania.
- **Pralka Amica WA0S610DO** (14 opinii na Ceneo) – Produkt jest oceniany pozytywnie przez większość klientów, którzy doceniają jego jakość, cenę oraz zgodność z opisem.
- **Jigoo C200** (245 opinii na Ceneo) – Produkt jest oceniany bardzo pozytywnie, głównie ze względu na korzystny stosunek jakości do ceny.
- **PHILIPS AirFryer Seria 2000 Ovi NA231/00** (127 opinii na Ceneo) – Produkt cieszy się ogólnie pozytywnym odbiorem, szczególnie w zakresie zgodności z opisem oraz stosunku jakości do ceny.
- **Czajnik elektryczny Bosch TWK8611P** (200 opinii na Ceneo) – Produkt jest oceniany pozytywnie głównie ze względu na solidne wykonanie, jednak klienci zgłaszają problemy z trwałością mechanizmu pokrywki.
- **Whirlpool WFC 3C26 PF X** (19 opinii na Ceneo) – Produkt ma mieszany odbiór, przy czym jakość wykonania jest najczęściej krytykowanym aspektem.
- **Samsung DW60M6050FS** (22 opinii na Ceneo) – Produkt ma mieszany odbiór, w którym pozytywy dotyczące designu i ceny są przyćmione przez wyraźne zastrzeżenia do jakości wykonania.
- **Lodówka Bosch Serie 2 KGN33NWEB z zamrażalnikiem dolnym 176 cm Biała** (46 opinii na Ceneo) – Produkt jest oceniany pozytywnie, głównie dzięki wysokiej jakości wykonania i zgodności z opisem.
- **Xiaomi Water Ionic Hair Dryer H500** (108 opinii na Ceneo) – Produkt jest oceniany pozytywnie, głównie ze względu na wysoką jakość wykonania i korzystny stosunek jakości do ceny.

## Zastrzeżenia

- Produkty demo mają tylko po ok. 10 opinii z pierwszej strony Ceneo, w większości pozytywnych – liczby
  w tematach są małe, a raport opiera się na kilku opiniach.
- Ocena w gwiazdkach nie zawsze zgadza się z treścią opinii, więc część „błędów” metod to w rzeczywistości
  niespójność klienta.
- Słownik wydźwięku nie ma podanej licencji, dlatego pobieramy go przy przygotowaniu danych i nie trzymamy w repozytorium.
