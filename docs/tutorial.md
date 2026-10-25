# Jak korzystać z Głosu Klienta

Głos Klienta czyta opinie o produkcie i pokazuje, co klienci chwalą, a na co narzekają, w pięciu tematach:
**jakość**, **dopasowanie**, **cena**, **dostawa** i **obsługa**. Do każdego tematu podaje cytaty z opinii, a model
językowy (LLM) pisze krótki raport dla menedżera. Aplikacja porównuje też cztery metody analizy, od prostego
słownika po LLM.

Opinie pochodzą ze zbioru Amazon Reviews 2023 (kategoria AGD) i są po angielsku, interfejs jest po polsku.
Jak uruchomić aplikację, opisuje [README](../README.md). Liczby w tym tutorialu dotyczą wyników zapisanych
w repozytorium (LLM: `gemma4:12b`).

## 1. Produkt

![Produkt](img/product.png)

Wybierz jeden z 10 produktów demo, głównie marek znanych w Polsce (Samsung, Bosch, LG, Haier, De'Longhi, Krups,
Melitta i inne). Dla każdego aplikacja przeanalizowała 200 najnowszych opinii.

- **Kafelki** na górze: liczba opinii, średnia ocena oraz odsetek opinii pozytywnych (4–5★) i negatywnych (1–2★).
- **Raport dla menedżera**: jedno zdanie o odbiorze produktu, zalety i rzeczy do poprawy. Pisze go LLM
  wyłącznie na podstawie tematów i cytatów z opinii.
- **Tematy**: wykres pokazuje, ile opinii chwali (zielone, w prawo) lub krytykuje (pomarańczowe, w lewo) każdy
  temat. Pod wykresem są rozwijane sekcje z cytatami.

**Spróbuj:**

1. Zostań przy kapsułkach wielokrotnego użytku do ekspresów Nespresso (3,0★, 46% opinii negatywnych). Klienci
   chwalą oszczędność (cena: 31 pochwał), ale najczęstsza skarga to dopasowanie (30 opinii). Rozwiń
   „Dopasowanie”: „The capsules are too small for the Nespresso machine, it keeps going through the hole”.
2. Przełącz „Źródło tematów” na **Słownik słów kluczowych**. Według słownika dopasowanie częściej się chwali
   (14 opinii), niż krytykuje (10). Słownik ocenia wydźwięk zdania po słowach, a zdanie z cytatu powyżej nie ma
   słów nacechowanych emocjonalnie, więc je pomija. LLM rozumie, że to skarga.
3. Wybierz minilodówkę BLACK+DECKER (3,5★). Tu dominują skargi na jakość (67 opinii) i dostawę (26).
4. Wybierz filtry do kawy Melitta (4,6★). To produkt chwalony głównie za cenę (53 opinie).

## 2. Analiza na żywo

![Analiza na żywo](img/live.png)

Tu wklejasz link do dowolnego produktu AGD z Amazona. Aplikacja bierze 100 jego najnowszych opinii
z pobranego zbioru (nie z Amazona) i liczy wszystko od nowa: cztery metody, tematy i raport. Wynik wygląda tak
samo jak na ekranie Produkt.

- Działa tylko przy lokalnym uruchomieniu, z kartą graficzną i Ollamą. W Dockerze ekran wyjaśnia, czego brakuje.
- Działa dla produktów, które mają opinie w zbiorze (opinie do 2023 roku).
- Na RTX 5070 analiza trwa około minuty. Pierwsza po dłuższej przerwie trwa do 2 minut, bo Ollama musi
  wczytać model.

**Spróbuj:** wklej przykładowy link spod pola (filtr wody do lodówki Electrolux) i kliknij **Analizuj**.
Zobaczysz 100 opinii ze średnią 4,1★ i 18% negatywnych. Raport wskaże dopasowanie i cenę jako główne zalety.

**Ograniczenie LLM:** porównuj zalety z raportu z wykresem tematów. Zdarza się, że model wymienia jako zaletę
temat, który więcej opinii krytykuje, niż chwali. Raport warto czytać razem z liczbami i cytatami.

## 3. Porównanie metod

![Porównanie metod](img/comparison.png)

Cztery metody oceniają sentyment tych samych 1000 opinii testowych (spoza produktów demo). Prawdą jest ocena
klienta: 1–2★ to opinia negatywna, 3★ neutralna, 4–5★ pozytywna.

| Metoda | Trafność | F1 makro | Czas na 1000 opinii |
|---|---|---|---|
| Słownik (VADER) | 80,0% | 0,52 | poniżej 1 s |
| Klasyczne ML (TF-IDF) | 88,1% | 0,70 | poniżej 1 s |
| Gotowy model (Hugging Face) | 91,7% | 0,75 | 10 s |
| LLM | 88,3% | 0,70 | 7 min |

**Jak czytać wyniki:**

- Do samego sentymentu wystarczy gotowy model z Hugging Face: jest najlepszy i liczy 1000 opinii w 10 sekund.
- LLM najlepiej wyłapuje opinie negatywne (F1 0,88), a przy tym jako jedyny podaje tematy i cytaty.
- Wykres „Które opinie sprawiają kłopot” pokazuje F1 osobno dla każdej klasy. Wszystkie metody najsłabiej
  radzą sobie z opiniami neutralnymi, których jest tylko 4,7%.
- Tabela „Ile kosztowałby LLM przez API” przelicza zużyte tokeny na ceny trzech tanich modeli w chmurze:
  od 0,23 do 0,93 zł za 1000 opinii.
- Wykres „Tematy: słownik czy LLM?” pokazuje, że LLM znajduje więcej wzmianek o każdym temacie niż słownik,
  np. o dopasowaniu 304 zamiast 193 na 1000 opinii.

Na dole ekranu jest wyjaśnienie, dlaczego główną miarą jest F1 makro, a nie trafność.

## 4. Opinie

![Opinie](img/reviews.png)

Opinie testowe z werdyktem każdej metody. Pomyłki (werdykt inny niż ocena klienta) są zaznaczone kolorem.
Kolumna **Produkt** podaje, czego dotyczy opinia, a ikonka w kolumnie **Link** otwiera stronę produktu na Amazonie.
Produkt mógł już zniknąć z oferty, bo opinie są do 2023 roku.

Przełącznik **Pokaż** filtruje tabelę:

- **Wszystkie**: 1000 opinii,
- **Metody się nie zgadzają**: 284 opinie, w których co najmniej jedna metoda ma inne zdanie,
- **Słownik się myli, LLM nie**: 138 opinii.

**Spróbuj:** kliknij szósty wiersz („What a waste of money!!! Poor quality”, 1★, zamienna grzałka do suszarek Samsung).
Pod tabelą pojawi się cała opinia i tematy z obu źródeł. Słownik uznał ją za pozytywną, bo klient pisze
z ironią: „if you like taking your dryer apart every 5 months, this is the perfect item to get”. Słowo „perfect”
przeważyło. Pozostałe metody poprawnie uznały opinię za negatywną.

## Słowniczek

| Pojęcie | Znaczenie |
|---|---|
| Sentyment | Wydźwięk opinii: negatywna, neutralna albo pozytywna |
| Słownik (VADER) | Gotowa lista angielskich słów z przypisanym wydźwiękiem; ocena opinii wynika z jej słów |
| Słownik słów kluczowych | Listy słów dla każdego tematu (np. „fit”, „compatible” dla dopasowania); tania alternatywa dla LLM |
| Klasyczne ML (TF-IDF) | Model uczony na 100 000 innych opinii: uczy się, jakie słowa i pary słów pojawiają się przy danej ocenie |
| Gotowy model (Hugging Face) | Sieć neuronowa wytrenowana przez kogoś innego na opiniach o produktach, pobierana za darmo |
| LLM | Duży model językowy; tu działa lokalnie i oprócz sentymentu znajduje tematy i cytaty |
| Trafność | Jaka część opinii dostała dobrą ocenę |
| F1 makro | Średnia F1 z trzech klas; każda klasa liczy się tak samo, więc rzadkie opinie neutralne też mają znaczenie |
| Token | Kawałek tekstu, od którego liczy się cenę w płatnych API LLM |
| Cytat dosłowny | Cytat z LLM, który rzeczywiście jest w opinii; pozostałe aplikacja pomija |
