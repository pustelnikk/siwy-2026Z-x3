# SIWY - Design Proposal - 2026Z

Autorzy:

- Mariusz Pakulski
- Filip Misztal
- Paweł Łasica

Temat: Do audio explanations pass the tests that image explanations failed? (X3)


## Harmonogram

| Tydzień | Okres | Krótki opis | Plan |
|---------|-------|-------|------|
| 1 | 26.10 - 01.11 | Design proposal | Przygotowanie design proposalu (28.10 deadline). Zapoznanie się z literaturą. Określenie, które metody w wizji nie przeszły których testów. Inwentaryzacja checkpointów. Spisanie wyniku obalającego i progów zaliczenia. |
| 2 | 02.11 - 08.11 | Prototyp | Stworzenie prototypu + spotkanie (6.11 deadline). Pilot na gotowych modelach (detektor SONICS + tagger AudioSet): 4 metody, 50 klipów, randomizacja kaskadowa z normalizacją i bez, krzywa. |
| 3 | 09.11 - 15.11 | Fundamenty | Pobranie pozostałych checkpointów, przygotowanie danych SONICS i ESC-50, wybór 300 klipów (środki długich klipów). Interfejsy pakietu. Start testów głowic (AST, PANN, wav2vec2). |
| 4 | 16.11 - 22.11 | Walidacja modeli i testów | Dokończenie głowic i sprawdzenie dokładności. Rozszerzenie randomizacji na nowe modele. Określenie mocy testów. Start testów audio. |
| 5 | 23.11 - 29.11 | Testy audio | Ekwiwariancja przesunięcia, wzmocnienie, resampling, kodek, test głośności. Mapy dla wszystkich architektur. |
| 6 | 30.11 - 06.12 | Domknięcie testów | Test zgodności, 5. metoda, meta-ewaluacja miar podobieństwa i wierności, testy jednostkowe. Zamrożenie definicji testów, start pełnej siatki. |
| 7 | 07.12 - 13.12 | Pełna siatka | 30 par × 300 klipów, ziarna. Przygotowanie pipeline'u tabeli i wykresów. |
| 8 | 14.12 - 20.12 | Analiza | Tabela zaliczeń i niezaliczeń z przedziałami, wrażliwość na normalizację i miarę, kolumna „czy wynik z wizji się powtarza” dla każdej pary metoda–test. Start raportu. |
| 9 | 21.12 - 27.12 | Bufor | Ponowienie nieudanych przebiegów, szlify kodu. Święta. |
| 10 | 28.12 - 03.01 | Wolne | Tydzień wolnego. |
| 11 | 04.01 - 10.01 | Dokumentacja | Raport z wynikami, dokumentacja, instrukcja użytkowania, wydanie pakietu (GitHub/PyPI), filmik podsumowujący. |
| 12 | 11.01 - 17.01 | Oddanie | Oddanie projektu (deadline 14.01). |


## Bibliografia

* Sanity Checks for Saliency Maps (arXiv:1810.03292)
* The (Un)reliability of saliency methods (arXiv:1711.00867)
* When Explanations Lie: Why Many Modified BP Attributions Fail (arXiv:1912.09818)
* Investigating sanity checks for saliency maps with image and text classification (arXiv:2106.07475)
* The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective (arXiv:2202.01602)
* Sanity Checks for Saliency Metrics (arXiv:1912.01451)
* The Meta-Evaluation Problem in Explainable AI: Identifying Reliable Estimators with MetaQuantus (arXiv:2302.07265)
* Listenable Maps for Audio Classifiers (arXiv:2403.13086)
* A Framework for Evaluating Faithfulness in Explainable AI for Machine Anomalous Sound Detection (arXiv:2601.19017)
* The Perceived Fragility of Explanations in Audio Models: Manipulation of Attribution with Unchanged Predictions (arXiv:2606.14466)

## Planowany zakres eksperymentów

Eksperymenty zostaną przeprowadzone w oparciu o siatkę badawczą obejmującą 3 odmienne architektury audio, 2 zadania, 5 metod atrybucji oraz 300 próbek testowych per konfiguracja (z wykorzystaniem próbkowania bootstrapowego i próby kontrolnej). Każda z 30 powstałych konfiguracji zostanie poddana serii 5 testów.



### Metody atrybucji

5 metod do wyboru z:

* AudioLIME
* GradCAM
* LRP
* Integrated Gradients
* Listenable Map
* Occlusion
* Guided BackProp
* Saliency



### Architektury bazowe:
* model spektrogramowy - PANNs.
* patch transformer - Audio Spectrogram Transformer (AST).
* samo-nadzorowany model fali dźwiękowej - wav2vec 2.0.

### Zadania/zbiory:
* SONICS (https://www.kaggle.com/datasets/awsaf49/sonics-dataset) - detekcja syntetycznego dźwięku/muzyki.
* ESC-50 (https://www.kaggle.com/datasets/eliasmarcon/environmental-sound-classification-50) -detekcja 50 klas zdarzeń dźwiękowych.

### Zestaw testów:
1. Model Randomization - sukcesywna randomizacja wag warstwa po warstwie, z włączoną i wyłączoną normalizacją map atrybucji.
2. Time-Shift Equivariance - przesunięcie czasowe sygnału audio i ocena, czy mapa atrybucji przesuwa się w sposób równoważny, z uwzględnieniem obcięcia ramek w centrach długich nagrań.
3. Gain & Codec Invariance - odporność wyjaśnień na zmianę wzmocnienia (gain) oraz kompresję stratną.
4. Loudness Confound - pomiar korelacji mapy atrybucji ze spektrogramem amplitudy logarytmicznej oraz funkcją onset strength.
5. Disagreement Test & Reliability Checks - ocena spójności wskazań różnych metod atrybucji na tych samych próbkach.

Wynik obalający

Projekt zostanie uznany za obalony na wczesnym etapie, jeśli wszystkie badane metody atrybucji przejdą każdy z testów na wszystkich trzech architekturach modeli, a ich odchylenia nie będą odbiegać od szumu losowej próby kontrolnej.

## Planowana funkcjonalność programu

* dobór testowanego modelu audio
* przekształcenia audio (zgodnie z punktem Zestaw testów)
* kalkulacja przedziałów ufności
* wyliczenie reliablity metrics
* osobny interfejs per test z wbudowaną losową próbą kontrolną random control i konfigurowalną normalizacją map
* automatyczne uruchamiania całego zbioru testów weryfikacyjnych z wbudowaną losową próbą kontrolną random control i konfigurowalną normalizacją map
* export macierzy wyników testów we wszystkich 30 konfiguracjach oraz ich porównanie z analogicznymi wynikami znanymi z wizji komputerowej, wartościami przedziałów ufności oraz reliability metrics.

## Planowany stack technologiczny
Poniższe narzędzia są propozycją i mogą być w razie potrzeby zmienione lub niewykorzystane, w zależności od potrzeb.

### Język i środowisko
- Python
- just
- uv 
- pytest

### Biblioteki DL
- transformers
- huggingface
- torch

### Biblioteki XAI/Metryki
- captum
- zennit
- quantus
- speechbrain

### Biblioteki audio
- torchaudio
- Pydub / FFmpeg / SoundFile

### Pozostałe
- numpy / scipy