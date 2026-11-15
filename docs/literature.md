# Analiza literatury

Prace z karty zadania X3 w trzech grupach: **A** to testy z wizji, które przenosimy do audio,
**B** to wiarygodność miar, **C** to prace o wyjaśnieniach w audio. „Nie podano” oznacza, że
autorzy nie zamieszczają danej informacji.

## Zestawienie

| # | Praca | Kod | Modele i dane | Metryki | Zasoby | Wykorzystanie u nas |
|---|---|---|---|---|---|---|
| A1 | Adebayo, Gilmer, Muelly, Goodfellow, Hardt, Kim, *Sanity Checks for Saliency Maps*, NeurIPS 2018, [arXiv:1810.03292](https://arxiv.org/abs/1810.03292) | link w pracy: [goo.gl/hBmhDt](https://goo.gl/hBmhDt) | Inception v3 (ImageNet, pretrenowany, TF-slim); CNN i MLP na MNIST i Fashion-MNIST (trenowane przez autorów); Inception v4 na zdjęciach RTG (wiek kości) | Spearman na wartościach bezwzględnych i ze znakiem, SSIM, korelacja histogramów HOG | nie podano | `checks/randomization.py` |
| A2 | Kindermans, Hooker, Adebayo, Alber, Schütt, Dähne, Erhan, Kim, *The (Un)reliability of Saliency Methods*, 2017, [arXiv:1711.00867](https://arxiv.org/abs/1711.00867) | brak linku w pracy | MLP (3 × 1024 ReLU) trenowany przez autorów na MNIST (98,3%); ilustracje na VGG-16 (ImageNet) | brak miary liczbowej, porównanie wizualne map | nie podano | `checks/invariance.py` (`input_shift_*`), `attribution.ig_baseline` |
| A3 | Sixt, Granz, Landgraf, *When Explanations Lie: Why Many Modified BP Attributions Fail*, ICML 2020, [arXiv:1912.09818](https://arxiv.org/abs/1912.09818) | [berleon/when-explanations-lie](https://github.com/berleon/when-explanations-lie) | VGG-16 i ResNet-50 (ImageNet), mała sieć na CIFAR-10; 200 obrazów walidacyjnych; implementacje z iNNvestigate i deeplift | Cosine Similarity Convergence (CSC), SSIM w teście randomizacji, test losowego logitu | jedna maszyna z dwiema kartami GPU, ok. 1 doba | wybór LRP (Zennit) i jego reguł |
| A4 | Kokhlikyan, Miglani, Alsallakh, Martin, Reblitz-Richardson, *Investigating Sanity Checks for Saliency Maps with Image and Text Classification*, 2021, [arXiv:2106.07475](https://arxiv.org/abs/2106.07475) | brak linku w pracy | Inception (obrazy, randomizacja Xavier); BERT dostrojony przez autorów na SST-2 (sentyment); MNIST do randomizacji etykiet | SSIM (obrazy), podobieństwo cosinusowe (tekst), infidelity, max-sensitivity | nie podano | porównanie metod z mnożnikiem wejścia i bez (Saliency vs IxG/IG) |
| A5 | Krishna, Han, Gu, Wu, Jabbari, Lakkaraju, *The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective*, TMLR, [arXiv:2202.01602](https://arxiv.org/abs/2202.01602) | brak linku w tekście | dane tabelaryczne (COMPAS, German Credit; LR, MLP, las losowy, boosting), tekst (AG News, LSTM), obrazy (ImageNet, pretrenowany ResNet-18) | feature, rank, sign i signed rank agreement (top-k), korelacja rangowa, pairwise rank agreement | nie podano | `checks/agreement.py` |
| B1 | Tomsett, Harborne, Chakraborty, Gurram, Preece, *Sanity Checks for Saliency Metrics*, AAAI 2020, [arXiv:1912.01451](https://arxiv.org/abs/1912.01451) | brak linku w pracy | CNN trenowany przez autorów na CIFAR-10 (86%) | AOPC (MoRF, LeRF), faithfulness; rzetelność: alfa Krippendorffa, spójność między metodami i między metrykami (Spearman) | nie podano | meta-ewaluacja miar (tydzień 6) |
| B2 | Hedström, Bommer, Wickstrøm, Samek, Lapuschkin, Höhne, *The Meta-Evaluation Problem in Explainable AI: Identifying Reliable Estimators with MetaQuantus*, TMLR 2023, [arXiv:2302.07265](https://arxiv.org/abs/2302.07265) | [annahedstroem/MetaQuantus](https://github.com/annahedstroem/MetaQuantus) | MNIST, Fashion-MNIST, cMNIST (LeNet, ResNet-9, trenowane przez autorów); ImageNet (pretrenowany ResNet-18; ViT i Swin w porównaniu architektur) | testy IPT i MPT, kryteria noise resilience i adversary reactivity, wynik MC; 10 miar z Quantus w 5 kategoriach | GPU NVIDIA A100 (40 i 80 GB), Tesla V100S | meta-ewaluacja miar (tydzień 6) |
| C1 | Paissan, Ravanelli, Subakan, *Listenable Maps for Audio Classifiers (L-MAC)*, ICML 2024, [arXiv:2403.13086](https://arxiv.org/abs/2403.13086) | [fpaissan.github.io/lmac](https://fpaissan.github.io/lmac) | CNN14 pretrenowany na VGGSound (SimCLR), dostrojony przez autorów na ESC-50 z szumem WHAM!; dekoder trenowany przez autorów; dodatkowo UrbanSound8K | AI, AD, AG, FF, Fid-In, SPS, COMP; ROAR i randomizacja modelu (SSIM); badanie odsłuchowe MUSHRA (15 osób) | nie podano | poza siatką (wymaga treningu dekodera) |
| C2 | Buck, Cosma, Phillips, Conway, Baker, *A Framework for Evaluating Faithfulness in XAI for Machine Anomalous Sound Detection Using Frequency-Band Perturbation*, preprint 2026, [arXiv:2601.19017](https://arxiv.org/abs/2601.19017) | brak linku w pracy | DCASE 2023 Task 2 (ToyADMOS2, MIMII-DG, 14 typów maszyn); enkoder konwolucyjny Wilkinghoffa i in. | korelacja Spearmana między ważnością pasma a zmianą predykcji po jego usunięciu; AUC, pAUC | nie podano | wybór Occlusion; kandydat na miarę wierności |
| C3 | Kitłowski, Wiącek, Modrzejewski, *The Perceived Fragility of Explanations in Audio Models*, ICML 2026 Workshop on ML for Audio, [arXiv:2606.14466](https://arxiv.org/abs/2606.14466) | [cncPomper/Audio-XAI](https://github.com/cncPomper/Audio-XAI) | SONICS (100 nagrań); detektory VGGish, AST i SpecTTTra (`spectttra-gamma-5s`) | PESQ, STOI, ViSQOL, PEAQ, Zimtohrli, CDPAM; podobieństwo cosinusowe i top-10 map; Audio Fragility Score | klaster PLGrid (Cyfronet AGH), bez typu GPU i czasu | testy niezmienniczości na SONICS; punkt odniesienia dla detektora SONICS i LRP na AST |

## Komentarze

Dla każdej pracy: co pokazała i co z tego bierzemy.

### A. Testy z wizji

**A1, Adebayo i in. 2018.** Jeśli zamienimy wagi modelu na losowe, a mapa prawie się nie zmieni,
to metoda nie wyjaśnia modelu. Gradient i Grad-CAM przechodzą, Guided Backprop oblewa. Mapy IxG
i IG po randomizacji nadal przypominają wejście.
*U nas:* to nasz test randomizacji. Pilot na AST daje ten sam wynik: Saliency i SmoothGrad
przechodzą, IxG i IG nie.

**A2, Kindermans i in. 2017.** Dwa modele dają identyczne wyniki, ale jeden dostaje wejście
przesunięte o stałą. Dobra metoda powinna dać obu tę samą mapę. Gradient przechodzi, IxG oblewa,
a IG zależy od wyboru baseline'u.
*U nas:* głośność w log-melu to właśnie przesunięcie o stałą. Dodaliśmy ten test dla AST
i baseline IG jako parametr. Wynik jest zgodny z pracą.

**A3, Sixt i in. 2020.** Wyjaśnia, dlaczego część wariantów LRP oblewa: przy regule z⁺ mapa
przestaje zależeć od głębszych warstw i od klasy. LRP-z i DeepLIFT tego problemu nie mają.
*U nas:* porównamy dwa warianty LRP na PANNs. Na transformerach (AST, wav2vec2) LRP jest
niedostępne.

**A4, Kokhlikyan i in. 2021.** Metody, które mnożą przez wejście (np. IG), przenoszą do mapy jego
strukturę i na obrazach oblewają randomizację. Na tekście tego efektu nie ma.
*U nas:* pilot pokazuje ten sam wzór co na obrazach: IxG i IG oblewają, Saliency nie.

**A5, Krishna i in. 2022.** Różne metody często wskazują różne fragmenty jako ważne, a praktycy
nie wiedzą, której wierzyć. Praca daje gotowe miary zgodności.
*U nas:* test agreement. Dodamy zgodność znaków, bo pilot pokazał, że metody wskazują te same
miejsca, ale z przeciwnym znakiem.

### B. Rzetelność miar

**B1, Tomsett i in. 2020.** Miary oceniające wyjaśnienia są mało wiarygodne: dają różne rankingi
metod i zależą od szczegółów implementacji.

**B2, Hedström i in. 2023 (MetaQuantus).** Proponuje sposób sprawdzania miar: dobra miara nie
reaguje na drobne zmiany, ale reaguje na zepsucie modelu.

*U nas (B1, B2):* nasze werdykty opierają się na Spearmanie, SSIM i top-k, więc te miary też
trzeba sprawdzić. Planowane na tydzień 6.

### C. Wyjaśnienia w audio

**C1, L-MAC (Paissan i in. 2024).** Zamiast mapy generuje maskę, z której powstaje dźwięk do
odsłuchania. Wymaga trenowania osobnego dekodera dla każdego modelu.
*U nas:* nie wchodzi do siatki z powodu kosztu treningu.

**C2, Buck i in. 2026.** Usuwa kolejne pasma częstotliwości i sprawdza, czy metoda wskazała
pasma, które naprawdę wpływają na decyzję. Najlepiej wypada Occlusion.
*U nas:* Occlusion w zestawie metod; test pasm to kandydat na dodatkową miarę.

**C3, Kitłowski i in. 2026.** Niesłyszalne zmiany w dźwięku potrafią zmienić mapy detektorów
deepfake'ów bez zmiany decyzji. Najbardziej podatny jest AST.
*U nas:* uzasadnia testy niezmienniczości na SONICS. Ich kod przyda się przy dodawaniu
detektora SONICS.

## Luka, którą wypełnia projekt

W wizji wiele metod wyjaśniania oblało podstawowe testy, a same miary okazały się mało
wiarygodne. W audio sprawdzono tylko pojedyncze testy (C1–C3). Nikt nie uruchomił pełnego
zestawu testów na kilku modelach audio i to robi nasz projekt.
