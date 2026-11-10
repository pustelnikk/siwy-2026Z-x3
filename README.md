# audiosanity: testy poprawności map atrybucji dla klasyfikatorów audio

Projekt SIWY 2026Z, X3: *Do audio explanations pass the tests that image explanations failed?*
Autorzy: Mariusz Pakulski, Filip Misztal, Paweł Łasica.

Pakiet uruchamia na modelu audio zestaw testów znanych z wizji komputerowej:

| test | moduł | pytanie |
|---|---|---|
| model randomization | `checks/randomization.py` | czy mapa się zmienia, gdy kolejne warstwy (od wyjścia) dostają losowe wagi? |
| time-shift equivariance | `checks/shift.py` | czy mapa przesuniętego klipu to przesunięta mapa? |
| gain / resample / codec invariance | `checks/invariance.py` | czy mapa jest odporna na zmiany, które nie powinny zmieniać decyzji? |
| loudness confound | `checks/loudness.py` | czy mapa po prostu odwzorowuje log-spektrogram / onsety? |
| agreement | `checks/agreement.py` | czy różne metody wskazują to samo? |

Każdy test ma wbudowaną **losową próbę kontrolną**: to samo porównanie wykonane między dwoma
*różnymi* klipami. Każdy wynik jest raportowany dla każdej kombinacji **normalizacji** mapy
(`raw`, `abs`, `abs_max`, `max_abs`) i **miary podobieństwa** (`spearman`, `pearson`, `ssim`, `topk`),
bo werdykt randomizacji zależy od obu tych wyborów. Przedziały ufności są liczone metodą bootstrap.

Design proposal: [docs/design_proposal.md](docs/design_proposal.md)

## Instalacja

Wymagane: [uv](https://docs.astral.sh/uv/) oraz opcjonalnie [just](https://github.com/casey/just)
(`uv tool install rust-just`). Na Windows/Linux instalowany jest torch z CUDA 12.8; działa też bez GPU.

```bash
uv sync                          # tworzy .venv z zależnościami (także dev: pytest, ruff)
uv run audiosanity download esc50   # ~600 MB do data/raw/ESC-50-master
```

## Użycie

```bash
just smoke                                   # 4 klipy, ok. 30 s na GPU: sprawdza, czy wszystko działa
just run configs/pilot.yaml                  # pilot: AST + ESC-50, 50 klipów, 4 metody, 5 testów
just run configs/pilot.yaml data.n_clips=10 attribution.methods=[saliency]   # nadpisanie configu
just test                                    # testy jednostkowe (małe losowe modele, bez pobierania)
just lint / just fmt                         # ruff
just tb                                      # TensorBoard nad katalogiem results/
uv run audiosanity methods                   # lista modeli, metod, testów i miar
```

Bez `just`: `uv run audiosanity run configs/pilot.yaml data.n_clips=10`.

Każdy przebieg tworzy `results/<name>_<data>/`:

| plik | zawartość |
|---|---|
| `config.yaml` | pełny rozwiązany config (można go uruchomić ponownie) |
| `predictions.csv` | klipy, wyjaśniana klasa, wynik modelu |
| `maps.npz` | oryginalne mapy atrybucji, `(n_clips, T, F)` per metoda |
| `results.csv` | wszystkie porównania na poziomie klipu (format długi) |
| `summary.csv` | średnie + 95% CI per test / metoda / warunek / normalizacja / miara |
| `figures/<test>.png` | wykres per test (linia przerywana = próba kontrolna) |
| `tb/` | logi TensorBoard |

## Struktura

```
configs/                YAML z eksperymentami (schemat i wartości domyślne: src/audiosanity/config.py)
src/audiosanity/
  config.py             schemat configu (dataclasses + OmegaConf)
  data.py               pobieranie ESC-50, wybór klipów, okna ze środka klipu
  models/               interfejs AudioClassifier + wrappery AST i wav2vec2 (HF)
  attribution.py        metody atrybucji (Captum) i explain(): fala -> mapa na siatce czas x częstotliwość
  transforms.py         gain, resampling, kodek (MP3/OGG przez libsndfile, z wyrównaniem opóźnienia)
  measures.py           normalizacje, miary podobieństwa, bootstrap
  checks/               po jednym module na test + wspólny CheckContext
  pipeline.py           uruchomienie eksperymentu i zapis wyników
  plotting.py, cli.py
tests/                  pytest, małe losowo zainicjalizowane modele
docs/                   design proposal
data/raw/               surowe dane, tylko do odczytu (poza gitem)
results/                wyniki przebiegów (poza gitem)
```

### Wspólna siatka map

Każdy model rzutuje atrybucję na siatkę `(T, F)` o skoku `hop_length` próbek (10 ms dla 16 kHz):
AST ma 128 pasm mel, a model falowy (wav2vec2) ma `F = 1`, czyli sumę atrybucji próbek w ramce.
Ramka `k` zaczyna się od próbki `k * hop_length`, więc przesunięcie audio o `k` ramek przesuwa mapę
dokładnie o `k` ramek. Analizowane okno (`data.clip_seconds`) jest wycinane ze **środka** klipu,
co eliminuje padding w teście przesunięcia.

## Jak rozszerzać

- **Nowy model**: podklasa `AudioClassifier` (`features`, `forward`, `to_grid`, `reference`,
  `layer_groups`, `from_pretrained`) zarejestrowana w `models/__init__.py:REGISTRY`.
  Do tego metoda `tiny()` i parametr w `tests/conftest.py`, żeby testy objęły nowy model.
- **Nowa metoda atrybucji**: funkcja w `attribution.py` dodana do `METHODS`. Jeśli nie da się jej
  zastosować do danej architektury, para trafia do `UNAVAILABLE` i w wynikach jest oznaczona jako
  *unavailable* (nie jako *fail*).
- **Nowy test**: moduł w `checks/` z `NAME` i `run(ctx) -> list[dict]`, zarejestrowany w
  `checks/__init__.py`. Do porównań i próby kontrolnej służą `ctx.compare(...)` i `ctx.control(...)`.

## Konwencje

- Formatowanie i lint: `ruff` (długość linii 110). Przed commitem: `just fmt && just test`.
- Commity: [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, ...).
- Konfiguracja tylko w `configs/`, bez stałych w kodzie. `data/raw` jest tylko do odczytu.
