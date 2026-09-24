## Dane

W analizie wykorzystano dane **World Development Indicators (WDI)** udostępniane przez Bank Światowy.

Dane źródłowe można pobrać ze strony:

https://databank.worldbank.org/source/world-development-indicators

Podczas pobierania danych należy zaznaczyć **wszystkie pozycje w sekcjach Country, Series oraz Time**, tak aby pobrany zbiór obejmował pełny zakres danych wykorzystywany w procesie przygotowania zbioru.

Ze względu na rozmiar oryginalny plik danych nie został umieszczony w repozytorium.

Po pobraniu należy umieścić go w folderze:

`datasets/`

pod nazwą:

`WDI_Original_Dataset.csv`

Repozytorium zawiera przetworzone zbiory danych wykorzystane w kolejnych etapach badania w folderze `processed_data/` oraz pełne wyniki analiz w folderze `results/`.

## Uruchomienie

Wymagane biblioteki można zainstalować za pomocą:

```bash
pip install -r requirements.txt
