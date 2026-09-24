from pathlib import Path

# import matplotlib.pyplot as plt
import pandas as pd

from my_funcs.hellwig import hellwig_compensational, hellwig_manhattan
from preprocessing.WDI_hellwig import D0, df_panel, pattern, stim_or_destim

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Obliczenie rankingów klasycznych i kompensacyjnych
#    dla każdego roku z okresu 2000–2020
# ------------------------------------------------------------

years = [str(y) for y in range(2000, 2021)]

results = []
comp_distance_results = []

for year in years:
    df_year = (
        df_panel[df_panel["Year"] == year]
        .drop(columns=["Year"])
        .set_index("Country Name")
    )

    # Metoda klasyczna Manhattan
    q_classic, score_classic, deviations_classic, _, _ = hellwig_manhattan(
        df_year,
        stim_or_destim
    )

    classic_results = q_classic.reset_index()
    classic_results.columns = ["Country Name", "q_classic"]
    classic_results["rank_classic"] = (
        classic_results["q_classic"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Metoda kompensacyjna
    q_comp, score_comp, deviations_comp, _, _ = hellwig_compensational(
        df_year,
        stim_or_destim,
        pattern=pattern,
        D0=D0
    )

    year_comp_distances = score_comp.reset_index()
    year_comp_distances.columns = ["Country Name", "deviation"]
    year_comp_distances["Year"] = int(year)

    comp_distance_results.append(year_comp_distances)

    comp_results = q_comp.reset_index()
    comp_results.columns = ["Country Name", "q_compensational"]
    comp_results["rank_compensational"] = (
        comp_results["q_compensational"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Połączenie wyników dla danego roku
    year_results = classic_results.merge(
        comp_results,
        on="Country Name",
        how="inner"
    )

    year_results["Year"] = int(year)

    results.append(year_results)


comparison_results = pd.concat(results, ignore_index=True)

# ------------------------------------------------------------
# Eksport 02: pełne wyniki metody kompensacyjnej, 2000–2020
# ------------------------------------------------------------

compensational_rankings_full = (
    comparison_results[
        [
            "Country Name",
            "Year",
            "q_compensational",
            "rank_compensational"
        ]
    ]
    .sort_values(
        ["Year", "rank_compensational", "Country Name"]
    )
    .reset_index(drop=True)
)

compensational_rankings_full.to_csv(
    RESULTS_DIR / "02_compensational_rankings_2000_2020.csv",
    index=False
)


# ------------------------------------------------------------
# Eksport 04: pełne rozwinięcie tabel 5 i 6
# lata 2000, 2010 i 2020
# ------------------------------------------------------------

compensational_table_parts = []

for year in [2000, 2010, 2020]:
    year_table = (
        comparison_results[
            comparison_results["Year"] == year
        ]
        .sort_values(
            ["rank_compensational", "Country Name"]
        )
        [["Country Name", "q_compensational"]]
        .reset_index(drop=True)
        .rename(columns={
            "Country Name": f"Country_{year}",
            "q_compensational": f"q_{year}"
        })
    )

    compensational_table_parts.append(year_table)

compensational_rankings_3years = pd.concat(
    compensational_table_parts,
    axis=1
)

compensational_rankings_3years.insert(
    0,
    "Position",
    range(1, len(compensational_rankings_3years) + 1)
)

for column in ["q_2000", "q_2010", "q_2020"]:
    compensational_rankings_3years[column] = (
        compensational_rankings_3years[column].round(4)
    )

compensational_rankings_3years.to_csv(
    RESULTS_DIR / "04_compensational_rankings_2000_2010_2020.csv",
    index=False
)

# ------------------------------------------------------------
# Eksport 06: łączne odchylenia od wzorca
# w metodzie kompensacyjnej, lata 2000–2020
# ------------------------------------------------------------

compensational_deviations = pd.concat(
    comp_distance_results,
    ignore_index=True
)

compensational_deviations = (
    compensational_deviations
    .sort_values(["Year", "Country Name"])
    .reset_index(drop=True)
)

compensational_deviations.to_csv(
    RESULTS_DIR / "06_compensational_deviations_2000_2020.csv",
    index=False
)

# ------------------------------------------------------------
# 2. Korelacja rang Spearmana między metodą klasyczną
#    i kompensacyjną w każdym roku
# ------------------------------------------------------------

correlations = []

for year in range(2000, 2021):
    df_corr = comparison_results[comparison_results["Year"] == year]

    spearman_corr = df_corr["rank_classic"].corr(
        df_corr["rank_compensational"],
        method="spearman"
    )

    pearson_corr = df_corr["rank_classic"].corr(
        df_corr["rank_compensational"],
        method="pearson"
    )

    correlations.append({
        "Year": year,
        "spearman_corr": spearman_corr,
        "pearson_corr": pearson_corr
    })


correlation_results = pd.DataFrame(correlations)

print("\nKorelacja między rankingiem klasycznym i kompensacyjnym:")
print(correlation_results.round(4).to_string(index=False))


# ------------------------------------------------------------
# 3. Średnia korelacja w całym okresie
# ------------------------------------------------------------

mean_spearman = correlation_results["spearman_corr"].mean()
mean_pearson = correlation_results["pearson_corr"].mean()

print("\nŚrednia korelacja w latach 2000–2020:")
print(f"Spearman: {mean_spearman:.4f}")
print(f"Pearson:  {mean_pearson:.4f}")


# ------------------------------------------------------------
# 4. Największy wzrost wartości miernika kompensacyjnego
#    między rokiem 2000 a 2020
# ------------------------------------------------------------

comp_2000_2020 = (
    comparison_results[comparison_results["Year"].isin([2000, 2020])]
    [["Country Name", "Year", "q_compensational", "rank_compensational"]]
    .pivot(index="Country Name", columns="Year", values=["q_compensational", "rank_compensational"])
)

comp_2000_2020.columns = [
    "q_2000", "q_2020",
    "rank_2000", "rank_2020"
]

comp_2000_2020["q_change_2000_2020"] = (
    comp_2000_2020["q_2020"] - comp_2000_2020["q_2000"]
)

comp_2000_2020["rank_change_2000_2020"] = (
    comp_2000_2020["rank_2000"] - comp_2000_2020["rank_2020"]
)

largest_increases = (
    comp_2000_2020
    .sort_values("q_change_2000_2020", ascending=False)
    .head(10)
    .reset_index()
)

print("\nNajwiększy wzrost wartości miernika kompensacyjnego w latach 2000–2020:")
print(largest_increases.round(4).to_string(index=False))

# ------------------------------------------------------------
# 5. Największy spadek wartości miernika kompensacyjnego
#    między rokiem 2000 a 2020
# ------------------------------------------------------------

largest_decreases = (
    comp_2000_2020
    .sort_values("q_change_2000_2020", ascending=True)
    .head(10)
    .reset_index()
)

print("\nNajwiększy spadek wartości miernika kompensacyjnego w latach 2000–2020:")
print(largest_decreases.round(4).to_string(index=False))

# ------------------------------------------------------------
# 6. Analiza stabilności wartości miernika kompensacyjnego
#    w całym okresie 2000–2020
# ------------------------------------------------------------

comp_values_panel = (
    comparison_results
    .pivot(index="Country Name", columns="Year", values="q_compensational")
)

# Zakres zmienności miernika w całym okresie
stability_results = pd.DataFrame(index=comp_values_panel.index)

stability_results["q_min"] = comp_values_panel.min(axis=1)
stability_results["q_max"] = comp_values_panel.max(axis=1)
stability_results["q_range"] = stability_results["q_max"] - stability_results["q_min"]
stability_results["q_std"] = comp_values_panel.std(axis=1)

stability_results["q_2000"] = comp_values_panel[2000]
stability_results["q_2020"] = comp_values_panel[2020]
stability_results["q_change_2000_2020"] = (
    stability_results["q_2020"] - stability_results["q_2000"]
)

# Najbardziej stabilne kraje według najmniejszego zakresu zmian
most_stable_by_range = (
    stability_results
    .sort_values("q_range", ascending=True)
    .head(10)
    .reset_index()
)

print("\nNajbardziej stabilne kraje według zakresu zmian wartości miernika:")
print(most_stable_by_range.round(4).to_string(index=False))


# Najbardziej zmienne kraje według największego zakresu zmian
most_variable_by_range = (
    stability_results
    .sort_values("q_range", ascending=False)
    .head(10)
    .reset_index()
)

print("\nNajbardziej zmienne kraje według zakresu zmian wartości miernika:")
print(most_variable_by_range.round(4).to_string(index=False))


# Najbardziej stabilne kraje według odchylenia standardowego
most_stable_by_std = (
    stability_results
    .sort_values("q_std", ascending=True)
    .head(10)
    .reset_index()
)

print("\nNajbardziej stabilne kraje według odchylenia standardowego wartości miernika:")
print(most_stable_by_std.round(4).to_string(index=False))


# Najbardziej zmienne kraje według odchylenia standardowego
most_variable_by_std = (
    stability_results
    .sort_values("q_std", ascending=False)
    .head(10)
    .reset_index()
)

print("\nNajbardziej zmienne kraje według odchylenia standardowego wartości miernika:")
print(most_variable_by_std.round(4).to_string(index=False))