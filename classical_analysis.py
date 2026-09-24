from pathlib import Path

import pandas as pd

from my_funcs.hellwig import hellwig_manhattan
from preprocessing.WDI_hellwig import df_panel, stim_or_destim

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 1. Obliczenie wyników Hellwiga i pozycji rankingowych
#    dla każdego roku osobno
# ------------------------------------------------------------

years = [str(y) for y in range(2000, 2021)]

results = []
distance_results = []

for year in years:
    df_year = (
        df_panel[df_panel["Year"] == year]
        .drop(columns=["Year"])
        .set_index("Country Name")
    )

    q, score, deviations, _, _ = hellwig_manhattan(
        df_year,
        stim_or_destim
    )

    year_distances = score.reset_index()
    year_distances.columns = ["Country Name", "distance"]
    year_distances["Year"] = int(year)

    distance_results.append(year_distances)

    year_results = q.reset_index()
    year_results.columns = ["Country Name", "q"]
    year_results["Year"] = int(year)

    year_results["rank"] = (
        year_results["q"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    results.append(year_results)


hellwig_results = pd.concat(results, ignore_index=True)

classical_distances = pd.concat(
    distance_results,
    ignore_index=True
)

classical_distances = (
    classical_distances
    .sort_values(["Year", "Country Name"])
    .reset_index(drop=True)
)

classical_distances.to_csv(
    RESULTS_DIR / "05_classical_distances_2000_2020.csv",
    index=False
)

# ------------------------------------------------------------
# Eksport 01: pełne wyniki klasycznej metody, 2000–2020
# ------------------------------------------------------------

classical_rankings_full = (
    hellwig_results[
        ["Country Name", "Year", "q", "rank"]
    ]
    .rename(columns={
        "q": "q_classical",
        "rank": "rank_classical"
    })
    .sort_values(
        ["Year", "rank_classical", "Country Name"]
    )
    .reset_index(drop=True)
)

classical_rankings_full.to_csv(
    RESULTS_DIR / "01_classical_rankings_2000_2020.csv",
    index=False
)


# ------------------------------------------------------------
# Eksport 03: pełne rozwinięcie tabel 3 i 4
# lata 2000, 2010 i 2020
# ------------------------------------------------------------

classical_table_parts = []

for year in [2000, 2010, 2020]:
    year_table = (
        hellwig_results[
            hellwig_results["Year"] == year
        ]
        .sort_values(["rank", "Country Name"])
        [["Country Name", "q"]]
        .reset_index(drop=True)
        .rename(columns={
            "Country Name": f"Country_{year}",
            "q": f"q_{year}"
        })
    )

    classical_table_parts.append(year_table)

classical_rankings_3years = pd.concat(
    classical_table_parts,
    axis=1
)

classical_rankings_3years.insert(
    0,
    "Position",
    range(1, len(classical_rankings_3years) + 1)
)

for column in ["q_2000", "q_2010", "q_2020"]:
    classical_rankings_3years[column] = (
        classical_rankings_3years[column].round(4)
    )

classical_rankings_3years.to_csv(
    RESULTS_DIR / "03_classical_rankings_2000_2010_2020.csv",
    index=False
)


# ------------------------------------------------------------
# 2. Zmiana pozycji rankingowej między 2000 a 2020 rokiem
# ------------------------------------------------------------

rank_change_2000_2020 = (
    hellwig_results[
        hellwig_results["Year"].isin([2000, 2020])
    ]
    .pivot(index="Country Name", columns="Year", values="rank")
    .reset_index()
    .rename(columns={
        2000: "rank_2000",
        2020: "rank_2020"
    })
)

rank_change_2000_2020["change_2000_2020"] = (
    rank_change_2000_2020["rank_2000"] 
    - rank_change_2000_2020["rank_2020"]
)

# ------------------------------------------------------------
# 3. Największe awanse
# ------------------------------------------------------------

biggest_improvements = (
    rank_change_2000_2020
    .sort_values("change_2000_2020", ascending=False)
    .head(10)
)

print("\nNajwiększe awanse w rankingu 2000–2020:")
print(biggest_improvements.to_string(index=False))


# ------------------------------------------------------------
# 4. Największe spadki
# ------------------------------------------------------------

biggest_declines = (
    rank_change_2000_2020
    .sort_values("change_2000_2020", ascending=True)
    .head(10)
)

print("\nNajwiększe spadki w rankingu 2000–2020:")
print(biggest_declines.to_string(index=False))


# ------------------------------------------------------------
# 5. Najbardziej stabilne pozycje
# ------------------------------------------------------------

stable_countries = rank_change_2000_2020.copy()

stable_countries["abs_change"] = (
    stable_countries["change_2000_2020"]
    .abs()
)

stable_countries = (
    stable_countries
    .sort_values("abs_change", ascending=True)
    .head(10)
)

print("\nNajbardziej stabilne pozycje w rankingu 2000–2020:")
print(stable_countries.to_string(index=False))


# ------------------------------------------------------------
# 6. Stabilność pozycji rankingowych w latach 2000–2020
# ------------------------------------------------------------

ranking_stability = (
    hellwig_results
    .groupby("Country Name")["rank"]
    .agg(
        best_rank="min",
        worst_rank="max",
        mean_rank="mean",
        std_rank="std"
    )
    .reset_index()
)

# rozstęp pozycji rankingowej
ranking_stability["rank_range"] = (
    ranking_stability["worst_rank"] - ranking_stability["best_rank"]
)

# Zaokrąglenie wartości dla czytelniejszego wydruku
ranking_stability["mean_rank"] = ranking_stability["mean_rank"].round(2)
ranking_stability["std_rank"] = ranking_stability["std_rank"].round(2)


# ------------------------------------------------------------
# Kraje o najbardziej stabilnych pozycjach
# ------------------------------------------------------------

most_stable_countries = (
    ranking_stability
    .sort_values(["std_rank", "rank_range", "mean_rank"], ascending=[True, True, True])
    .head(15)
)

print("\nNajbardziej stabilne pozycje rankingowe w latach 2000–2020:")
print(most_stable_countries.to_string(index=False))


# ------------------------------------------------------------
# Kraje o najmniej stabilnych pozycjach
# ------------------------------------------------------------

least_stable_countries = (
    ranking_stability
    .sort_values(["std_rank", "rank_range"], ascending=[False, False])
    .head(15)
)

print("\nNajmniej stabilne pozycje rankingowe w latach 2000–2020:")
print(least_stable_countries.to_string(index=False))


# ------------------------------------------------------------
# Pełna tabela stabilności, opcjonalnie posortowana
# ------------------------------------------------------------

ranking_stability_sorted = (
    ranking_stability
    .sort_values(["std_rank", "rank_range", "mean_rank"], ascending=[True, True, True])
)

print("\nPełna tabela stabilności pozycji rankingowych:")
print(ranking_stability_sorted.to_string(index=False))