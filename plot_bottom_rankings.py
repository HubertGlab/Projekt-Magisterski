import matplotlib.pyplot as plt
import pandas as pd

from my_funcs.hellwig import hellwig_manhattan
from preprocessing.WDI_hellwig import df_panel, stim_or_destim

years = [str(y) for y in range(2000, 2021)]

results = []

for year in years:
    df_year = (
        df_panel[df_panel["Year"] == year]
        .drop(columns=["Year"])
        .set_index("Country Name")
    )

    q, score, deviations, _, _ = hellwig_manhattan(
        df_year,
        stim_or_destim,
    )

    year_results = q.reset_index()
    year_results.columns = ["Country Name", "q"]
    year_results["Year"] = int(year)

    # ranking w danym roku
    # najwyższa wartość q = najlepsza pozycja
    year_results["rank"] = year_results["q"].rank(
        ascending=False,
        method="min"
    ).astype(int)

    results.append(year_results)


hellwig_results = pd.concat(results, ignore_index=True)


# Bottom 10 krajów z lat 2000, 2010 i 2020
bottom10_2000 = (
    hellwig_results[hellwig_results["Year"] == 2000]
    .sort_values("rank", ascending=False)
    .head(10)["Country Name"]
    .tolist()
)

bottom10_2010 = (
    hellwig_results[hellwig_results["Year"] == 2010]
    .sort_values("rank", ascending=False)
    .head(10)["Country Name"]
    .tolist()
)

bottom10_2020 = (
    hellwig_results[hellwig_results["Year"] == 2020]
    .sort_values("rank", ascending=False)
    .head(10)["Country Name"]
    .tolist()
)


# Unikalna lista krajów, które znalazły się w bottom 10
# przynajmniej w jednym z analizowanych lat
selected_countries = list(dict.fromkeys(bottom10_2000 + bottom10_2010 + bottom10_2020))


# Dane dla wybranych krajów w całym okresie 2000–2020
plot_data = hellwig_results[
    (hellwig_results["Country Name"].isin(selected_countries)) &
    (hellwig_results["Year"].between(2000, 2020))
].copy()


plt.figure(figsize=(13, 8))

# tab20 w kolejności: najpierw mocniejsze, bardziej kontrastowe kolory
tab20 = plt.get_cmap("tab20").colors

color_order = (
    list(range(0, 20, 2)) +
    list(range(1, 20, 2))
)

colors = [tab20[i] for i in color_order]


for i, country in enumerate(selected_countries):
    country_data = (
        plot_data[plot_data["Country Name"] == country]
        .sort_values("Year")
    )

    plt.plot(
        country_data["Year"],
        country_data["rank"],
        linewidth=2.2,
        color=colors[i % len(colors)],
        label=country
    )


# Próg bottom 10
# Zakładamy, że liczba krajów w rankingu jest stała w latach.
n_countries = hellwig_results[hellwig_results["Year"] == 2000]["rank"].max()
bottom10_threshold = n_countries - 9

plt.axhline(
    y=bottom10_threshold,
    color="red",
    linestyle="--",
    linewidth=1.8,
    alpha=0.8,
    label="Próg bottom 10"
)


plt.title("Pozycje rankingowe w latach 2000–2020 krajów należących do ostatniej dziesiątki w latach 2000, 2010 lub 2020")
plt.xlabel("Rok")
plt.ylabel("Pozycja w rankingu")

plt.xticks(range(2000, 2021, 2))

# zostawiamy odwróconą oś, żeby najlepsze miejsca były u góry,
# a najgorsze na dole
plt.gca().invert_yaxis()

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()