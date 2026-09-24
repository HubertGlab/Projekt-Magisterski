from pathlib import Path

import pandas as pd

from my_funcs.hellwig import hellwig_compensational
from preprocessing.WDI_preprocessing import to_remove, chosen_columns

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datasets"

dataframe = pd.read_csv(DATA_DIR / "WDI_Original_Dataset.csv")
df = dataframe.copy()

print("START PROGRAMU")

years = [str(y) for y in range(2000, 2021)]

long_df = df.melt(
    id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
    value_vars=years,
    var_name="Year",
    value_name="Value"
)

pivot_df = long_df.pivot_table(
    index=["Country Name", "Country Code", "Year"],
    columns="Indicator Name",
    values="Value"
).reset_index()

pivot_df = pivot_df[~pivot_df['Country Name'].isin(to_remove)]
pivot_df = pivot_df.drop(columns=['Country Code'])
pivot_df_afclus = pivot_df[chosen_columns + ["Year"]].copy()

meta = pivot_df_afclus[["Country Name", "Year"]].copy()
X = pivot_df_afclus.drop(columns=["Country Name", "Year"]).copy()

# scaler = StandardScaler()
scaler = MinMaxScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

imputer = KNNImputer(n_neighbors=5, weights="distance")
X_imputed = pd.DataFrame(
    imputer.fit_transform(X_scaled),
    columns=X_scaled.columns,
    index=X_scaled.index
)

df_panel = pd.concat(
    [meta.reset_index(drop=True), X_imputed.reset_index(drop=True)],
    axis=1
)
df_fin = (
    df_panel[df_panel["Year"] == "2010"]
    .drop(columns=["Year"])
    .set_index("Country Name")
)

stim_cols = [
    "Import volume index (2015 = 100)",
    "Export volume index (2015 = 100)",
    "Tuberculosis treatment success rate (% of new cases)",
    "Employment in industry, female (% of female employment) (modeled ILO estimate)",
    "Computer, communications and other services (% of commercial service exports)",
    "Foreign direct investment, net inflows (% of GDP)",
    "Current account balance (% of GDP)",
    "Personal remittances, received (% of GDP)",
    "Merchandise trade (% of GDP)",
    "Electricity production from renewable sources, excluding hydroelectric (% of total)",
    "Compulsory education, duration (years)",
    "Services, value added (% of GDP)",
    "Immunization, measles (% of children ages 12-23 months)",
    "Electricity production from nuclear sources (% of total)",
    "Adjusted savings: education expenditure (% of GNI)",
    "Statistical performance indicators (SPI): Pillar 3 data products score  (scale 0-100)",
    "Industry (including construction), value added (% of GDP)",
    "Statistical performance indicators (SPI): Pillar 1 data use score (scale 0-100)",
    "Services, value added (annual % growth)",
    "GDP per capita growth (annual %)",
    "Industry (including construction), value added (annual % growth)",
    "Livestock production index (2014-2016 = 100)",
    "Agricultural land (% of land area)",
    "Arable land (% of land area)",
    "Permanent cropland (% of land area)",
    "Arable land (hectares per person)",
    "Proportion of seats held by women in national parliaments (%)",
    "Employment to population ratio, 15+, total (%) (modeled ILO estimate)",
    "Ratio of female to male labor force participation rate (%) (modeled ILO estimate)",
    "Forest area (% of land area)",
    "Renewable internal freshwater resources per capita (cubic meters)",
    "Renewable electricity output (% of total electricity output)",
    "Agriculture, forestry, and fishing, value added (annual % growth)",
    "Crop production index (2014-2016 = 100)",
    "Employers, total (% of total employment) (modeled ILO estimate)",
    "Political Stability and Absence of Violence/Terrorism: Estimate",
    "Current health expenditure (% of GDP)",
    "Employment in industry (% of total employment) (modeled ILO estimate)"
]

destim_cols = [
    "Energy intensity level of primary energy (MJ/$2017 PPP GDP)",
    "Inflation, consumer prices (annual %)",
    "Adjusted savings: net forest depletion (% of GNI)",
    "Out-of-pocket expenditure (% of current health expenditure)",
    "PM2.5 air pollution, population exposed to levels exceeding WHO guideline value (% of total)",
    "PM2.5 pollution, population exposed to levels exceeding WHO Interim Target-3 value (% of total)",
    "Adjusted savings: natural resources depletion (% of GNI)",
    "Mortality from CVD, cancer, diabetes or CRD between exact ages 30 and 70 (%)",
    "Mortality rate attributed to unintentional poisoning (per 100,000 population)",
    "Mortality rate, under-5 (per 1,000 live births)",
    "Mortality caused by road traffic injury (per 100,000 population)",
    "Prevalence of anemia among women of reproductive age (% of women ages 15-49)",
    "Fertilizer consumption (kilograms per hectare of arable land)",
    "Adjusted savings: consumption of fixed capital (% of GNI)",
    "Suicide mortality rate (per 100,000 population)",
    "Total greenhouse gas emissions excluding LULUCF (% change from 1990)",
    "Mineral rents (% of GDP)",
    "Coal rents (% of GDP)",
    "Electricity production from coal sources (% of total)",
    "Electricity production from oil sources (% of total)",
    "Electricity production from natural gas sources (% of total)",
    "Death rate, crude (per 1,000 people)",
    "Unemployment, female (% of female labor force) (modeled ILO estimate)",
    "Share of youth not in education, employment or training, total (% of youth population)  (modeled ILO estimate)",
    "Carbon intensity of GDP (kg CO2e per constant 2015 US$ of GDP)",
    "Inflation, GDP deflator (annual %)",
    "PM2.5 pollution, population exposed to levels exceeding WHO Interim Target-1 value (% of total)",
    "Forest rents (% of GDP)",
    "Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)",
    "Adjusted savings: energy depletion (% of GNI)"
]

stim_cols = [col for col in stim_cols if col in df_fin.columns]
destim_cols = [col for col in destim_cols if col in df_fin.columns]

stim_or_destim = pd.DataFrame({
    "variable": stim_cols + destim_cols,
    "type": ["stim"] * len(stim_cols) + ["destim"] * len(destim_cols)
})

df = df_fin.copy()

q, score, deviations, pattern, D0 = hellwig_compensational(df, stim_or_destim)
q.sort_values(ascending=False)

stim_or_destim = stim_or_destim
chosen_columns = df.columns.to_list()
pattern = pattern
D0 = D0

print("KONIEC PROGRAMU")