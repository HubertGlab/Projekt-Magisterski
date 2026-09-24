from pathlib import Path
import requests
import json

import pandas as pd
import numpy as np

from my_funcs.choose_rep import choose_cluster_representatives
from my_funcs.cluster import cluster_variables_by_correlation


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datasets"

dataframe = pd.read_csv(DATA_DIR / "WDI_Original_Dataset.csv")
df = dataframe.copy()


long_df = df.melt(
    id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
    value_vars='2010',
    var_name="Year",
    value_name="Value"
)

pivot_df = long_df.pivot_table(
    index=["Country Name", "Country Code", "Year"],
    columns="Indicator Name",
    values="Value"
).reset_index()

to_remove = [
    'Africa Eastern and Southern',
    'Africa Western and Central',
    'American Samoa',
    'Arab World',
    'Aruba',
    'Bermuda',
    'British Virgin Islands',
    'Caribbean small states',
    'Central Europe and the Baltics',
    'Channel Islands',
    'Curacao',
    'Early-demographic dividend',
    'East Asia & Pacific',
    'East Asia & Pacific (IDA & IBRD countries)',
    'East Asia & Pacific (excluding high income)',
    'Euro area',
    'Europe & Central Asia',
    'Europe & Central Asia (IDA & IBRD countries)',
    'Europe & Central Asia (excluding high income)',
    'European Union',
    'Faroe Islands',
    'Fragile and conflict affected situations',
    'French Polynesia',
    'Gibraltar',
    'Greenland',
    'Guam',
    'Heavily indebted poor countries (HIPC)',
    'High income',
    'Hong Kong SAR, China',
    'IBRD only',
    'IDA & IBRD total',
    'IDA blend',
    'IDA only',
    'IDA total',
    'Isle of Man',
    'Late-demographic dividend',
    'Latin America & Caribbean',
    'Latin America & Caribbean (excluding high income)',
    'Latin America & the Caribbean (IDA & IBRD countries)',
    'Least developed countries: UN classification',
    'Low & middle income',
    'Low income',
    'Lower middle income',
    'Macao SAR, China',
    'Middle East, North Africa, Afghanistan & Pakistan',
    'Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD)',
    'Middle East, North Africa, Afghanistan & Pakistan (excluding high income)',
    'Middle income',
    'New Caledonia',
    'North America',
    'Northern Mariana Islands',
    'OECD members',
    'Other small states',
    'Pacific island small states',
    'Post-demographic dividend',
    'Pre-demographic dividend',
    'Puerto Rico (US)',
    'Sint Maarten (Dutch part)',
    'Small states',
    'South Asia',
    'South Asia (IDA & IBRD)',
    'St. Martin (French part)',
    'Sub-Saharan Africa',
    'Sub-Saharan Africa (IDA & IBRD countries)',
    'Sub-Saharan Africa (excluding high income)',
    'Turks and Caicos Islands',
    'Upper middle income',
    'Virgin Islands (U.S.)',
    'West Bank and Gaza',
    'World',
    'Cayman Islands'
]

pivot_df = pivot_df[~pivot_df['Country Name'].isin(to_remove)]
pivot_df = pivot_df.drop(columns=['Country Code', 'Year'])
pivot_df = pivot_df.loc[:, pivot_df.isnull().mean() < 0.1].copy()


num_df = pivot_df.select_dtypes(include=[np.number]).copy()

def coeff_var(col):
    mean = col.mean(skipna=True)
    std = col.std(skipna=True, ddof=1)
    return std / abs(mean) * 100 if mean != 0 else np.nan

cv = num_df.apply(coeff_var, axis=0)
cv = cv.sort_values()

cv_threshold = 15.05

cols_to_keep = cv[cv >= cv_threshold].index
pivot_df = pivot_df[["Country Name"] + list(cols_to_keep)].copy()

base_url = "https://api.worldbank.org/v2/source/2/indicator"

all_indicators = []
page = 1

while True:
    url = f"{base_url}?format=json&per_page=1000&page={page}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    data = json.loads(r.text)
    meta = data[0]
    indicators = data[1]

    all_indicators.extend(indicators)

    if page >= meta["pages"]:
        break

    page += 1

rows = []

for ind in all_indicators:
    topics_list = ind.get("topics", [])
    rows.append({
        "indicator_code": ind.get("id"),
        "indicator_name": ind.get("name"),
        "unit": ind.get("unit"),
        "source_id": ind.get("source", {}).get("id"),
        "source_name": ind.get("source", {}).get("value"),
        "source_note": ind.get("sourceNote"),
        "source_org": ind.get("sourceOrganization"),
        "topic_ids": "; ".join([str(t.get("id")) for t in topics_list if t.get("id") is not None]),
        "topic_names": "; ".join([t.get("value") for t in topics_list if t.get("value")])
    })

topics = pd.DataFrame(rows)
topics = topics.drop(columns=["indicator_code", "unit", "source_name", "source_id", "source_note", "source_org"])


topic_parts = topics.copy()

topic_parts["topic_names"] = (
    topic_parts["topic_names"]
    .fillna("")
    .astype(str)
    .str.strip()
)

topic_parts["topic_split"] = topic_parts["topic_names"].str.split(";")

topic_parts = topic_parts.explode("topic_split")

topic_parts["topic_split"] = topic_parts["topic_split"].str.strip()

topic_parts = topic_parts[topic_parts["topic_split"] != ""]

topic_counts = (
    topic_parts["topic_split"]
    .value_counts()
    .rename_axis("topic")
    .reset_index(name="count")
)

special_topic_combinations = [
    "Private Sector; Trade",
    "Education ; Gender"
]

topic_count_dict = dict(zip(topic_counts["topic"], topic_counts["count"]))

def choose_topic_final(topic_name):
    if pd.isna(topic_name) or str(topic_name).strip() == "":
        return None
    
    topic_name = str(topic_name).strip()
    
    if topic_name in special_topic_combinations:
        return topic_name
    
    topic_parts = [t.strip() for t in topic_name.split(";")]
    
    if len(topic_parts) == 1:
        return topic_parts[0]
    
    return min(topic_parts, key=lambda t: topic_count_dict.get(t, 0))


topics["topic_names"] = topics["topic_names"].apply(choose_topic_final)


topic_map = topics.set_index("indicator_name")["topic_names"].to_dict()

column_topics = pd.DataFrame({
    "column_name": pivot_df.columns,
    "topic_names": [topic_map.get(col) for col in pivot_df.columns]
})
column_topics["topic_names"] = column_topics["topic_names"].str.strip()


final_cols = []
column_topics.loc[column_topics["topic_names"].isna(), "topic_names"] = "Brak kategorii"

for item in column_topics["topic_names"].unique():
    Agr = column_topics["column_name"][column_topics["topic_names"] == item].to_list()
    if len(Agr) > 1:
        cols = [col for col in Agr if col != "Country Name"]
        corr_matrix, dist_matrix, linkage_matrix, cluster_map = cluster_variables_by_correlation(
        df=pivot_df.drop(columns=["Country Name"]),
        columns=cols,
        method_corr="spearman",
        threshold_corr=0.7
        )

        selected_cols, summary_df, two_vars = choose_cluster_representatives(cluster_map, corr_matrix)
        final_cols.extend(selected_cols)


    

rep_of_two_var = [
    "Political Stability and Absence of Violence/Terrorism: Standard Error",
    "Refugees under the mandate of the UNHCR by country or territory of origin",
    "Out-of-pocket expenditure (% of current health expenditure)",
    "Population ages 30-34, female (% of female population)",
    "Immunization, measles (% of children ages 12-23 months)",
    "Unemployment, youth female (% of female labor force ages 15-24) (modeled ILO estimate)",
    "Employment to population ratio, 15+, male (%) (modeled ILO estimate)",
    "Employment to population ratio, 15+, female (%) (modeled ILO estimate)",
    "Employers, female (% of female employment) (modeled ILO estimate)",
    "Unemployment, female (% of female labor force) (modeled ILO estimate)",
    "Labor force participation rate for ages 15-24, total (%) (modeled ILO estimate)",
    "Unemployment, total (% of total labor force) (modeled ILO estimate)",
    "Adjusted savings: net forest depletion (% of GNI)",
    "PM2.5 pollution, population exposed to levels exceeding WHO Interim Target-3 value (% of total)",
    "Total fisheries production (metric tons)",
    "Methane (CH4) emissions from Building (Energy) (Mt CO2e)",
    "Methane (CH4) emissions from Fugitive Emissions (Energy) (Mt CO2e)",
    "GDP per capita growth (annual %)",
    "Merchandise exports to high-income economies (% of total merchandise exports)",
    "Merchandise imports from high-income economies (% of total merchandise imports)",
    "Merchandise exports to low- and middle-income economies in Sub-Saharan Africa (% of total merchandise exports)",
    "Merchandise exports to economies in the Arab World (% of total merchandise exports)",
    "Merchandise imports from economies in the Arab World (% of total merchandise imports)",
    "Import volume index (2015 = 100)",
    "Export volume index (2015 = 100)",
    "Merchandise exports to low- and middle-income economies in Latin America & the Caribbean (% of total merchandise exports)",
    "Electricity production from renewable sources, excluding hydroelectric (% of total)",
    "Adjusted savings: natural resources depletion (% of GNI)",
    "Mobile cellular subscriptions",
    "Official exchange rate (LCU per US$, period average)",
    "Net trade in goods and services (BoP, current US$)"
]

pivot_df_afclus = pivot_df[["Country Name"] + final_cols + rep_of_two_var].copy()


corr_matrix, dist_matrix, linkage_matrix, cluster_map = cluster_variables_by_correlation(
        df=pivot_df_afclus.drop(columns=["Country Name"]),
        columns=pivot_df_afclus.drop(columns=["Country Name"]).columns.to_list(),
        method_corr="spearman",
        threshold_corr=0.7
        )

selected_cols, summary_df, two_vars = choose_cluster_representatives(cluster_map, corr_matrix)

rep_of_two_var = [
    "GDP deflator (base year varies by country)",
    "Employers, total (% of total employment) (modeled ILO estimate)",
    "Inflation, GDP deflator (annual %)",
    "Political Stability and Absence of Violence/Terrorism: Estimate",
    "PM2.5 pollution, population exposed to levels exceeding WHO Interim Target-1 value (% of total)",
    "Current health expenditure (% of GDP)",
    "Population growth (annual %)",
    "Forest rents (% of GDP)",
    "Employment in industry (% of total employment) (modeled ILO estimate)",
    "Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)",
    "Official exchange rate (LCU per US$, period average)",
    "Adjusted savings: energy depletion (% of GNI)"
]

pivot_df_afclus = pivot_df_afclus[["Country Name"] + selected_cols + rep_of_two_var]


columns_to_drop = [
    "Import volume index (2015 = 100)",
    "Export volume index (2015 = 100)",
    'Aquaculture production (metric tons)',
    'Carbon dioxide (CO2) net fluxes from LULUCF - Deforestation (Mt CO2e)',
    'Carbon dioxide (CO2) net fluxes from LULUCF - Forest Land (Mt CO2e)',
    'Carbon dioxide (CO2) net fluxes from LULUCF - Organic Soil (Mt CO2e)',
    'Carbon dioxide (CO2) net fluxes from LULUCF - Other Land (Mt CO2e)',
    'Claims on central government, etc. (% GDP)',
    'Current account balance (BoP, current US$)',
    'Foreign direct investment, net (BoP, current US$)',
    'GNI: linked series (current LCU)',
    'Government Effectiveness: Number of Sources',
    'Imports of goods and services (BoP, current US$)',
    'Merchandise exports to economies in the Arab World (% of total merchandise exports)',
    'Merchandise exports to high-income economies (% of total merchandise exports)',
    'Merchandise exports to low- and middle-income economies in East Asia & Pacific (% of total merchandise exports)',
    'Merchandise exports to low- and middle-income economies in Europe & Central Asia (% of total merchandise exports)',
    'Merchandise exports to low- and middle-income economies in Latin America & the Caribbean (% of total merchandise exports)',
    'Merchandise exports to low- and middle-income economies in South Asia (% of total merchandise exports)',
    'Merchandise exports to low- and middle-income economies in Sub-Saharan Africa (% of total merchandise exports)',
    'Merchandise exports to low- and middle-income economies outside region (% of total merchandise exports)',
    'Merchandise imports by the reporting economy, residual (% of total merchandise imports)',
    'Merchandise imports from economies in the Arab World (% of total merchandise imports)',
    'Merchandise imports from high-income economies (% of total merchandise imports)',
    'Merchandise imports from low- and middle-income economies in East Asia & Pacific (% of total merchandise imports)',
    'Merchandise imports from low- and middle-income economies in Europe & Central Asia (% of total merchandise imports)',
    'Merchandise imports from low- and middle-income economies in Latin America & the Caribbean (% of total merchandise imports)',
    'Merchandise imports from low- and middle-income economies in South Asia (% of total merchandise imports)',
    'Merchandise imports from low- and middle-income economies outside region (% of total merchandise imports)',
    'Net errors and omissions (BoP, current US$)',
    'Net primary income (Net income from abroad) (current US$)',
    'Net secondary income (Net current transfers from abroad) (current US$)',
    'Personal remittances, received (current US$)',
    'Population, total',
    'Renewable internal freshwater resources, total (billion cubic meters)',
    'Reserves and related items (BoP, current US$)',
    'Total fisheries production (metric tons)',
    'Average precipitation in depth (mm per year)',
    'Carbon dioxide (CO2) net fluxes from LULUCF - Total excluding non-tropical fires (Mt CO2e)',
    'GDP deflator (base year varies by country)',
    'Official exchange rate (LCU per US$, period average)',
    'Population ages 25-29, female (% of female population)',
    'Population ages 30-34, female (% of female population)',
    'Population growth (annual %)',
    'Population density (people per sq. km of land area)',
    'Rural population growth (annual %)',
    'Urban population (% of total population)',
    'Net migration',
    'International migrant stock (% of population)',
    'Cause of death, by injury (% of total)',
    'Computer, communications and other services (% of commercial service imports)',
    #TEST
    "Renewable internal freshwater resources per capita (cubic meters)",
    "Foreign direct investment, net inflows (% of GDP)",
    "Current account balance (% of GDP)",
    "Personal remittances, received (% of GDP)",
    "Mineral rents (% of GDP)",
    "Coal rents (% of GDP)",
    "Forest rents (% of GDP)",
    "Agricultural land (% of land area)",
    "Arable land (% of land area)",
    "Permanent cropland (% of land area)",
    "Arable land (hectares per person)",
    "Forest area (% of land area)",
    "Services, value added (% of GDP)",
    "Industry (including construction), value added (% of GDP)",
    "Employment in industry (% of total employment) (modeled ILO estimate)",
    "Employment in industry, female (% of female employment) (modeled ILO estimate)",
    # TEST 2
    # 'PM2.5 pollution, population exposed to levels exceeding WHO Interim Target-3 value (% of total)',
    # 'PM2.5 pollution, population exposed to levels exceeding WHO Interim Target-1 value (% of total)',
    # 'Electricity production from renewable sources, excluding hydroelectric (% of total)',
    # 'Electricity production from nuclear sources (% of total)',
    # 'Electricity production from oil sources (% of total)',
    # 'Electricity production from natural gas sources (% of total)',
    # 'Inflation, GDP deflator (annual %)',
    # 'Services, value added (annual % growth)',
    # 'Industry (including construction), value added (annual % growth)',
    # 'Agriculture, forestry, and fishing, value added (annual % growth)',
    # 'Livestock production index (2014-2016 = 100)',
    # 'Crop production index (2014-2016 = 100)',
    # 'Fertilizer consumption (kilograms per hectare of arable land)',
    # 'Adjusted savings: net forest depletion (% of GNI)',
    # 'Adjusted savings: consumption of fixed capital (% of GNI)',
    # 'Adjusted savings: energy depletion (% of GNI)',
    # 'Mortality rate attributed to unintentional poisoning (per 100,000 population)',
    # 'Suicide mortality rate (per 100,000 population)',
    # 'Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)'
    # TEST 3
    # 'PM2.5 pollution, population exposed to levels exceeding WHO Interim Target-3 value (% of total)',
    # 'PM2.5 pollution, population exposed to levels exceeding WHO Interim Target-1 value (% of total)',
    # 'Electricity production from renewable sources, excluding hydroelectric (% of total)',
    # 'Electricity production from nuclear sources (% of total)',
    # 'Electricity production from oil sources (% of total)',
    # 'Electricity production from natural gas sources (% of total)',
    # 'Inflation, GDP deflator (annual %)',
    # 'Death rate, crude (per 1,000 people)',
    # 'Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)',
    # 'Adjusted savings: energy depletion (% of GNI)'
]

pivot_df_afclus = pivot_df_afclus.drop(columns=columns_to_drop)
chosen_columns = pivot_df_afclus.columns.to_list()

