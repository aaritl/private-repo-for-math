# HiMCM 2026 — Best Places to Live (simple version)
#
# BIG IDEA: narrow the whole world down in 3 rounds, like a tournament.
#   Round 1: score every country        -> keep the best 30
#   Round 2: score their regions        -> keep the best 5 in each country
#   Round 3: score their cities         -> keep the best 5 in each region
#   Bonus:   group the final cities into "types" with K-Means (machine learning)
#
# HOW A PLACE GETS A SCORE (same in every round):
#   0. Cap extreme values (outliers), so one extreme place can't distort the scale.
#   1. Rescale each criterion to 0..1, where 1 = best.
#   2. Give each criterion a weight (how important it is), using the entropy method.
#   3. Score = weighted average of the criteria the place has data for.

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


# ---------------------------------------------------------------
# STEP 1: SETTINGS  (change these)
# ---------------------------------------------------------------

COUNTRIES_TO_KEEP = 30     # round 1 keeps this many countries
KEEP_PER_PARENT = 5        # rounds 2 and 3 keep this many per country / per region
MIN_DATA = 0.60            # a place needs data for 60% of the importance to be ranked
NUMBER_OF_TYPES = 3        # K-Means: how many groups to split the final cities into
OUTLIER_CAP = 0.05         # cap the lowest 5% and highest 5% of each criterion (0 = turn off)

# For each round: criterion name -> is "higher" or "lower" better?
# Each name is a file: data/input/<round>/<name>.csv  with columns geo_id, value
CRITERIA = {
    "tier1_country": {
        "physicians_per_1000": "higher",
        "health_expenditure_per_capita_ppp": "higher",
        "secondary_completion_rate": "higher",
        "tertiary_education_rate": "higher",
        "gdp_per_capita_ppp": "higher",
        "unemployment_rate": "lower",
        "poverty_rate": "lower",
        "median_household_income_ppp": "higher",
        "housing_cost_burden": "lower",
    },
    "tier2_region": {
        "regional_hospital_beds_per_1000": "higher",
        "regional_physicians_per_1000": "higher",
        "regional_unemployment_rate": "lower",
        "regional_median_income_ppp": "higher",
        "regional_housing_cost_burden": "lower",
        "regional_homicide_rate": "lower",
        "regional_pm25": "lower",
        "regional_green_space_access": "higher",
    },
    "tier3_city": {
        "city_healthcare_access": "higher",
        "public_transport_access": "higher",
        "city_homicide_rate": "lower",
        "city_pm25": "lower",
        "city_green_space_per_capita": "higher",
        "restaurants_per_10000": "higher",
        "cultural_venues_per_10000": "higher",
        "diversity_index": "higher",
    },
}

DATA = Path("data")

# Read files as plain text, and only treat EMPTY cells as missing.
# (Otherwise pandas reads "NA" — the code for Namibia — as "missing".)
TEXT = {"dtype": str, "keep_default_na": False, "na_values": [""]}


# ---------------------------------------------------------------
# STEP 2: MAKE FOLDERS AND BLANK CSV FILES FOR YOUR DATA
# ---------------------------------------------------------------

(DATA / "places").mkdir(parents=True, exist_ok=True)
(DATA / "output").mkdir(parents=True, exist_ok=True)

for round_name in CRITERIA:
    folder = DATA / "input" / round_name
    folder.mkdir(parents=True, exist_ok=True)
    for name in CRITERIA[round_name]:
        file = folder / (name + ".csv")
        if not file.exists():  # never overwrite data you already typed in
            pd.DataFrame(columns=["geo_id", "value", "year", "source"]).to_csv(file, index=False)

print("Put your data in the CSV files inside:", DATA / "input")


# ---------------------------------------------------------------
# STEP 3: DOWNLOAD THE LIST OF EVERY COUNTRY, REGION, AND CITY
# ---------------------------------------------------------------
# From GeoNames (free). Every place gets:
#   geo_id     its ID code:  "US" (country), "US.MI" (region), "4990729" (city)
#   place_name its name
#   parent     the place one level up — this is how the rounds connect

def download(filename):
    file = DATA / "places" / filename
    if not file.exists():  # only download once
        url = "https://download.geonames.org/export/dump/" + filename
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        file.write_bytes(response.content)
    return file

def read_geonames(file, column_numbers, column_names, **extra):
    table = pd.read_csv(file, sep="\t", header=None, usecols=column_numbers, **TEXT, **extra)
    table.columns = column_names
    return table

countries = read_geonames(download("countryInfo.txt"), [0, 4], ["geo_id", "place_name"], comment="#")
countries["parent"] = "world"

regions = read_geonames(download("admin1CodesASCII.txt"), [0, 1], ["geo_id", "place_name"])
regions["parent"] = regions["geo_id"].str.split(".").str[0]   # "US.MI" -> "US"

with zipfile.ZipFile(download("cities1000.zip")) as z:
    cities = read_geonames(z.open("cities1000.txt"), [0, 1, 8, 10],
                           ["geo_id", "place_name", "country", "state"])
cities["parent"] = cities["country"] + "." + cities["state"]  # "US" + "MI" -> "US.MI"

print(len(countries), "countries,", len(regions), "regions,", len(cities), "cities")


# ---------------------------------------------------------------
# STEP 4: THE MODEL
# ---------------------------------------------------------------

def load_data(places, round_name):
    """Add one column per criterion to the list of places."""
    table = places.copy()
    for name in CRITERIA[round_name]:
        data = pd.read_csv(DATA / "input" / round_name / (name + ".csv"), **TEXT)
        data["geo_id"] = data["geo_id"].str.strip()
        data["value"] = pd.to_numeric(data["value"], errors="coerce")  # "n/a" -> missing
        data = data.dropna(subset=["value"])
        # If a place is listed more than once, keep the newest year.
        if "year" in data.columns:
            data["year"] = pd.to_numeric(data["year"], errors="coerce")
            data = data.sort_values("year", ascending=False)
        data = data.drop_duplicates("geo_id")
        # Look up each place's value by its geo_id (blank if it has none).
        table[name] = table["geo_id"].map(data.set_index("geo_id")["value"])
    return table


def entropy_weights(scaled):
    """
    How important is each criterion? The ENTROPY METHOD lets the data decide:
    a criterion where places are very different gets a big weight,
    a criterion where every place is about the same gets a weight near 0.

      p = each place's share of the column's total
      e = -sum(p * ln p) / ln(number of places)    (0 = very different, 1 = all the same)
      weight = (1 - e), then scaled so all weights add up to 1
    """
    x = scaled.dropna(how="all")       # only places that have some data
    x = x.fillna(x.mean())             # fill gaps with the average (only for the weights)
    p = (x + 1e-12) / (x + 1e-12).sum()  # tiny +1e-12 so we never take log(0)
    e = -(p * np.log(p)).sum() / np.log(len(x))
    d = (1 - e).clip(lower=0)
    if len(x) < 2 or d.sum() == 0:     # nothing to compare: give every criterion the same weight
        return pd.Series(1 / len(d), index=d.index)
    return d / d.sum()


def score_places(table, round_name, weights=None):
    """Give every place a score from 0 to 1 (1 = best) and a rank."""
    directions = CRITERIA[round_name]
    columns = [name for name in directions if table[name].notna().any()]  # skip empty files
    if not columns:
        raise ValueError("No data yet! Fill in the CSV files in " + str(DATA / "input" / round_name))

    # 4a. Cap outliers (winsorizing). Same idea as Huber loss: extreme values still
    #     count, but they can't pull everything else around.
    #     Example: GDP of $140k vs most countries at $2k-$60k. Without the cap,
    #     the $140k country sets the top of the scale and squeezes everyone else near 0.
    #     With the cap, anything above the 95th percentile is treated as the 95th percentile.
    values = table[columns]
    lowest_allowed = values.quantile(OUTLIER_CAP)
    highest_allowed = values.quantile(1 - OUTLIER_CAP)
    values = values.clip(lowest_allowed, highest_allowed, axis=1)

    # 4b. Rescale every criterion to 0..1 so dollars, percents, etc. can be compared.
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(values),
                          columns=columns, index=table.index)
    for name in columns:
        if directions[name] == "lower":
            scaled[name] = 1 - scaled[name]   # flip it, so 1 is always the best

    # 4c. Weights: calculate them, unless they were given (the what-if test keeps them fixed).
    if weights is None:
        weights = entropy_weights(scaled)

    # 4d. Score = weighted average, using ONLY the criteria the place has data for.
    #     (A missing value is skipped — it does not count as 0.)
    table = table.copy()
    table["data_share"] = (scaled.notna() * weights).sum(axis=1)  # importance we have data for
    table["score"] = (scaled * weights).sum(axis=1) / table["data_share"]

    # 4e. Rank places that have enough data (1 = best). The rest get no rank.
    enough = table["data_share"] >= MIN_DATA
    table["rank"] = table.loc[enough, "score"].rank(ascending=False, method="min")
    return table.sort_values("score", ascending=False), weights, scaled


def run_round(round_name, places, how_many, per_parent):
    """Score one round and keep the winners."""
    table, weights, scaled = score_places(load_data(places, round_name), round_name)
    ranked = table[table["rank"].notna()]
    if per_parent:
        winners = ranked.groupby("parent").head(how_many)  # top N inside each parent
    else:
        winners = ranked.head(how_many)                     # top N overall

    ranked.to_csv(DATA / "output" / (round_name + "_all_scores.csv"), index=False)
    winners.to_csv(DATA / "output" / (round_name + "_winners.csv"), index=False)

    print("\n=====", round_name, "=====")
    print(len(ranked), "places ranked,", len(winners), "move on")
    print("Weights:", weights.round(3).to_dict())
    print(winners[["geo_id", "place_name", "score", "rank"]].head(15).to_string(index=False))
    return table, winners, weights, scaled


# ---------------------------------------------------------------
# STEP 5: RUN THE 3 ROUNDS
# ---------------------------------------------------------------

country_table, best_countries, country_weights, _ = run_round(
    "tier1_country", countries, COUNTRIES_TO_KEEP, per_parent=False)

region_table, best_regions, region_weights, _ = run_round(
    "tier2_region", regions[regions["parent"].isin(best_countries["geo_id"])],
    KEEP_PER_PARENT, per_parent=True)

city_table, best_cities, city_weights, city_scaled = run_round(
    "tier3_city", cities[cities["parent"].isin(best_regions["geo_id"])],
    KEEP_PER_PARENT, per_parent=True)


# ---------------------------------------------------------------
# STEP 6 (BONUS MACHINE LEARNING): GROUP THE FINAL CITIES WITH K-MEANS
# ---------------------------------------------------------------
# K-Means puts cities with similar 0..1 values in the same group.
# It needs no "right answers" to learn from — it just finds cities that are alike.
# Example result: one group is "clean and safe", another is "lively but polluted".

final = city_scaled.loc[best_cities.index]
final = final.fillna(final.mean())            # K-Means cannot handle blanks
groups = min(NUMBER_OF_TYPES, len(final))     # can't make more groups than cities

kmeans = KMeans(n_clusters=groups, n_init=10, random_state=0)  # random_state = same result every run
best_cities = best_cities.copy()
best_cities["type"] = kmeans.fit_predict(final)

print("\n===== City types (K-Means) =====")
print("Average 0..1 value of each criterion in each group (1 = best):")
summary = final.groupby(best_cities["type"]).mean().round(2)
summary["average_score"] = best_cities.groupby("type")["score"].mean().round(3)
summary["number_of_cities"] = best_cities["type"].value_counts()
print(summary.to_string())

best_cities.to_csv(DATA / "output" / "final_cities_with_types.csv", index=False)


# ---------------------------------------------------------------
# STEP 7: WHAT-IF TEST (how stable is a ranking?)
# ---------------------------------------------------------------
# "If this city had X restaurants instead, what would its score and rank be?"
# The weights stay the same, so only that one number changes.

WHAT_IF_CITY = None     # put a city's geo_id here, e.g. "2942819", to run this test
WHAT_IF_CRITERION = "restaurants_per_10000"
WHAT_IF_VALUES = [20, 30, 40, 50, 60]

if WHAT_IF_CITY is not None:
    results = []
    for value in WHAT_IF_VALUES:
        changed = city_table.copy()
        changed.loc[changed["geo_id"] == WHAT_IF_CITY, WHAT_IF_CRITERION] = value
        new_table, _, _ = score_places(changed, "tier3_city", city_weights)
        city = new_table[new_table["geo_id"] == WHAT_IF_CITY].iloc[0]
        results.append({"value": value, "score": city["score"], "rank": city["rank"]})

    results = pd.DataFrame(results)
    print("\n===== What-if test for", WHAT_IF_CITY, "=====")
    print(results.to_string(index=False))
    results.to_csv(DATA / "output" / "what_if.csv", index=False)

    plt.plot(results["value"], results["score"], marker="o")
    plt.xlabel(WHAT_IF_CRITERION)
    plt.ylabel("score (1 = best)")
    plt.title("What if " + WHAT_IF_CITY + " had a different " + WHAT_IF_CRITERION + "?")
    plt.grid(True)
    plt.show()
