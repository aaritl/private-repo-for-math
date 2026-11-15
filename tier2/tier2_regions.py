"""Tier 2 = regions INSIDE the best Tier 1 countries (funnel), using OECD regional data (TL2 regions).

Same model as himcm26_simple.py: winsorize 5/95 -> MinMax 0-1 (flip "lower") -> entropy weights
-> weighted average over criteria a region has -> rank regions with >= 60% data -> keep top 5 per country.
usage: python tier2_regions.py <oecd_raw dir> <countryInfo.txt> <tier1 ranking csv> <results dir>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

RAW, GEONAMES, TIER1, OUT = map(Path, sys.argv[1:5])
COUNTRIES_TO_KEEP = 30   # same as the model's Step 1 setting
KEEP_PER_PARENT = 5
MIN_DATA = 0.60
OUTLIER_CAP = 0.05
FIRST_YEAR = 2015        # ignore values older than this

# criterion: (direction, folder, description)
CRITERIA = {
    "homicide_rate":       ("lower",  "Crime",          "Intentional homicides per 100,000"),
    "vehicle_theft_rate":  ("lower",  "Crime",          "Motor vehicle thefts per 100,000"),
    "pm25_exposure":       ("lower",  "Environment",    "Mean population exposure to PM2.5 (ug/m3)"),
    "tertiary_education":  ("higher", "Education",      "Adults 25-64 with tertiary education (%)"),
    "transport_deaths":    ("lower",  "Transportation", "Deaths from transport accidents per 100,000"),
}

def latest(df, value="OBS_VALUE"):
    """Most recent year >= FIRST_YEAR for each region."""
    df = df[(df.TIME_PERIOD >= FIRST_YEAR) & df[value].notna()]
    return df.sort_values("TIME_PERIOD").groupby("REF_AREA").last()

# ---------- 1. which countries go forward from Tier 1 ----------
geo = pd.read_csv(GEONAMES, sep="\t", comment="#", header=None, usecols=[0, 1], names=["iso2", "iso3"],
                  dtype=str, keep_default_na=False)
ISO3 = dict(zip(geo.iso2, geo.iso3))
t1 = pd.read_csv(TIER1, keep_default_na=False, na_values=[""])
best = t1.sort_values("rank").head(COUNTRIES_TO_KEEP).copy()
best["iso3"] = best.geo_id.map(ISO3)

# ---------- 2. load OECD regional data ----------
soc = pd.read_csv(RAW / "DSD_REG_SOC@DF_SAFETY.csv", low_memory=False)
soc = soc[soc.TERRITORIAL_LEVEL == "TL2"]
edu = pd.read_csv(RAW / "DSD_REG_EDU@DF_ATTAIN.csv", low_memory=False)
edu = edu[(edu.TERRITORIAL_LEVEL == "TL2") & (edu.EDUCATION_LEV == "ISCED11_5T8") & (edu.AGE == "Y25T64")
          & (edu.SEX == "_T") & (edu.STATISTICAL_OPERATION == "MEAN")]
air = pd.read_csv(RAW / "air2.csv", low_memory=False)
air = air[(air.DD_DIM == "_T") & (air.POLLUTANT == "PM2_5")]   # region totals (some countries tagged _Z instead of TL2)

parts = {
    "homicide_rate":      latest(soc[soc.MEASURE == "HOMIC"]),
    "vehicle_theft_rate": latest(soc[(soc.MEASURE == "VEH_THEFT") & (soc.UNIT_MEASURE == "CS_10P5PS")]),
    "pm25_exposure":      latest(air),
    "tertiary_education": latest(edu),
    "transport_deaths":   latest(soc[soc.MEASURE == "MORT_ICDV_CRUDE_RATIO"]),
}
table = pd.DataFrame({k: v["OBS_VALUE"] for k, v in parts.items()})
years = {k: f"{int(v.TIME_PERIOD.min())}-{int(v.TIME_PERIOD.max())}" for k, v in parts.items()}

# region -> country (iso3) from the OECD rows themselves
country_of = pd.concat([soc[["REF_AREA", "COUNTRY"]], edu[["REF_AREA", "COUNTRY"]]]).drop_duplicates("REF_AREA")
country_of = country_of.set_index("REF_AREA").COUNTRY
table["iso3"] = country_of.reindex(table.index)
# air-only regions: country from the same code prefix used by other regions of that country
prefix = {code[:2]: c for code, c in country_of.items()}
table["iso3"] = table["iso3"].fillna(pd.Series(table.index.str[:2], index=table.index).map(prefix))

names = pd.read_csv(RAW / "soc_labels.csv", low_memory=False, usecols=["REF_AREA", "Reference area"])
NAME = names.drop_duplicates("REF_AREA").set_index("REF_AREA")["Reference area"]

# ---------- 3. keep only regions of the Tier 1 winners ----------
table = table[table.iso3.isin(best.iso3)].copy()
iso3_to_name = dict(zip(best.iso3, best.country))
table["country"] = table.iso3.map(iso3_to_name)
table["region"] = [NAME.get(c, c) for c in table.index]
table = table.rename_axis("region_code").reset_index()
no_regions = best[~best.iso3.isin(table.iso3)]

# ---------- 4. the model (same steps as himcm26_simple.py) ----------
def entropy_weights(scaled):
    x = scaled.dropna(how="all")
    x = x.fillna(x.mean())
    p = (x + 1e-12) / (x + 1e-12).sum()
    e = -(p * np.log(p)).sum() / np.log(len(x))
    d = (1 - e).clip(lower=0)
    return d / d.sum()

def score(table, weights=None):
    cols = list(CRITERIA)
    v = table[cols]
    v = v.clip(v.quantile(OUTLIER_CAP), v.quantile(1 - OUTLIER_CAP), axis=1)      # 4a winsorize
    s = pd.DataFrame(MinMaxScaler().fit_transform(v), columns=cols, index=table.index)  # 4b
    for c in cols:
        if CRITERIA[c][0] == "lower":
            s[c] = 1 - s[c]
    w = entropy_weights(s) if weights is None else weights / weights.sum()         # 4c
    t = table.copy()
    t["data_share"] = (s.notna() * w).sum(axis=1)
    t["score"] = (s * w).sum(axis=1) / t["data_share"]                              # 4d
    ok = t["data_share"] >= MIN_DATA
    t["rank_overall"] = t.loc[ok, "score"].rank(ascending=False, method="min")      # 4e
    t["rank_in_country"] = t.loc[ok].groupby("iso3")["score"].rank(ascending=False, method="min")
    return t.sort_values("score", ascending=False), w

ranked, weights = score(table)
ranked = ranked[ranked.rank_overall.notna()]
winners = ranked[ranked.rank_in_country <= KEEP_PER_PARENT]

# robustness: equal weights
eq, _ = score(table, pd.Series(1.0, index=list(CRITERIA)))
eq = eq[eq.rank_overall.notna()]
rho = ranked.set_index("region_code").rank_overall.corr(eq.set_index("region_code").rank_overall, method="spearman")
eq_win = eq[eq.rank_in_country <= KEEP_PER_PARENT]
same_winners = len(set(winners.region_code) & set(eq_win.region_code))

OUT.mkdir(parents=True, exist_ok=True)
cols = ["rank_overall", "rank_in_country", "country", "region", "region_code", "score", "data_share"] + list(CRITERIA)
ranked[cols].to_csv(OUT / "tier2_region_ranking.csv", index=False)
winners.sort_values(["country", "rank_in_country"])[cols].to_csv(OUT / "tier2_regions_to_tier3.csv", index=False)

# ---------- 5. report to stdout (picked up by the PDF script) ----------
print("countries carried from Tier 1:", len(best), "| with OECD regions:", table.iso3.nunique())
print("no regional data:", ", ".join(no_regions.country))
print("regions with any data:", len(table), "| ranked:", len(ranked), "| passed to Tier 3:", len(winners))
print("years:", years)
print("weights %:", (weights * 100).round(1).to_dict())
print(f"equal weights: Spearman {rho:.2f}, same Tier-3 regions {same_winners}/{len(winners)}")
print(ranked.head(20)[["rank_overall", "country", "region", "score", "data_share"]].to_string(index=False))
