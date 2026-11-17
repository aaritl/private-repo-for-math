"""Tier 3 = cities INSIDE the Tier 2 winning regions (funnel), using OECD city data (Functional Urban Areas).

Same model as Tier 1: cap at 5th/95th percentile -> rescale 0-1 (flip "lower") -> entropy weights
(computed from real values only) -> weighted average over the criteria a city has -> rank cities with
enough data -> top 10 cities.
No criterion repeats Tier 1 or Tier 2: air quality, green space and public transport access are scored in Tier 2.
usage: python tier3_cities.py <oecd_city dir> <fua_located.csv> <tier2_regions_to_tier3.csv> <results dir>
       <wd_venues.csv> <city_foreign_born.csv> <OECD DSD_REG_MIGRANT@DF_MIGR_STOCK.csv>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

CITY, LOCATED, TIER2, OUT, ENT, CITY_FB, REG_FB = map(Path, sys.argv[1:8])
MIN_DATA = 0.75
OUTLIER_CAP = 0.05
FIRST_YEAR = 2015
TOP_N = 10
DROP_REGIONS = {"FRY5"}   # Mayotte: overseas territory that only reached France's top 5 through missing data

# criterion: (direction, label, description)
CRITERIA = {
    # air quality, green space and public transport are NOT Tier 3 criteria: they are scored in Tier 2 (no double counting)
    "entertainment": ("higher", "Entertainment",  "log(1 + number of notable museums, theatres, cinemas, concert halls, galleries, music venues, stadiums, zoos and theme parks within 5 km of the centre)"),
    "diversity":    ("higher", "Diversity",       "Residents born in another country (%); region's share where the city has none"),
}

def read(name):
    return pd.read_csv(CITY / f"{name}.csv", low_memory=False)
def latest(df):
    df = df[(df.TIME_PERIOD >= FIRST_YEAR) & df.OBS_VALUE.notna()]
    return df.sort_values("TIME_PERIOD").groupby("REF_AREA").last()

sv = read("DSD_FUA_TRAN@DF_SERVICE_ACCESS"); sv = sv[sv.TRAVEL_TIME == "MN_LE15"]
lab = read("DSD_FUA_LAB@DF_LABOUR")
lab = lab[(lab.MEASURE == "UNE_RATE") & (lab.TIME_PERIOD >= 2018)]
lab = lab.sort_values("AGE").drop_duplicates(["REF_AREA", "TIME_PERIOD"], keep="first")   # prefer ages 15-64 ("Y15T64" < "Y_GE15")
reg = read("DSD_REG_LAB@DF_RATES")
reg = reg[(reg.MEASURE == "UNE_RATE") & (reg.SEX == "_T") & (reg.AGE == "Y15T64") & (reg.TERRITORIAL_LEVEL == "TL2")]
REGION_UNEMP = latest(reg)

sv_latest = sv[sv.OBS_VALUE.notna()].sort_values("TIME_PERIOD").groupby(["REF_AREA", "SERVICE"]).OBS_VALUE.last().unstack()
parts = {"services": sv_latest.mean(axis=1), "unemployment": latest(lab).OBS_VALUE}
years = {"services": sv.groupby("REF_AREA").TIME_PERIOD.max(), "unemployment": latest(lab).TIME_PERIOD}
data = pd.DataFrame(parts)

# ---------- 1. keep only cities inside the Tier 2 winning regions ----------
t2 = pd.read_csv(TIER2)
t2 = t2[~t2.region_code.isin(DROP_REGIONS)]
codes = set(t2.region_code)
loc = pd.read_csv(LOCATED, keep_default_na=False, na_values=[""])
def pick(s):
    for c in str(s).split("|"):
        for k in (c, c + "0"):            # NUTS level-1 'MT0' = OECD 'MT00' (single-region countries)
            if k in codes: return k
    return None
loc["region_code"] = loc.nuts_or_tl2.map(pick)
cities = loc[loc.region_code.notna()].merge(t2[["region_code", "region", "country", "rank_in_country", "score"]]
                                            .rename(columns={"score": "region_score", "rank_in_country": "region_rank"}), on="region_code")
cities = cities.set_index("fua_code").join(data, how="left")
cities = cities.rename(columns={"fua_name": "city"})

# Jobs: the city's own unemployment rate (2018 or later); if the OECD has none for that city,
# the unemployment rate of its Tier 2 region (so missing data cannot push a city up the ranking)
own = cities["unemployment"].notna()
cities["unemployment_source"] = np.where(own, "city", "region")
cities.loc[~own, "unemployment"] = cities.loc[~own, "region_code"].map(REGION_UNEMP.OBS_VALUE)
years["unemployment"] = pd.concat([years["unemployment"], REGION_UNEMP.TIME_PERIOD])

# Entertainment: notable venues within 5 km of the city centre (Wikidata)
ent = pd.read_csv(ENT).set_index("fua_code").venues_5km
cities["venues"] = pd.to_numeric(ent.reindex(cities.index), errors="coerce")
# log scale: venue counts are very lopsided (Paris 747, most cities 10-50). On a log scale the step from
# 10 to 50 venues counts more than 700 to 740 (diminishing returns), and one huge city cannot set the scale.
cities["entertainment"] = np.log1p(cities["venues"])
# Diversity: share of residents born abroad. City value from Eurostat Urban Audit (European cities);
# otherwise the share in its Tier 2 region (OECD, ages 15-64); New Zealand has neither, so the national
# share is used (UN international migrant stock via World Bank SM.POP.TOTL.ZS, 2024: 28.2%).
cfb = pd.read_csv(CITY_FB).set_index("fua_code")
cities["diversity"] = cfb.foreign_born.reindex(cities.index)
rfb = pd.read_csv(REG_FB, low_memory=False)
rfb = rfb[(rfb.MEASURE == "POP") & (rfb.ORIGIN == "FB") & (rfb.UNIT_MEASURE == "PT_POP_SUB") & (rfb.AGE == "Y15T64")
          & (rfb.SEX == "_T") & (rfb.EDUCATION_LEV == "_T")]
REGION_FB = latest(rfb)
COUNTRY_FB = {"NZ": (28.2, 2024)}
own = cities["diversity"].notna()
cities["diversity_source"] = np.where(own, "city (Eurostat)", "region (OECD)")
cities.loc[~own, "diversity"] = cities.loc[~own, "region_code"].map(REGION_FB.OBS_VALUE)
miss = cities.diversity.isna() & cities.iso2.isin(COUNTRY_FB)
cities.loc[miss, "diversity"] = cities.loc[miss, "iso2"].map(lambda k: COUNTRY_FB[k][0])
cities.loc[miss, "diversity_source"] = "country (UN)"
years["diversity"] = pd.concat([cfb.year.reindex(cities.index[own]), REGION_FB.TIME_PERIOD.reindex(cities.region_code[~own]),
                                pd.Series([COUNTRY_FB[k][1] for k in cities.iso2[miss]], dtype=float)]).dropna()

YEARS = {k: f"{int(v.min())}" if v.min() == v.max() else f"{int(v.min())}-{int(v.max())}" for k, v in years.items()}
YEARS["entertainment"] = "2026 (Wikidata snapshot)"

# ---------- 2. the model (same as Tier 1) ----------
def entropy_weights(scaled):
    d = {}
    for c in scaled:
        x = scaled[c].dropna()
        p = (x + 1e-12) / (x + 1e-12).sum()
        e = -(p * np.log(p)).sum() / np.log(len(x))
        d[c] = max(0.0, 1 - e)
    d = pd.Series(d); return d / d.sum()

def scale(t, cols, cap=OUTLIER_CAP):
    v = t[cols]
    if cap: v = v.clip(v.quantile(cap), v.quantile(1 - cap), axis=1)
    s = (v - v.min()) / (v.max() - v.min())
    for c in cols:
        if CRITERIA[c][0] == "lower": s[c] = 1 - s[c]
    return s

def score(t, cols=None, weights=None, min_data=MIN_DATA, cap=OUTLIER_CAP):
    cols = cols or list(CRITERIA)
    s = scale(t, cols, cap)
    w = entropy_weights(s) if weights is None else weights.reindex(cols) / weights.reindex(cols).sum()
    r = t.copy()
    r["data_share"] = (s.notna() * w).sum(axis=1)
    r["score"] = (s * w).sum(axis=1) / r["data_share"]
    ok = r.data_share >= min_data - 1e-9
    r["rank"] = r.loc[ok, "score"].rank(ascending=False, method="min")
    return r.sort_values("score", ascending=False), w, s

ranked, weights, scaled = score(cities)
ranked = ranked[ranked["rank"].notna()]

# ---------- 3. robustness ----------
def top(r, n=TOP_N): return set(r[r["rank"].notna()].nsmallest(n, "rank").index)
def rho(a, b):
    a = a[a["rank"].notna()]["rank"]; b = b[b["rank"].notna()]["rank"]; i = a.index.intersection(b.index)
    return a[i].corr(b[i], method="spearman")
checks = []
eq, _, _ = score(cities, weights=pd.Series(1.0, index=list(CRITERIA)))
checks.append(("Equal weights (each criterion 1/2)", len(top(ranked) & top(eq)), rho(ranked, eq)))
loose, _, _ = score(cities, min_data=0.60)
checks.append(("Looser data rule (60% instead of 75%)", len(top(ranked) & top(loose)), rho(ranked, loose)))
nocap, _, _ = score(cities, cap=0)
checks.append(("No outlier cap (raw min-max)", len(top(ranked) & top(nocap)), rho(ranked, nocap)))
loo = []
for c in CRITERIA:
    r2, _, _ = score(cities, cols=[k for k in CRITERIA if k != c])
    loo.append((CRITERIA[c][1], len(top(ranked) & top(r2)), rho(ranked, r2)))

variants = [ranked, eq, loose, nocap] + [score(cities, cols=[k for k in CRITERIA if k != c])[0] for c in CRITERIA]
freq = pd.Series(0, index=cities.index)
for v in variants: freq[list(top(v))] += 1
freq = freq[freq > 0].sort_values(ascending=False)
N_VARIANTS = len(variants)
OUT.mkdir(parents=True, exist_ok=True)
cols = ["rank", "city", "country", "region", "region_code", "score", "data_share"] + list(CRITERIA) + \
       ["venues", "unemployment_source", "diversity_source", "lat", "lon"]
ranked.reset_index().rename(columns={"index": "fua_code"})[["fua_code"] + cols].to_csv(OUT / "tier3_city_ranking.csv", index=False)
scaled.loc[ranked.index].assign(city=ranked.city, country=ranked.country, rank=ranked["rank"]).to_csv(OUT / "tier3_grades.csv", index_label="fua_code")
pd.DataFrame({"criterion": list(CRITERIA), "label": [CRITERIA[c][1] for c in CRITERIA],
              "direction": [CRITERIA[c][0] for c in CRITERIA], "years": [YEARS[c] for c in CRITERIA],
              "cities_with_data": [int(cities[c].notna().sum()) for c in CRITERIA],
              "weight": weights.values}).to_csv(OUT / "tier3_weights.csv", index=False)
pd.DataFrame(checks + [("Leave out " + n, o, r) for n, o, r in loo], columns=["test", "top10_overlap", "spearman"]
             ).to_csv(OUT / "tier3_stability.csv", index=False)

pd.DataFrame({"city": cities.loc[freq.index, "city"], "country": cities.loc[freq.index, "country"],
              "times_in_top10": freq.values, "out_of": N_VARIANTS, "main_rank": ranked["rank"].reindex(freq.index).values}
             ).to_csv(OUT / "tier3_top10_frequency.csv", index_label="fua_code")
if __name__ == "__main__" or True:
    print("cities in Tier 2 regions:", len(cities), "| ranked:", len(ranked), "| regions with a city:", cities.region_code.nunique(), "of", len(codes))
    print("coverage:", {c: int(cities[c].notna().sum()) for c in CRITERIA})
    print("years:", YEARS)
    print("weights %:", (weights * 100).round(1).to_dict())
    for n, o, r in checks: print(f"  {n}: top-10 {o}/10, rho {r:.3f}")
    for n, o, r in loo: print(f"  leave out {n}: top-10 {o}/10, rho {r:.3f}")
    print("top-10 frequency (of", N_VARIANTS, "runs):"); print(pd.DataFrame({"city": cities.loc[freq.index, "city"], "n": freq, "rank": ranked["rank"].reindex(freq.index)}).head(20).to_string())
    print(ranked.head(25)[["rank", "city", "country", "region", "score", "data_share"] + list(CRITERIA)].round(3).to_string())
