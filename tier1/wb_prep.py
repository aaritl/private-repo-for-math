"""Build new Batch 1 inputs from World Bank WDI (latest value 2010+ per country), plus divorce from the old batch."""
import sys, json
from pathlib import Path
import pandas as pd
raw, geonames, old, out = map(Path, sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
IND = {"basic_water":"SH.H2O.BASW.ZS", "basic_sanitation":"SH.STA.BASS.ZS",
       "internet_users":"IT.NET.USER.ZS", "traffic_deaths":"SH.STA.TRAF.P5", "gini":"SI.POV.GINI", "test_scores":"HD.HCI.HLOS"}
countries = json.loads((raw/"countries.json").read_text())
real = {c["id"] for c in countries if c["region"]["id"] != "NA"}          # drop aggregates (World, regions, income groups)
geo = pd.read_csv(geonames, sep="\t", comment="#", header=None, usecols=[0,1], names=["iso2","iso3"], dtype=str, keep_default_na=False)
iso3_to_2 = dict(zip(geo.iso3, geo.iso2)); iso3_to_2["XKX"] = "XK"
cols, log, years = {}, [], {}
for name, code in IND.items():
    rows = pd.DataFrame([{"iso3": r["countryiso3code"], "year": int(r["date"]), "v": r["value"]}
                         for r in json.loads((raw/f"{code}.json").read_text())
                         if r["value"] is not None and r["countryiso3code"] in real])
    rows = rows.sort_values("year").groupby("iso3").last()                 # latest year per country
    rows.index = rows.index.map(iso3_to_2.get)
    unmatched = rows.index.isna().sum(); rows = rows[rows.index.notna()]
    cols[name] = rows["v"]; years[name] = f"{rows.year.min()}-{rows.year.max()} (median {int(rows.year.median())})"
    log.append({"criterion": name, "countries": len(rows), "unmatched": int(unmatched), "years": years[name]})
t1 = pd.DataFrame(cols)
oldt = pd.read_csv(old/"tier1_inputs.csv", keep_default_na=False, na_values=[""]).set_index("geo_id"); d = oldt["divorce_rate"].dropna(); t1.insert(0, "uhc_service_coverage", oldt["uhc_service_coverage"])
t1["divorce_rate"] = d
log.append({"criterion": "divorce_rate", "countries": int(t1.divorce_rate.notna().sum()), "unmatched": 0, "years": "2002-2006"})
t1 = t1.dropna(how="all")
t1.rename_axis("geo_id").reset_index().to_csv(out/"tier1_inputs.csv", index=False)
# keep Batch 2 exactly as before
(out/"tier2_inputs.csv").write_text((old/"tier2_inputs.csv").read_text())
old_log = [r for r in json.loads((old/"match_log.json").read_text()) if r["criterion"] not in {x["criterion"] for x in log}]
(out/"match_log.json").write_text(json.dumps(log + old_log, indent=1))
for r in log: print(r)
n = t1.notna().sum(axis=1); print("rows", len(t1), "| countries with all 8:", (n==8).sum(), "| >=6:", (n>=6).sum())
