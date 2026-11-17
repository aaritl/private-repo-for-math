"""City diversity = share of residents born abroad (%).
City value from Eurostat Urban Audit (urb_cpopcb, DE2009I, latest year >= 2011), matched to OECD cities by name.
usage: python div_prep.py <urb_cpopcb.csv> <CITIES codelist xml> <fua_located.csv> <out csv>"""
import re, sys, unicodedata
import pandas as pd
URB, CL, LOC, OUT = sys.argv[1:5]
x = open(CL, encoding="utf-8").read()
name = {m.group(1): m.group(2) for m in re.finditer(r'<s:Code id="([^"]+)".*?<c:Name xml:lang="en">([^<]*)', x, re.S)}
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)|greater city|city of|city|kommune|municipality", " ", s)
    return re.sub(r"[^a-z]+", " ", s).strip().split("/")[0]
d = pd.read_csv(URB, low_memory=False)
d = d[(d.indic_ur == "DE2009I") & d.OBS_VALUE.notna() & (d.TIME_PERIOD >= 2011) & d.cities.str.endswith("C")]
# Germany's 2021 Urban Audit values are broken (e.g. Gera 205.9%, Jena 14.8 -> 34.3 -> 15.1), so they are skipped
d = d[~(d.cities.str.startswith("DE") & (d.TIME_PERIOD == 2021)) & (d.OBS_VALUE <= 100)]
d = d.sort_values("TIME_PERIOD").groupby("cities").last().reset_index()
d["cc"] = d.cities.str[:2]; d["key"] = d.cities.map(name).map(norm)
loc = pd.read_csv(LOC, keep_default_na=False, na_values=[""])
loc["cc"] = loc.fua_code.str[:2]; loc["key"] = loc.fua_name.map(norm)
m = loc.merge(d[["cc", "key", "cities", "TIME_PERIOD", "OBS_VALUE"]], on=["cc", "key"], how="inner").drop_duplicates("fua_code")
m = m.rename(columns={"OBS_VALUE": "foreign_born", "TIME_PERIOD": "year", "cities": "eurostat_code"})
m[["fua_code", "fua_name", "eurostat_code", "year", "foreign_born"]].to_csv(OUT, index=False)
print(len(m), "cities matched")
