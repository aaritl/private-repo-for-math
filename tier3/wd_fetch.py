"""Entertainment: notable venues within 5 km of each city centre, counted from Wikidata.
Venue types (and their direct subclasses): museum, theatre, cinema, concert hall, opera house, art gallery,
music venue, stadium, amusement park, zoo.  Resumable (appends to the output CSV).
usage: python wd_fetch.py <fua_located.csv> <tier2_regions_to_tier3.csv> <out csv>
"""
import sys, time, csv
from pathlib import Path
import pandas as pd
import requests

LOC, T2, OUT = map(Path, sys.argv[1:4])
TYPES = "wd:Q33506 wd:Q24354 wd:Q41253 wd:Q1060829 wd:Q153562 wd:Q1007870 wd:Q8719053 wd:Q483110 wd:Q194195 wd:Q43501"
Q = ('SELECT (COUNT(DISTINCT ?item) AS ?n) WHERE { SERVICE wikibase:around { ?item wdt:P625 ?loc . '
     'bd:serviceParam wikibase:center "Point(LON LAT)"^^geo:wktLiteral . bd:serviceParam wikibase:radius "5" . } '
     '?item wdt:P31/wdt:P279? ?type . VALUES ?type { ' + TYPES + ' } }')
HEAD = {"User-Agent": "himcm-city-model/1.0 (student math modeling project)", "Accept": "application/sparql-results+json"}

codes = set(pd.read_csv(T2).region_code)
loc = pd.read_csv(LOC, keep_default_na=False, na_values=[""])
def inside(s): return any(c in codes or c + "0" in codes for c in str(s).split("|"))
todo = loc[loc.lat.notna() & loc.nuts_or_tl2.map(inside)]
done = set(pd.read_csv(OUT).dropna(subset=["venues_5km"]).fua_code) if OUT.exists() else set()
new = not OUT.exists()
f = open(OUT, "a", newline=""); w = csv.writer(f)
if new: w.writerow(["fua_code", "fua_name", "venues_5km"])

for i, r in enumerate(todo.itertuples()):
    if r.fua_code in done: continue
    n = ""
    for attempt in range(5):
        try:
            resp = requests.get("https://query.wikidata.org/sparql", params={"query": Q.replace("LON", str(r.lon)).replace("LAT", str(r.lat))},
                                headers=HEAD, timeout=65)
            if resp.status_code == 200:
                n = int(resp.json()["results"]["bindings"][0]["n"]["value"]); break
            print("  status", resp.status_code, resp.headers.get("Retry-After"), flush=True)
            time.sleep(min(60, int(resp.headers.get("Retry-After", 10 * (attempt + 1)))))
        except Exception as e:
            print("  error", type(e).__name__, flush=True); time.sleep(10 * (attempt + 1))
    w.writerow([r.fua_code, r.fua_name, n]); f.flush()
    print(i, len(todo), r.fua_name, n, flush=True)
    time.sleep(1)
print("done")
