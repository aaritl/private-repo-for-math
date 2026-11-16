import sys, json, requests
from pathlib import Path
out = Path(sys.argv[1])
IND = ["HD.HCI.HLOS","SH.H2O.BASW.ZS","SH.STA.BASS.ZS","IT.NET.USER.ZS","SH.STA.TRAF.P5","SI.POV.GINI"]
for i in [x for x in IND if not (Path(sys.argv[1])/f"{x}.json").exists()]:
    r = requests.get(f"https://api.worldbank.org/v2/country/all/indicator/{i}",
                     params={"format":"json","per_page":20000,"date":"2010:2025"}, timeout=120)
    r.raise_for_status(); data = r.json()
    (out / f"{i}.json").write_text(json.dumps(data[1]))
    print(i, data[0]["total"], data[1][0]["indicator"]["value"])
r = requests.get("https://api.worldbank.org/v2/country", params={"format":"json","per_page":400}, timeout=60)
(out / "countries.json").write_text(json.dumps(r.json()[1])); print("countries", len(r.json()[1]))
