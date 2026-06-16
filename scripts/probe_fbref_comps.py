"""One-off: dump every competition name FBref lists on /en/comps/ so we know the
exact strings to put in soccerdata's league_dict.json for the broad leagues that
cover our coverage hole (Championship, Liga MX, MLS, Portugal, Saudi, Eredivisie)."""

import soccerdata as sd
from lxml import html

fb = sd.FBref(leagues="ENG-Premier League", seasons="2024")
reader = fb.get("https://fbref.com/en/comps/", fb.data_dir / "leagues.html")
tree = html.parse(reader)

rows = []
for t in tree.xpath("//table[contains(@id,'comps')]"):
    tid = t.get("id")
    for tr in t.xpath(".//tbody/tr"):
        name = tr.xpath(".//th[@data-stat='league_name']//a/text()")
        gender = tr.xpath(".//td[@data-stat='gender']/text()")
        country = tr.xpath(".//td[@data-stat='country']//text()")
        if name:
            rows.append((tid, (gender or [""])[0], (country or [""])[0], name[0].strip()))

with open("data/processed/fbref_comps.txt", "w", encoding="utf-8") as f:
    for tid, g, c, n in sorted(set(rows), key=lambda r: (r[0], r[3])):
        f.write(f"{tid}\t{g}\t{c}\t{n}\n")
print(f"wrote {len(set(rows))} competition rows to data/processed/fbref_comps.txt")
