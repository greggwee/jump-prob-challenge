"""One-shot verbose scheduler diagnostic — reproduces what scheduler_tick sees.

For every match: open_market_count, closing_time, minutes-to-close (UTC). Flags
the firing window [4,16]. For matches near kickoff it cross-checks (a) the REAL
open-market count via /markets, and (b) SportsPredict closing_time vs the actual
kickoff (Odds API commence_time) — to find whether the window is mis-aimed or
markets lock before the window.
"""

from datetime import datetime, timezone

from jumpprob.ingest import odds_api
from jumpprob.normalize.teams import canonical
from jumpprob.submit.api_client import SportsPredictClient

EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
WIN_MIN, WIN_MAX = 4.0, 16.0

now = datetime.now(timezone.utc)
print(f"now (UTC): {now.strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
c = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
matches = c.matches(EVENT, LOBBY)

rows = []
for m in matches:
    ko = datetime.fromisoformat(m["opening_time"].replace("Z", "+00:00"))  # kickoff = lock
    mins = (ko - now).total_seconds() / 60.0
    rows.append((mins, m))
rows.sort(key=lambda r: r[0])

# the upcoming/just-closed matches are what matter
print(f"{'match':16} {'open_cnt':>8} {'min_to_KO':>12} {'in_window':>9}  opening_time")
near_future = [r for r in rows if -30 <= r[0] <= 180]
for mins, m in near_future:
    inw = "FIRE" if (WIN_MIN <= mins <= WIN_MAX and m.get("open_market_count", 0) > 0) else ""
    print(f"{m['name']:16} {m.get('open_market_count',0):>8} {mins:>12.1f} {inw:>9}  {m['opening_time']}")

# cross-checks on the nearest still-open match
upcoming = [(mins, m) for mins, m in rows if mins > 0 and m.get("open_market_count", 0) > 0]
if upcoming:
    mins, m = upcoming[0]
    print(f"\n=== cross-check nearest open match: {m['name']} (closes in {mins:.1f} min) ===")
    mk = c.markets(LOBBY, m["id"])
    statuses = {}
    for k in mk:
        statuses[k.get("status", "?")] = statuses.get(k.get("status", "?"), 0) + 1
    print(f"  /matches open_market_count = {m.get('open_market_count')}")
    print(f"  /markets actual statuses   = {statuses}")
    # SportsPredict closing_time vs Odds API true kickoff
    try:
        evs = odds_api.load_latest()
        parts = m["name"].split(" vs ")
        cand = None
        for e in evs:
            if canonical(e["home_team"]) in [canonical(p) for p in parts] or \
               canonical(e["away_team"]) in [canonical(p) for p in parts]:
                cand = e
                break
        if cand:
            print(f"  SportsPredict closing_time = {m['closing_time']}")
            print(f"  Odds API commence_time     = {cand['commence_time']}  (true kickoff)")
    except Exception as e:
        print(f"  (kickoff cross-check skipped: {e})")
