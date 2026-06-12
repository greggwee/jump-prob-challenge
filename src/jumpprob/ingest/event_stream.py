"""Minute-stamped event stream from cached StatsBomb event files.

One row per event: (match_id, tournament, team, period, slot, family) with
slot in 0..89 (period 1 -> min(minute,44), period 2 -> clip(minute,45,89);
stoppage time folds into slots 44/89). Families:
  goal    score-changing: shots with outcome Goal + Own Goal For (beneficiary)
  corner  pass type Corner
  card    yellow / second yellow / red (fouls + bad behaviour)
  foul    Foul Committed
  sot     shot outcome in {Goal, Saved, Saved To Post}
  offside Offside events + passes with outcome Pass Offside
Extra time and shootouts excluded (regulation-90 simulator scope).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from jumpprob.ingest import statsbomb as sb_mod
from jumpprob.ingest.aggregate_events import SOT_OUTCOMES

PROCESSED = Path("data/processed")


def _slot(period: int, minute: int) -> int:
    if period == 1:
        return min(minute, 44)
    return min(max(minute, 45), 89)


def parse_stream(events: list[dict], match_id: int) -> list[dict]:
    rows = []

    def add(team, period, minute, family):
        if period in (1, 2) and team:
            rows.append({"match_id": match_id, "team": team, "period": period,
                         "slot": _slot(period, minute), "family": family})

    for e in events:
        period = e.get("period", 0)
        team = (e.get("team") or {}).get("name")
        etype = (e.get("type") or {}).get("name")
        minute = e.get("minute", 0)
        if etype == "Shot":
            outcome = (e.get("shot", {}).get("outcome") or {}).get("name")
            if outcome == "Goal":
                add(team, period, minute, "goal")
            if outcome in SOT_OUTCOMES:
                add(team, period, minute, "sot")
        elif etype == "Own Goal For":
            add(team, period, minute, "goal")
        elif etype == "Pass":
            p = e.get("pass", {})
            if (p.get("type") or {}).get("name") == "Corner":
                add(team, period, minute, "corner")
            if (p.get("outcome") or {}).get("name") == "Pass Offside":
                add(team, period, minute, "offside")
        elif etype == "Offside":
            add(team, period, minute, "offside")
        elif etype == "Foul Committed":
            add(team, period, minute, "foul")
            card = (e.get("foul_committed", {}).get("card") or {}).get("name")
            if card:
                add(team, period, minute, "card")
        elif etype == "Bad Behaviour":
            card = (e.get("bad_behaviour", {}).get("card") or {}).get("name")
            if card:
                add(team, period, minute, "card")
    return rows


def build_stream(out_dir: Path = PROCESSED) -> pd.DataFrame:
    matches = sb_mod.load_matches()
    rows = []
    for mid in matches["match_id"]:
        rows.extend(parse_stream(sb_mod.fetch_events(mid), mid))
    df = pd.DataFrame(rows).merge(
        matches[["match_id", "tournament", "date", "home_team", "away_team"]], on="match_id"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "event_stream.parquet", index=False)
    return df
