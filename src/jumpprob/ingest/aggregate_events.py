"""Aggregate StatsBomb event files into analysis tables (one pass, two outputs).

team_half table: per (match_id, team, period in {1,2}): corners, yellows, reds,
offside_events, pass_offsides, shots, shots_on_target, goals, penalties_awarded.

players table: per (match_id, player): team, started, minutes, shots, sot,
goals (excl. own goals), penalties_taken.

Notes on StatsBomb semantics:
- Corners: Pass events with pass.type == "Corner".
- Cards: foul_committed.card or bad_behaviour.card (Yellow/Second Yellow/Red).
- Offsides appear BOTH as dedicated "Offside" events and as passes with
  outcome "Pass Offside"; we store both counts and choose empirically
  (scripts/estimate_counts.py prints the reconciliation).
- Shots on target: outcome in {Goal, Saved, Saved To Post} per Opta convention.
- Periods: 1, 2 regulation halves; 3, 4 extra time; 5 shootout (excluded).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from jumpprob.ingest import statsbomb as sb_mod

SOT_OUTCOMES = {"Goal", "Saved", "Saved To Post"}
PROCESSED = Path("data/processed")


def parse_match(events: list[dict], match_id: int) -> tuple[list[dict], list[dict]]:
    team_half = defaultdict(lambda: defaultdict(int))
    players = defaultdict(lambda: {"shots": 0, "sot": 0, "goals": 0, "assists": 0, "penalties_taken": 0,
                                   "started": False, "on_min": None, "off_min": None, "team": None})
    teams = set()
    max_minute_regulation = 90
    has_extra_time = any(e.get("period") in (3, 4) for e in events)
    duration = 120 if has_extra_time else 90

    for e in events:
        period = e.get("period", 0)
        team = (e.get("team") or {}).get("name")
        etype = (e.get("type") or {}).get("name")
        minute = e.get("minute", 0)
        if team:
            teams.add(team)
        key = (team, period)

        if etype == "Starting XI":
            for p in e["tactics"]["lineup"]:
                pl = players[p["player"]["name"]]
                pl["started"] = True
                pl["on_min"] = 0
                pl["team"] = team
        elif etype == "Substitution":
            off = players[(e.get("player") or {}).get("name")]
            off["off_min"] = minute
            repl = e.get("substitution", {}).get("replacement", {}).get("name")
            if repl:
                players[repl]["on_min"] = minute
                players[repl]["team"] = team
        elif etype == "Pass":
            if e.get("pass", {}).get("goal_assist") and period != 5:
                pname = (e.get("player") or {}).get("name")
                if pname:
                    players[pname]["assists"] += 1
                    players[pname]["team"] = team
            if (e.get("pass", {}).get("type") or {}).get("name") == "Corner" and period in (1, 2):
                team_half[key]["corners"] += 1
            if (e.get("pass", {}).get("outcome") or {}).get("name") == "Pass Offside" and period in (1, 2):
                team_half[key]["pass_offsides"] += 1
        elif etype == "Offside" and period in (1, 2):
            team_half[key]["offside_events"] += 1
        elif etype == "Shot" and period != 5:
            shot = e.get("shot", {})
            outcome = (shot.get("outcome") or {}).get("name")
            pname = (e.get("player") or {}).get("name")
            if pname:
                pl = players[pname]
                pl["team"] = team
                pl["shots"] += 1
                if outcome in SOT_OUTCOMES:
                    pl["sot"] += 1
                if outcome == "Goal":
                    pl["goals"] += 1
                if (shot.get("type") or {}).get("name") == "Penalty":
                    pl["penalties_taken"] += 1
                    team_half[key]["penalties_awarded"] += 1
            if period in (1, 2):
                team_half[key]["shots"] += 1
                if outcome in SOT_OUTCOMES:
                    team_half[key]["shots_on_target"] += 1
                if outcome == "Goal":
                    team_half[key]["goals"] += 1
        if etype == "Foul Committed" and period in (1, 2):
            team_half[key]["fouls"] += 1
        if etype in ("Foul Committed", "Bad Behaviour") and period in (1, 2):
            card = (e.get("foul_committed", {}).get("card") or e.get("bad_behaviour", {}).get("card") or {})
            name = card.get("name")
            if name in ("Yellow Card", "Second Yellow"):
                team_half[key]["yellows"] += 1
            elif name == "Red Card":
                team_half[key]["reds"] += 1

    th_rows = []
    for (team, period), counts in team_half.items():
        if team is None or period not in (1, 2):
            continue
        row = {"match_id": match_id, "team": team, "period": period}
        for col in ("corners", "yellows", "reds", "offside_events", "pass_offsides",
                    "shots", "shots_on_target", "goals", "penalties_awarded", "fouls"):
            row[col] = counts.get(col, 0)
        th_rows.append(row)
    # ensure zero rows exist for both teams both halves (a half with no corners
    # must appear as 0, not be missing)
    for team in teams:
        for period in (1, 2):
            if not any(r["team"] == team and r["period"] == period for r in th_rows):
                th_rows.append({"match_id": match_id, "team": team, "period": period,
                                "corners": 0, "yellows": 0, "reds": 0, "offside_events": 0,
                                "pass_offsides": 0, "shots": 0, "shots_on_target": 0,
                                "goals": 0, "penalties_awarded": 0, "fouls": 0})

    pl_rows = []
    for name, p in players.items():
        if p["team"] is None:
            continue
        on = p["on_min"]
        minutes = 0.0
        if on is not None:
            minutes = (p["off_min"] if p["off_min"] is not None else duration) - on
        pl_rows.append({"match_id": match_id, "player": name, "team": p["team"],
                        "started": p["started"], "minutes": max(minutes, 0.0),
                        "shots": p["shots"], "sot": p["sot"], "goals": p["goals"],
                        "assists": p["assists"], "penalties_taken": p["penalties_taken"]})
    return th_rows, pl_rows


def build_tables(out_dir: Path = PROCESSED) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = sb_mod.load_matches()
    th_all, pl_all = [], []
    for mid in matches["match_id"]:
        events = sb_mod.fetch_events(mid)  # cached on disk by now
        th, pl = parse_match(events, mid)
        th_all.extend(th)
        pl_all.extend(pl)
    team_half = pd.DataFrame(th_all).merge(
        matches[["match_id", "tournament", "date", "home_team", "away_team", "stage", "referee"]],
        on="match_id",
    )
    players = pd.DataFrame(pl_all).merge(
        matches[["match_id", "tournament", "date", "stage"]], on="match_id"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    team_half.to_parquet(out_dir / "team_half_counts.parquet", index=False)
    players.to_parquet(out_dir / "player_match.parquet", index=False)
    return team_half, players
