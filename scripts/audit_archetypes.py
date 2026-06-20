"""Rigorously audit the archetype parser against the full live + submitted corpus.

A misparsed market routes the wrong calibration / expert blend / risk shrink to a
question, so this checks every question three ways:
  1. COVERAGE   — does parse_question return something (else it's silently dropped)?
  2. KEYWORD    — does the assigned archetype's defining keyword actually appear?
                  (e.g. an `offsides_ge` question must contain "offside")
  3. SUBJECT    — for PLAYER archetypes, is the subject a player and NOT "both teams"
                  or one of the match's two teams (the class of bug that mislabelled
                  both-teams-SoT as a single player's prop)?
Also diffs the current parse against the archetype we SUBMITTED with (matchday_
predictions.csv) so any parser drift since 2026-06-12 is visible.

  .venv/Scripts/python.exe scripts/audit_archetypes.py
"""

from __future__ import annotations

import re
import pandas as pd

from jumpprob.submit.api_client import SportsPredictClient
from jumpprob.submit.parser import parse_question
from jumpprob.normalize.teams import canonical

LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"

# the defining token(s) each archetype's question MUST contain (regex, lowercased text)
KEYWORD = {
    "offsides_ge": r"offside",
    "fouls_gt": r"foul",
    "cards_ge": r"card", "cards_gt": r"card", "team_cards_ge": r"card",
    "corners_gt": r"corner", "corners_total_ge": r"corner", "team_corners_ge": r"corner",
    "sot_gt": r"shots? on target", "team_sot_ge": r"shots? on target",
    "total_sot_ge": r"shots? on target", "total_sot_half_ge": r"shots? on target",
    "both_teams_sot_ge": r"both teams.*shots? on target",
    "player_sot_ge": r"shots? on target", "player_ga": r"score or assist",
    "player_scores": r"score a goal", "win": r"win the match",
    "total_ge": r"goals?", "total_le": r"fewer total goals",
    "btts": r"both teams score", "btts_and_total_ge": r"both teams score and",
    "ht_tied": r"tied", "ht_leading": r"winning",
    "team_goals_ge": r"score (at least|\d+ or more).*goals?",
    "team_scores_half": r"score in the (first|second) half",
    "h2_more_goals_than_h1": r"second half have more",
    "first_goal_half": r"first goal of the (first|second) half",
    "first_goal_and_win": r"first goal.*win", "first_goal_and_opp_2h": r"first goal.*second half",
    "goals_gt_half": r"more goals than.*half", "half_total_ge": r"(first|second) half have \d+ or more",
    "penalty_awarded": r"penalty", "pen_or_red": r"penalty.*red card",
}
PLAYER_ARCHS = {"player_sot_ge", "player_ga", "player_scores"}


def _corpus() -> pd.DataFrame:
    c = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
    rows = []
    for r in c.results(LOBBY):                      # settled
        rows.append({"question": r.get("question"), "match": None, "src": "settled"})
    for m in c.markets(LOBBY):                      # live (carries the match teams)
        rows.append({"question": m.get("question"), "match": m.get("match", {}).get("name"), "src": "live"})
    md = pd.read_csv("data/processed/matchday_predictions.csv")
    sub = dict(zip(md["question"], md["archetype"]))
    df = pd.DataFrame(rows).dropna(subset=["question"]).drop_duplicates("question").reset_index(drop=True)
    df["submitted_arch"] = df["question"].map(sub)
    return df


def _team_set() -> set:
    c = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
    teams = set()
    for m in c.matches("aa5572ec-5930-4d99-b06b-f8966333d172", LOBBY):
        nm = m.get("name", "")
        if " vs " in nm:
            teams |= {canonical(t.strip()) for t in nm.split(" vs ")}
    return teams


def main() -> None:
    df = _corpus()
    teams = _team_set()                       # team-aware parse (player vs team SoT)
    df["parsed"] = df["question"].map(lambda q: parse_question(q, teams=teams))
    df["arch"] = df["parsed"].map(lambda p: p.archetype if p else None)
    df["subject"] = df["parsed"].map(lambda p: p.subject if p else None)
    print(f"distinct questions audited: {len(df)}")

    # 1. coverage
    un = df[df["arch"].isna()]
    print(f"\n[1] UNPARSED (dropped): {len(un)}")
    for q in un["question"].head(15):
        print(f"    - {q[:80]}")

    # 2. keyword consistency
    bad_kw = []
    for r in df.dropna(subset=["arch"]).itertuples():
        pat = KEYWORD.get(r.arch)
        if pat and not re.search(pat, r.question.lower()):
            bad_kw.append((r.arch, r.question))
    print(f"\n[2] KEYWORD MISMATCH (archetype's defining word missing): {len(bad_kw)}")
    for a, q in bad_kw[:20]:
        print(f"    - [{a}] {q[:74]}")

    # 3. player-subject sanity
    bad_subj = []
    for r in df[df["arch"].isin(PLAYER_ARCHS)].itertuples():
        s = (r.subject or "").lower()
        match = r.match if isinstance(r.match, str) else ""
        teams = {t.strip().lower() for t in match.split(" vs ")} if " vs " in match else set()
        team_canon = {canonical(t) for t in match.split(" vs ")} if " vs " in match else set()
        if "both teams" in s or s in teams or (team_canon and canonical(r.subject or "") in team_canon):
            bad_subj.append((r.arch, r.subject, r.question))
    print(f"\n[3] PLAYER ARCHETYPE w/ NON-PLAYER subject (the both-teams class of bug): {len(bad_subj)}")
    for a, s, q in bad_subj[:20]:
        print(f"    - [{a}] subject={s!r}  {q[:60]}")

    # diff vs submitted
    d = df.dropna(subset=["arch", "submitted_arch"])
    drift = d[d["arch"] != d["submitted_arch"]]
    print(f"\n[4] PARSE DRIFT vs submitted label (2026-06-12): {len(drift)}")
    for r in drift.head(20).itertuples():
        print(f"    - submitted {r.submitted_arch:18s} now {r.arch:18s} | {r.question[:50]}")

    print("\n=== archetype distribution (current parse) ===")
    print(df["arch"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
