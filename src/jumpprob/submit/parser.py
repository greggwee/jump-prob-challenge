"""Parse market question text into (archetype, subject, threshold, half).

Patterns cover the ten archetypes from the competition brief. Anything that
does not match returns None and is routed to manual review — we never guess a
question's meaning silently (a misparsed market is worse than a skipped one).
Team/player names are resolved downstream via normalize.teams.canonical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedQuestion:
    archetype: str
    subject: str | None = None      # team or player the question is about
    other: str | None = None        # opponent in comparison questions
    threshold: int | None = None
    half: int | None = None


_PATTERNS: list[tuple[re.Pattern, callable]] = [
    # --- second wave of live phrasings (2026-06-12 full market scan) ---
    (re.compile(r"will (.+?) score the first goal of the game and (.+?) score in the second half", re.I),
     lambda m: ParsedQuestion("first_goal_and_opp_2h", m[1], m[2])),
    (re.compile(r"will (.+?) score the first goal of the game and (?:go on to )?win", re.I),
     lambda m: ParsedQuestion("first_goal_and_win", m[1])),
    (re.compile(r"will (.+?) score the first goal of the (first|second) half", re.I),
     lambda m: ParsedQuestion("first_goal_half", m[1], half=1 if m[2].lower() == "first" else 2)),
    (re.compile(r"will (.+?) finish with more corner kicks than (.+?)\?", re.I),
     lambda m: ParsedQuestion("corners_gt", m[1], m[2], half=None)),
    (re.compile(r"in the (first|second) half, will (.+?) have more corner kicks than (.+?)\?", re.I),
     lambda m: ParsedQuestion("corners_gt", m[2], m[3], half=1 if m[1].lower() == "first" else 2)),
    (re.compile(r"in the (first|second) half, will (.+?) have more shots on target than (.+?)\?", re.I),
     lambda m: ParsedQuestion("sot_gt", m[2], m[3], half=1 if m[1].lower() == "first" else 2)),
    (re.compile(r"will (.+?) receive more cards than (.+?)\?", re.I),
     lambda m: ParsedQuestion("cards_gt", m[1], m[2])),
    (re.compile(r"will (.+?) score more goals than (.+?) in the (first|second) half", re.I),
     lambda m: ParsedQuestion("goals_gt_half", m[1], m[2], half=1 if m[3].lower() == "first" else 2)),
    (re.compile(r"will (.+?) score at least (\d+) goals?", re.I),
     lambda m: ParsedQuestion("team_goals_ge", m[1], threshold=int(m[2]))),
    (re.compile(r"will (.+?) score (\d+) or more total goals", re.I),
     lambda m: ParsedQuestion("team_goals_ge", m[1], threshold=int(m[2]))),
    (re.compile(r"will (.+?) have (\d+) or more shots on target in the (first|second) half", re.I),
     lambda m: ParsedQuestion("team_sot_ge", m[1], threshold=int(m[2]), half=1 if m[3].lower() == "first" else 2)),
    (re.compile(r"will (.+?) have (\d+) or more shots on target", re.I),
     lambda m: ParsedQuestion("team_sot_ge", m[1], threshold=int(m[2]))),
    (re.compile(r"will (.+?) receive at least (\d+) cards? in the (first|second) half", re.I),
     lambda m: ParsedQuestion("team_cards_ge", m[1], threshold=int(m[2]), half=1 if m[3].lower() == "first" else 2)),
    (re.compile(r"will (.+?) receive at least (\d+) cards?", re.I),
     lambda m: ParsedQuestion("team_cards_ge", m[1], threshold=int(m[2]))),
    (re.compile(r"will (.+?) have at least (\d+) corner kicks? in the (first|second) half", re.I),
     lambda m: ParsedQuestion("team_corners_ge", m[1], threshold=int(m[2]), half=1 if m[3].lower() == "first" else 2)),
    (re.compile(r"will there be (\d+) or more total cards shown\?", re.I),
     lambda m: ParsedQuestion("cards_ge", threshold=int(m[1]), half=None)),
    (re.compile(r"will there be (\d+) or more total corner kicks in the (first|second) half", re.I),
     lambda m: ParsedQuestion("corners_total_ge", threshold=int(m[1]), half=1 if m[2].lower() == "first" else 2)),
    (re.compile(r"will there be (\d+) or more total corner kicks", re.I),
     lambda m: ParsedQuestion("corners_total_ge", threshold=int(m[1]))),
    (re.compile(r"will there be (\d+) or more total shots on target\?", re.I),
     lambda m: ParsedQuestion("total_sot_ge", threshold=int(m[1]))),
    (re.compile(r"will the match have (\d+) or fewer total goals", re.I),
     lambda m: ParsedQuestion("total_le", threshold=int(m[1]))),
    (re.compile(r"will the match be tied at halftime", re.I),
     lambda m: ParsedQuestion("ht_tied")),
    (re.compile(r"at halftime, will (.+?) be winning", re.I),
     lambda m: ParsedQuestion("ht_leading", m[1])),
    (re.compile(r"will the second half have more (?:total )?goals than the first", re.I),
     lambda m: ParsedQuestion("h2_more_goals_than_h1")),
    (re.compile(r"will (.+?) score or assist a goal", re.I),
     lambda m: ParsedQuestion("player_ga", m[1])),
    # --- archetypes observed in live WC2026 markets (2026-06-12) ---
    (re.compile(r"will (.+?) commit more fouls than (.+?)\?", re.I),
     lambda m: ParsedQuestion("fouls_gt", m[1], m[2])),
    (re.compile(r"will there be (\d+) or more total shots on target in the (first|second) half", re.I),
     lambda m: ParsedQuestion("total_sot_half_ge", threshold=int(m[1]), half=1 if m[2].lower() == "first" else 2)),
    (re.compile(r"will a penalty kick be awarded or a red card be shown", re.I),
     lambda m: ParsedQuestion("pen_or_red")),
    (re.compile(r"at halftime, will the match be tied", re.I),
     lambda m: ParsedQuestion("ht_tied")),
    (re.compile(r"will the (first|second) half have (\d+) or more total goals", re.I),
     lambda m: ParsedQuestion("half_total_ge", threshold=int(m[2]), half=1 if m[1].lower() == "first" else 2)),
    (re.compile(r"will (.+?) have (\d+) or more corner kicks", re.I),
     lambda m: ParsedQuestion("team_corners_ge", m[1], threshold=int(m[2]))),
    (re.compile(r"at halftime.*will (.+?) have more corner kicks than (.+?)\?", re.I),
     lambda m: ParsedQuestion("corners_gt", m[1], m[2], half=1)),
    (re.compile(r"will (.+?) be caught offside (\d+) or more times", re.I),
     lambda m: ParsedQuestion("offsides_ge", m[1], threshold=int(m[2]))),
    (re.compile(r"will there be (\d+) or more total cards .*second half", re.I),
     lambda m: ParsedQuestion("cards_ge", threshold=int(m[1]), half=2)),
    (re.compile(r"will there be (\d+) or more total cards .*first half", re.I),
     lambda m: ParsedQuestion("cards_ge", threshold=int(m[1]), half=1)),
    (re.compile(r"will both teams score and the match have (\d+) or more total goals", re.I),
     lambda m: ParsedQuestion("btts_and_total_ge", threshold=int(m[1]))),
    (re.compile(r"will a penalty kick be awarded", re.I),
     lambda m: ParsedQuestion("penalty_awarded")),
    (re.compile(r"will (.+?) have more shots on target than (.+?) in the second half", re.I),
     lambda m: ParsedQuestion("sot_gt", m[1], m[2], half=2)),
    (re.compile(r"will (.+?) have more shots on target than (.+?) in the first half", re.I),
     lambda m: ParsedQuestion("sot_gt", m[1], m[2], half=1)),
    (re.compile(r"will (.+?) win the match", re.I),
     lambda m: ParsedQuestion("win", m[1])),
    (re.compile(r"will (.+?) score in the second half", re.I),
     lambda m: ParsedQuestion("team_scores_half", m[1], half=2)),
    (re.compile(r"will (.+?) score in the first half", re.I),
     lambda m: ParsedQuestion("team_scores_half", m[1], half=1)),
    (re.compile(r"will (.+?) score a goal \(excluding own goals\)", re.I),
     lambda m: ParsedQuestion("player_scores", m[1])),
    (re.compile(r"will (.+?) have at least (\d+) shots? on target in the (first|second) half", re.I),
     lambda m: ParsedQuestion("player_sot_ge", m[1], threshold=int(m[2]), half=1 if m[3].lower() == "first" else 2)),
    (re.compile(r"will (.+?) have at least (\d+) shots? on target", re.I),
     lambda m: ParsedQuestion("player_sot_ge", m[1], threshold=int(m[2]))),
    (re.compile(r"will both teams score", re.I),
     lambda m: ParsedQuestion("btts")),
    (re.compile(r"will the match have (\d+) or more (?:total )?goals", re.I),
     lambda m: ParsedQuestion("total_ge", threshold=int(m[1]))),
]


def parse_question(text: str) -> ParsedQuestion | None:
    text = text.strip()
    for pattern, build in _PATTERNS:
        m = pattern.search(text)
        if m:
            return build(m)
    return None
