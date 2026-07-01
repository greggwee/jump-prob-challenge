"""Parse market question text into (archetype, subject, threshold, half).

Patterns cover the ten archetypes from the competition brief. Anything that
does not match returns None and is routed to manual review — we never guess a
question's meaning silently (a misparsed market is worse than a skipped one).
Team/player names are resolved downstream via normalize.teams.canonical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from jumpprob.normalize.teams import canonical

# A few archetypes are phrased identically for a player or a team ("... have at
# least N shots on target"); the regex defaults to the player reading, so when the
# subject is one of the match's known teams we re-route to the team archetype.
_PLAYER_TO_TEAM = {"player_sot_ge": "team_sot_ge"}


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
    (re.compile(r"will (.+?) score (\d+) or more (?:total )?goals", re.I),
     lambda m: ParsedQuestion("team_goals_ge", m[1], threshold=int(m[2]))),
    # "N or more shots on target" mirrors the "at least N" block below: both-teams FIRST
    # (else the greedy (.+?) swallows "both teams"), then DEFAULT to player_sot_ge so a
    # player subject ("Musiala (Germany) ... 2 or more shots on target") is priced as a
    # prop; _PLAYER_TO_TEAM re-routes to team_sot_ge when the subject is one of the teams.
    (re.compile(r"will both teams have (\d+) or more shots on target in the (first|second) half", re.I),
     lambda m: ParsedQuestion("both_teams_sot_ge", threshold=int(m[1]), half=1 if m[2].lower() == "first" else 2)),
    (re.compile(r"will both teams have (\d+) or more shots on target", re.I),
     lambda m: ParsedQuestion("both_teams_sot_ge", threshold=int(m[1]))),
    (re.compile(r"will (.+?) have (\d+) or more shots on target in the (first|second) half", re.I),
     lambda m: ParsedQuestion("player_sot_ge", m[1], threshold=int(m[2]), half=1 if m[3].lower() == "first" else 2)),
    (re.compile(r"will (.+?) have (\d+) or more shots on target", re.I),
     lambda m: ParsedQuestion("player_sot_ge", m[1], threshold=int(m[2]))),
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
    (re.compile(r"will (.+?) be (?:ahead|winning|leading) at halftime", re.I),
     lambda m: ParsedQuestion("ht_leading", m[1])),
    (re.compile(r"will the second half have more (?:total )?goals than the first", re.I),
     lambda m: ParsedQuestion("h2_more_goals_than_h1")),
    (re.compile(r"will (.+?) score or assist a goal", re.I),
     lambda m: ParsedQuestion("player_ga", m[1])),
    # --- knockout / derived goal archetypes (recovered from the 2026-06-28 drop audit);
    #     all answerable straight from the (market-anchored) goal grid — see runner.answer ---
    (re.compile(r"will (.+?) score the first goal of the match", re.I),
     lambda m: ParsedQuestion("first_goal_match", m[1])),
    (re.compile(r"will (.+?) win by (\d+) or more goals", re.I),
     lambda m: ParsedQuestion("win_margin", m[1], threshold=int(m[2]))),
    (re.compile(r"will (.+?) keep a clean sheet", re.I),
     lambda m: ParsedQuestion("clean_sheet", m[1])),
    (re.compile(r"will (.+?) score in both halves", re.I),
     lambda m: ParsedQuestion("score_both_halves", m[1])),
    (re.compile(r"end in a (?:tie|draw)", re.I),
     lambda m: ParsedQuestion("draw")),
    (re.compile(r"will the (first|second) half produce (\d+) or more goals", re.I),
     lambda m: ParsedQuestion("half_total_ge", threshold=int(m[2]), half=1 if m[1].lower() == "first" else 2)),
    (re.compile(r"will the second half produce more goals than the first", re.I),
     lambda m: ParsedQuestion("h2_more_goals_than_h1")),
    # knockout advancement = win THIS tie (incl. extra time + penalties); the round
    # named doesn't change the handler. Subject is the team; runner derives it from
    # the 90-min market + an ET/shootout model (see runner.answer "advance").
    (re.compile(r"will (.+?) advance to the (?:round of \d+|quarter[- ]?finals?|semi[- ]?finals?|final)", re.I),
     lambda m: ParsedQuestion("advance", m[1])),
    (re.compile(r"will a card be shown in the (first|second) half", re.I),
     lambda m: ParsedQuestion("cards_ge", threshold=1, half=1 if m[1].lower() == "first" else 2)),
    (re.compile(r"will a red card be shown", re.I),
     lambda m: ParsedQuestion("red_card")),
    # hydration / cooling-break windows (breaks at 22' of each half: match min 22 and 67) -> analytic
    (re.compile(r"will a goal be scored before the first hydration break", re.I),
     lambda m: ParsedQuestion("goal_pre_hb1")),
    (re.compile(r"will a goal be scored after the second hydration break", re.I),
     lambda m: ParsedQuestion("goal_post_hb2")),
    (re.compile(r"will (\d+) or more corner kicks be taken before the first hydration break", re.I),
     lambda m: ParsedQuestion("corners_pre_hb1", threshold=int(m[1]))),
    (re.compile(r"will a card be shown after the second hydration break", re.I),
     lambda m: ParsedQuestion("cards_post_hb2")),
    (re.compile(r"will either team be ruled offside before the first hydration break", re.I),
     lambda m: ParsedQuestion("offsides_pre_hb1")),
    # count markets summed over both teams (modeled from the existing count families;
    # total shots uses real team_half_counts.shots, not a SoT ratio)
    (re.compile(r"will there be (\d+) or more total shots \(on and off target\)", re.I),
     lambda m: ParsedQuestion("total_shots_ge", threshold=int(m[1]))),
    (re.compile(r"will there be (\d+) or more offside calls", re.I),
     lambda m: ParsedQuestion("offsides_total_ge", threshold=int(m[1]))),
    (re.compile(r"will both teams receive at least one card", re.I),
     lambda m: ParsedQuestion("both_teams_cards", threshold=1)),
    # stoppage-time goals + substitutions (base rates from api-football WC2022, runner.event_base)
    (re.compile(r"will a goal be scored in first-half stoppage", re.I),
     lambda m: ParsedQuestion("goal_1h_stoppage")),
    (re.compile(r"will a goal be scored in second-half stoppage", re.I),
     lambda m: ParsedQuestion("goal_2h_stoppage")),
    (re.compile(r"will a substitution be made before halftime", re.I),
     lambda m: ParsedQuestion("sub_before_ht")),
    (re.compile(r"will a substitute score a goal", re.I),
     lambda m: ParsedQuestion("substitute_scores")),
    # any-player markets (>=2): ride the team goal/SoT distribution x a concentration factor
    (re.compile(r"will any player record (\d+) or more shots on target", re.I),
     lambda m: ParsedQuestion("any_player_sot_ge", threshold=int(m[1]))),
    (re.compile(r"will any player score more than (\d+) goals?", re.I),
     lambda m: ParsedQuestion("any_player_goals_ge", threshold=int(m[1]) + 1)),
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
    (re.compile(r"will (.+?) have more corner kicks than (.+?)\?", re.I),
     lambda m: ParsedQuestion("corners_gt", m[1], m[2], half=None)),
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
    (re.compile(r"will an own goal be scored", re.I),
     lambda m: ParsedQuestion("own_goal")),
    (re.compile(r"will a goal be scored from outside the (?:penalty area|box)", re.I),
     lambda m: ParsedQuestion("goal_outside_box")),
    (re.compile(r"will (.+?) have more shots on target than (.+?) in the second half", re.I),
     lambda m: ParsedQuestion("sot_gt", m[1], m[2], half=2)),
    (re.compile(r"will (.+?) have more shots on target than (.+?) in the first half", re.I),
     lambda m: ParsedQuestion("sot_gt", m[1], m[2], half=1)),
    # full-match catch-all (mirrors the corners_gt one) — must follow the half-specific
    # patterns above. The knockout "...in regulation (90 minutes + stoppage time)?" suffix is
    # stripped upstream, so this catches "Will X have more shots on target than Y?" which was
    # silently dropped in every knockout match (only half-scoped SoT comparisons had a pattern).
    (re.compile(r"will (.+?) have more shots on target than (.+?)\?", re.I),
     lambda m: ParsedQuestion("sot_gt", m[1], m[2], half=None)),
    (re.compile(r"will (.+?) win(?: the match)?\b(?! by)", re.I),
     lambda m: ParsedQuestion("win", m[1])),
    (re.compile(r"will (.+?) score in the second half", re.I),
     lambda m: ParsedQuestion("team_scores_half", m[1], half=2)),
    (re.compile(r"will (.+?) score in the first half", re.I),
     lambda m: ParsedQuestion("team_scores_half", m[1], half=1)),
    (re.compile(r"will (.+?) score a goal \(excluding own goals\)", re.I),
     lambda m: ParsedQuestion("player_scores", m[1])),
    # "both teams ..." MUST precede the greedy player/team patterns below, or the
    # `(.+?)` swallows "both teams" and mis-prices it as a single player's prop.
    (re.compile(r"will both teams have at least (\d+) shots? on target in the (first|second) half", re.I),
     lambda m: ParsedQuestion("both_teams_sot_ge", threshold=int(m[1]), half=1 if m[2].lower() == "first" else 2)),
    (re.compile(r"will both teams have at least (\d+) shots? on target", re.I),
     lambda m: ParsedQuestion("both_teams_sot_ge", threshold=int(m[1]))),
    (re.compile(r"will (.+?) have at least (\d+) shots? on target in the (first|second) half", re.I),
     lambda m: ParsedQuestion("player_sot_ge", m[1], threshold=int(m[2]), half=1 if m[3].lower() == "first" else 2)),
    (re.compile(r"will (.+?) have at least (\d+) shots? on target", re.I),
     lambda m: ParsedQuestion("player_sot_ge", m[1], threshold=int(m[2]))),
    (re.compile(r"will both teams score", re.I),
     lambda m: ParsedQuestion("btts")),
    (re.compile(r"will the match have (\d+) or more (?:total )?goals", re.I),
     lambda m: ParsedQuestion("total_ge", threshold=int(m[1]))),
]


def parse_question(text: str, teams: set[str] | None = None) -> ParsedQuestion | None:
    """Parse a market question. If `teams` (canonical names of the match's two
    teams) is given, a player-prop parse whose subject is actually one of those
    teams is re-routed to the team archetype — e.g. 'Will Haiti have at least 1
    shot on target...' is team_sot_ge, not player_sot_ge."""
    text = text.strip()
    # Knockout-stage phrasing appends "... in regulation (90 minutes + stoppage time)";
    # our models are all 90-minute, so strip the qualifier so the base archetype matches.
    text = re.sub(r"\s*\b(?:in|of) regulation \(\d+ minutes \+ stoppage time\)", "", text, flags=re.I)
    for pattern, build in _PATTERNS:
        m = pattern.search(text)
        if m:
            q = build(m)
            if (teams and q.archetype in _PLAYER_TO_TEAM and q.subject
                    and canonical(q.subject) in teams):
                return ParsedQuestion(_PLAYER_TO_TEAM[q.archetype], q.subject,
                                      threshold=q.threshold, half=q.half)
            return q
    return None
