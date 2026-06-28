"""Tests for question parser, calibration, and submission formatting."""

import numpy as np
import pytest

from jumpprob.models.calibration import apply_calibration, fit_platt, submit_int
from jumpprob.submit.parser import parse_question

BRIEF_QUESTIONS = [
    ("At halftime, will Czechia have more corner kicks than South Korea?",
     "corners_gt", "Czechia", 1),
    ("Will South Korea be caught offside 2 or more times?", "offsides_ge", "South Korea", None),
    ("Will there be 2 or more total cards shown in the second half?", "cards_ge", None, 2),
    ("Will both teams score AND the match have 3 or more total goals?", "btts_and_total_ge", None, None),
    ("Will a penalty kick be awarded in the match?", "penalty_awarded", None, None),
    ("Will South Korea have more shots on target than Czechia in the second half?",
     "sot_gt", "South Korea", 2),
    ("Will South Korea win the match?", "win", "South Korea", None),
    ("Will South Korea score in the second half?", "team_scores_half", "South Korea", 2),
    ("Will Son Heung-min score a goal (excluding own goals)?", "player_scores", "Son Heung-min", None),
    ("Will Patrik Schick have at least 1 shot on target?", "player_sot_ge", "Patrik Schick", None),
]


def test_all_brief_questions_parse():
    for text, arch, subject, half in BRIEF_QUESTIONS:
        q = parse_question(text)
        assert q is not None, text
        assert q.archetype == arch, text
        if subject:
            assert q.subject == subject, text
        if half:
            assert q.half == half, text


def test_unknown_question_returns_none():
    assert parse_question("Will the referee wear gloves?") is None


def test_both_teams_sot_not_mis_parsed_as_player():
    """Regression: 'both teams' must not be swallowed by the greedy player-SoT
    regex (was mis-priced ~9% vs ~75% fair, a per-match point bleed)."""
    q = parse_question("Will both teams have at least 1 shot on target in the second half?")
    assert q is not None and q.archetype == "both_teams_sot_ge"
    assert q.threshold == 1 and q.half == 2
    q2 = parse_question("Will both teams have at least 1 shot on target?")
    assert q2.archetype == "both_teams_sot_ge" and q2.half is None
    # a real player prop still parses as a player prop
    assert parse_question("Will Son Heung-min have at least 1 shot on target?").archetype == "player_sot_ge"


def test_extremize_skips_risk_gate_archetypes():
    """No-edge archetypes (owned by the risk gate) must NOT be extremized, even when
    their family earns leverage (count k=1.3) — else stage 5 amplifies the very
    deviation stage 8 then shrinks, and the two layers fight (Flag 2, 2026-06-24)."""
    from jumpprob.submit.runner import Runner
    class Stub:  # _extremize only reads self.risk / gate / field_sharpness / aggression
        risk = {"offsides_ge": 0.5, "fouls_gt": 0.5, "both_teams_sot_ge": 0.3}
        gate = {"count": 1.3}
        field_sharpness = {"count": 0.4}
        aggression = 1.0
    s = Stub()
    # a risk-gate archetype is left untouched despite count k=1.3
    assert Runner._extremize(s, "offsides_ge", "count", 0.70, False) == 0.70
    assert Runner._extremize(s, "both_teams_sot_ge", "count", 0.80, False) == 0.80
    # a count archetype WITHOUT a risk-gate entry is still levered (deviation amplified)
    assert Runner._extremize(s, "sot_gt", "count", 0.70, False) > 0.70


def test_player_prop_name_join_and_devig():
    """Player-prop market anchor: subject<->book name join + Over/Under de-vig."""
    from jumpprob.ingest.odds_api import norm_player, _parse_player_event
    # team suffix + accents normalize to the same key as the bare name
    assert norm_player("Kai Havertz (Germany)") == norm_player("Kai Havertz") == "kai havertz"
    assert norm_player("Viktor Gyökeres") == "viktor gyokeres"
    # de-vig a SoT Over/Under 0.5 into P(>=1 SoT)  (1.25 ~ 80%, less the vig)
    event = {"bookmakers": [{"key": "pinnacle", "markets": [
        {"key": "player_shots_on_target", "outcomes": [
            {"description": "Ante Budimir", "name": "Over", "point": 0.5, "price": 1.25},
            {"description": "Ante Budimir", "name": "Under", "point": 0.5, "price": 4.0}]}]}]}
    p = _parse_player_event(event)["ante budimir"]["player_sot_ge"]
    assert 0.7 < p < 0.85


def test_drop_audit_recovered_archetypes():
    """2026-06-28 drop audit: knockout/derived archetypes that were silently dropped
    (no prediction from either bot) now parse. The goal ones are grid-derivable in
    runner.answer; cards-in-half reuses cards_ge; red_card is a constant."""
    cases = [
        ("Will Argentina score the first goal of the match?", "first_goal_match", "Argentina", None, None),
        ("Will the United States win by 2 or more goals in regulation (90 minutes + stoppage time)?",
         "win_margin", "the United States", 2, None),
        ("Will Argentina keep a clean sheet in regulation (90 minutes + stoppage time)?",
         "clean_sheet", "Argentina", None, None),
        ("Will Brazil score in both halves of regulation (90 minutes + stoppage time)?",
         "score_both_halves", "Brazil", None, None),
        ("Will regulation (90 minutes + stoppage time) end in a tie?", "draw", None, None, None),
        ("Will the first half produce 2 or more goals?", "half_total_ge", None, 2, 1),
        ("Will the second half produce more goals than the first half in regulation (90 minutes + stoppage time)?",
         "h2_more_goals_than_h1", None, None, None),
        ("Will a card be shown in the first half?", "cards_ge", None, 1, 1),
        ("Will a red card be shown in the match?", "red_card", None, None, None),
    ]
    for text, arch, subj, thr, half in cases:
        q = parse_question(text)
        assert q is not None and q.archetype == arch, text
        if subj:
            assert q.subject == subj, text
        if thr:
            assert q.threshold == thr, text
        if half:
            assert q.half == half, text
    # margin must not be mis-read as a plain win (the win regex excludes "win by")
    assert parse_question("Will Spain win the match?").archetype == "win"


def test_hydration_window_markets():
    """Hydration/cooling-break window markets parse, and each window is a proper
    sub-share of its family's minute-intensity profile (the analytic recovery)."""
    import numpy as np
    from pathlib import Path
    from jumpprob.models.dynamics import Dynamics, BUCKET
    from jumpprob.submit.runner import _WINDOW
    for text, arch in [
        ("Will a goal be scored before the first hydration break?", "goal_pre_hb1"),
        ("Will a goal be scored after the second hydration break in regulation (90 minutes + stoppage time)?",
         "goal_post_hb2"),
        ("Will 2 or more corner kicks be taken before the first hydration break?", "corners_pre_hb1"),
        ("Will a card be shown after the second hydration break, including any extra time?", "cards_post_hb2"),
        ("Will either team be ruled offside before the first hydration break?", "offsides_pre_hb1"),
    ]:
        q = parse_question(text)
        assert q is not None and q.archetype == arch, text
    dyn = Dynamics.from_json(Path("data/processed/minute_dynamics.json"))
    for arch, (fam, lo, hi) in _WINDOW.items():
        w = np.exp(np.asarray(dyn.alpha[fam])[BUCKET])
        frac = float(w[lo:hi].sum() / w.sum())
        assert 0.0 < frac < 1.0, arch          # a real probability mass, not all/none


def test_advance_estimator():
    """advance = win THIS knockout tie incl. ET/pens. Even match -> 0.5; a favorite
    sits between its 90-min win prob and 1; and the two teams' advance probs sum to
    1 (exactly one advances) -- a coherence property of the decomposition."""
    import pandas as pd
    import jumpprob.simulate.markets as mk
    from jumpprob.pipeline import Pipeline
    from jumpprob.submit.runner import Runner

    assert parse_question("Will Japan advance to the Round of 16?").archetype == "advance"
    assert parse_question("Will Spain advance to the quarterfinals?").archetype == "advance"
    r = Runner(Pipeline(), pd.read_parquet("data/processed/player_match.parquet"), mc=None)

    def adv(ta, tb):
        ctx = r.context(ta, tb)
        return r.answer(parse_question(f"Will {ta} advance to the Round of 16?"), ctx), mk.p_win(ctx.grid, "a")

    p_even, _ = adv("Brazil", "Brazil")
    assert abs(p_even - 0.5) < 1e-6                     # symmetry
    p_fav, w_fav = adv("Brazil", "Haiti")
    assert w_fav < p_fav < 1.0                          # draw share lifts the favorite above win90
    p_dog, _ = adv("Haiti", "Brazil")
    assert abs((p_fav + p_dog) - 1.0) < 1e-6            # exactly one team advances


def test_canonical_strips_definite_article():
    """'the United States' / 'the Netherlands' subjects must resolve to the same
    canonical as the bare name, else the side is unresolved and the market drops."""
    from jumpprob.normalize.teams import canonical
    assert canonical("the United States") == canonical("United States") == "United States"
    assert canonical("the Netherlands") == "Netherlands"
    assert canonical("USA") == "United States"   # alias still applies after strip
    assert canonical("Germany") == "Germany"     # a normal name is untouched


def test_platt_recovers_overconfidence():
    """Generate predictions that are systematically overconfident; Platt must
    fit a < 1 (shrink toward 0.5) and improve Brier."""
    rng = np.random.default_rng(11)
    n = 20_000
    true_p = rng.uniform(0.1, 0.9, n)
    # overconfident report: logit doubled
    logit_t = np.log(true_p / (1 - true_p))
    p_over = 1 / (1 + np.exp(-2.0 * logit_t))
    y = (rng.random(n) < true_p).astype(float)
    a, b = fit_platt(p_over, y)
    assert a < 0.75  # detects overconfidence (true a = 0.5)
    cal = np.array([apply_calibration(p, "x", {"x": (a, b)}) for p in p_over[:2000]])
    raw_brier = np.mean((p_over[:2000] - y[:2000]) ** 2)
    cal_brier = np.mean((cal - y[:2000]) ** 2)
    assert cal_brier < raw_brier


def test_submit_int_bounds():
    assert submit_int(0.001) == 1
    assert submit_int(0.9999) == 99
    assert submit_int(0.755) == 76  # 0.755*100 = 75.5 rounds to 76
    assert submit_int(0.5) == 50
