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


def test_sot_n_or_more_phrasing_routes_like_at_least():
    """Knockout phrasing 'N or more shots on target' must route exactly like 'at least N':
    player by default (book-anchorable prop), team via re-route, both-teams caught first.
    Bug: it defaulted to team_sot_ge, so a player subject failed to resolve -> dropped."""
    from jumpprob.normalize.teams import canonical
    teams = {canonical("Germany"), canonical("Paraguay")}
    assert parse_question("Will Jamal Musiala (Germany) have 2 or more shots on target in "
                          "regulation (90 minutes + stoppage time)?", teams=teams).archetype == "player_sot_ge"
    assert parse_question("Will Germany have 2 or more shots on target?", teams=teams).archetype == "team_sot_ge"
    assert parse_question("Will both teams have 1 or more shots on target?", teams=teams).archetype == "both_teams_sot_ge"
    q = parse_question("Will Kylian Mbappé (France) have 2 or more shots on target?")
    assert q.archetype == "player_sot_ge" and q.threshold == 2


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
    # non-decomposing Latin letters must transliterate the way the feeds do (NFKD+ascii DROPPED
    # them: 'Ødegaard'->'degaard') — the Sørloth/Ødegaard book-join miss class
    assert norm_player("Martin Ødegaard") == "martin odegaard"
    assert norm_player("Alexander Sørloth") == "alexander sorloth"
    assert norm_player("Aleksander Å") == "aleksander a"
    assert norm_player("Łukasz Łł") == "lukasz ll"
    # de-vig a SoT Over/Under 0.5 into P(>=1 SoT)  (1.25 ~ 80%, less the vig)
    event = {"bookmakers": [{"key": "pinnacle", "markets": [
        {"key": "player_shots_on_target", "outcomes": [
            {"description": "Ante Budimir", "name": "Over", "point": 0.5, "price": 1.25},
            {"description": "Ante Budimir", "name": "Under", "point": 0.5, "price": 4.0}]}]}]}
    p = _parse_player_event(event)["ante budimir"]["player_sot_ge"]
    assert 0.7 < p < 0.85
    # ONE-SIDED Over-only SoT (many books quote only the Over for the 0.5 line) must STILL
    # de-vig — the bug was dropping these, which zeroed SoT coverage for whole matches.
    one_sided = {"bookmakers": [{"key": "onexbet", "markets": [
        {"key": "player_shots_on_target", "outcomes": [
            {"description": "Neymar", "name": "Over", "point": 0.5, "price": 1.30},
            {"description": "Neymar", "name": "Over", "point": 1.5, "price": 2.40}]}]}]}
    pn = _parse_player_event(one_sided)
    assert "neymar" in pn and 0.65 < pn["neymar"]["player_sot_ge"] < 0.80   # 1/1.30/1.08 ~ 0.71


def test_r32_new_phrasings_header_and_any_player():
    """R32 template audit findings: 'Will a header goal be scored' was UNPARSED (dropped),
    and 'Will any player score 2 or more goals' mis-hit team_goals_ge with subject
    'any player' -> side resolution failed -> dropped. Both must parse and price."""
    import pandas as pd
    from jumpprob.pipeline import Pipeline
    from jumpprob.submit.runner import Runner
    h = parse_question("Will a header goal be scored in regulation (90 minutes + stoppage time)?")
    assert h.archetype == "goal_header"
    a = parse_question("Will any player score 2 or more goals in regulation (90 minutes + stoppage time)?")
    assert a.archetype == "any_player_goals_ge" and a.threshold == 2
    # the old phrasing still routes the same way
    assert parse_question("Will any player score more than 1 goal (excluding own goals)?").archetype == "any_player_goals_ge"
    # and both price sensibly; header scales with tempo like outside-box
    r = Runner(Pipeline(), pd.read_parquet("data/processed/player_match.parquet"), mc=None)
    strong, weak = r.context("Brazil", "Haiti"), r.context("Qatar", "Haiti")
    assert r.answer(h, strong) > r.answer(h, weak)
    assert 0.15 < r.answer(h, strong) < 0.75
    assert 0.05 < r.answer(a, strong) < 0.60
    # 3+ has no concentration data -> None (skip), never a wrong price
    a3 = parse_question("Will any player score 3 or more goals?")
    assert a3.archetype == "any_player_goals_ge" and r.answer(a3, strong) is None


def test_own_goal_and_outside_box_models():
    """#3 own goal = flat WC base rate (thinning by goals was TESTED and lost to flat); #2
    outside-box = Poisson thinning of the goal intensity, so it scales with match tempo."""
    import pandas as pd
    from jumpprob.pipeline import Pipeline
    from jumpprob.submit.runner import Runner
    r = Runner(Pipeline(), pd.read_parquet("data/processed/player_match.parquet"), mc=None)
    og = parse_question("Will an own goal be scored in regulation (90 minutes + stoppage time)?")
    ob = parse_question("Will a goal be scored from outside the penalty area in regulation "
                        "(90 minutes + stoppage time)?")
    strong, weak = r.context("Brazil", "Haiti"), r.context("Qatar", "Haiti")
    # own goal: FLAT — identical across a high- and low-tempo match, near the WC base rate
    assert r.answer(og, strong) == r.answer(og, weak)
    assert 0.05 < r.answer(og, strong) < 0.16
    # outside-box: scales with expected goals (more goals -> more likely one is from outside)
    assert r.answer(ob, strong) > r.answer(ob, weak)
    assert 0.10 < r.answer(ob, strong) < 0.60


def test_player_prop_anchor_weight_near_book():
    """The Jimenez/Kane under-weighting: a data-gapped model (7-9%) at 15% logit weight
    dragged anchored scorer props 5-9 pts below the book (Jimenez book 27 -> submitted 22,
    Kane 54 -> 45, both out=YES). On 51 settled anchored props Brier is monotone toward the
    book, so player props blend at w_p = 1 - PLAYER_MODEL_FRAC*(1-w_market) ~= 0.97 live:
    the submission must land within ~1.5 pts of the book, not 5+ below it."""
    import pandas as pd
    from jumpprob.pipeline import Pipeline
    from jumpprob.submit.runner import Runner, PLAYER_MODEL_FRAC
    from jumpprob.normalize.teams import canonical
    from jumpprob.ingest.odds_api import norm_player

    assert PLAYER_MODEL_FRAC <= 0.25          # the fix: players keep only a sliver of model
    r = Runner(Pipeline(), pd.read_parquet("data/processed/player_match.parquet"), mc=None)
    t = "Will Raúl Jiménez (Mexico) score a goal (excluding own goals) in regulation (90 minutes + stoppage time)?"
    pp = {(canonical("Mexico"), canonical("Ecuador")):
          {norm_player("Raúl Jiménez"): {"player_scores": 0.27}}}
    rows = [{"id": "m1", "lobby_id": "L", "match": {"name": "Mexico vs Ecuador"}, "question": t}]
    out = r.predictions_for(rows, player_props=pp)[0]
    assert abs(out[0]["p_bot_a"] - 0.27) < 0.02, out[0]   # was 0.22 at the old 0.85 weight


def test_r16_template_handlers():
    """The 6 R16 archetypes price sanely: shootout in the knockout-plausible band and higher
    for even matchups; exact-totals sum consistently with the >= family; goal-each-half below
    either half alone; cards>goals a real probability; leads-at-any-point tracks first-goal;
    full-90 at the sub-pattern base."""
    import pandas as pd
    from jumpprob.pipeline import Pipeline
    from jumpprob.submit.runner import Runner
    r = Runner(Pipeline(), pd.read_parquet("data/processed/player_match.parquet"), mc=None)
    even, lop = r.context("France", "Croatia"), r.context("Brazil", "Haiti")
    P = lambda q, c: r.answer(parse_question(q), c)
    # shootout: plausible band, more likely in an even tie than a blowout
    ps_even, ps_lop = P("Will the match be decided by a penalty shootout?", even), \
                      P("Will the match be decided by a penalty shootout?", lop)
    assert 0.02 < ps_lop < ps_even < 0.30
    # exact totals: P(exactly 0) == 1 - P(>=1); peak near the mode; sums below 1
    p0 = P("Will exactly 0 goals be scored in regulation (90 minutes + stoppage time)?", even)
    from jumpprob.simulate import markets as mk
    assert abs(p0 - (1 - mk.p_total_ge(even.grid, 1))) < 1e-6
    p2 = P("Will the match finish with exactly 2 total goals in regulation (90 minutes + stoppage time)?", even)
    assert 0.15 < p2 < 0.35
    # goal in each half: below either single half's P(>=1 goal), above their product floor
    geh = P("Will at least one goal be scored in each half in regulation (90 minutes + stoppage time)?", even)
    assert 0.3 < geh < 0.85
    # cards > goals: valid prob, higher in a low-scoring matchup than a goal-fest
    cg_even = P("Will there be more total cards than total goals in regulation (90 minutes + stoppage time)?", even)
    cg_lop = P("Will there be more total cards than total goals in regulation (90 minutes + stoppage time)?", lop)
    assert 0.05 < cg_lop < cg_even < 0.95
    # leads at any point ~ first goal + comeback margin; underdog << favorite
    la_f = P("Will France hold a lead at any point in the match (excluding a penalty shootout)?", even)
    la_c = P("Will Croatia hold a lead at any point in the match (excluding a penalty shootout)?", even)
    assert la_f > la_c and 0.05 < la_c < la_f < 0.95
    # full 90: the sub-pattern base rate
    pf = P("Will Christian Pulisic (United States) play the entire match in regulation (90 minutes + stoppage time)?", even)
    assert abs(pf - 0.42) < 1e-9


def test_p5_skipped_questions_carry_reasons():
    """P5 template audit: any open question that does not price must surface in `skipped`
    with a reason — the header-goal and any-player-2+ drops were invisible because skipped
    was discarded. A new phrasing must be catchable the first run it appears."""
    import pandas as pd
    from jumpprob.pipeline import Pipeline
    from jumpprob.submit.runner import Runner
    r = Runner(Pipeline(), pd.read_parquet("data/processed/player_match.parquet"), mc=None)
    rows = [{"id": "x1", "lobby_id": "L", "match": {"name": "Spain vs Austria"},
             "question": "Will a bicycle-kick goal be scored in regulation (90 minutes + stoppage time)?"},
            {"id": "x2", "lobby_id": "L", "match": {"name": "Spain vs Austria"},
             "question": "Will Spain win in regulation (90 minutes + stoppage time)?"}]
    out, skipped = r.predictions_for(rows)
    assert len(out) == 1 and len(skipped) == 1
    assert skipped[0]["reason"] == "unparsed" and "bicycle" in skipped[0]["question"]


def test_p2_fouls_offsides_near_flat():
    """P2: our per-match variation on fouls_gt/offsides_ge is ANTI-signal (flat beats both
    us and the crowd on all settled-with-crowd data), so the risk gate pins them near their
    calibrated bases: fouls 0.55 (crowd~realized agree), offsides 0.40 (crowd .49 shrunk
    toward realized .36). Deviations survive at <=~20%% strength; direction preserved."""
    import json
    from jumpprob.models import strategy
    g = json.loads(open("data/processed/risk_gate.json").read())
    b = json.loads(open("data/processed/base_rates.json").read())
    assert g["fouls_gt"] == 0.15 and g["offsides_ge"] == 0.15
    assert b["fouls_gt"] == 0.55 and b["offsides_ge"] == 0.40
    for arch, base in [("fouls_gt", 0.55), ("offsides_ge", 0.40)]:
        hi = strategy.apply_risk(base + 0.25, arch, g, b)
        lo = strategy.apply_risk(base - 0.25, arch, g, b)
        assert base < hi < base + 0.08          # hugged to base, direction kept
        assert base - 0.08 < lo < base


def test_crowd_clamp_p1():
    """P1 catastrophic-deviation clamp: unbacked extremes pull back to within 25 pts of the
    archetype's crowd prior; book anchors and fringe player LOWS (the validated edge) are
    exempt; matchup-wide / thin priors never clamp."""
    from jumpprob.submit.audit import crowd_clamp
    prior = {"both_teams_sot_ge": {"n": 6, "mean": 66.5, "std": 6.5},
             "substitute_scores": {"n": 6, "mean": 31.0, "std": 3.0},
             "player_sot_ge": {"n": 48, "mean": 46.0, "std": 9.9},
             "win": {"n": 25, "mean": 60.0, "std": 17.6},          # too wide -> excluded
             "own_goal": {"n": 1, "mean": 14.0, "std": None}}      # too thin -> excluded
    # the group-era catastrophe: 9 vs crowd ~66 -> clamped to the 25-pt boundary
    p, info = crowd_clamp("both_teams_sot_ge", 9, False, prior)
    assert info is not None and p == 42
    # the substitute 69-vs-29 catastrophe -> clamped to 56
    p, info = crowd_clamp("substitute_scores", 69, False, prior)
    assert info is not None and p == 56
    # fringe player LOW: 40 pts below the prior but WHITELISTED (the +EV extreme class)
    p, info = crowd_clamp("player_sot_ge", 6, False, prior)
    assert info is None and p == 6
    # player HIGH deviations are still clamped (no edge there)
    p, info = crowd_clamp("player_sot_ge", 85, False, prior)
    assert info is not None and p == 71
    # book-anchored: never clamped, whatever the value
    p, info = crowd_clamp("both_teams_sot_ge", 9, True, prior)
    assert info is None and p == 9
    # wide/thin priors and unknown archetypes: no-op
    assert crowd_clamp("win", 95, False, prior)[1] is None
    assert crowd_clamp("own_goal", 90, False, prior)[1] is None
    assert crowd_clamp("goal_header", 90, False, prior)[1] is None
    # moderate deviations (the +EV band) are untouched
    p, info = crowd_clamp("both_teams_sot_ge", 50, False, prior)
    assert info is None and p == 50


def test_anchor_regression_guard():
    """The Diallo 50->24 KO-lock guard: a run whose book prop vanished must not revert a
    previously anchored player prop to the bare model — but ONLY with ledger evidence the
    market was actually book-anchored before. Without it the guard kept a stale mis-parsed
    'DR Congo scores' 8% against the pipeline's corrected 42 (out=YES, biggest R32 loss)."""
    from jumpprob.submit.audit import is_anchor_regression as g
    # Diallo: existing anchored 50, this run bookless (market==pure), PRIOR ANCHOR -> KEEP
    assert g("player_sot_ge", 50, 23, 24, was_anchored=True) is True
    # DR Congo: existing 8 far from model 42, this run bookless, but NEVER book-anchored
    # -> the correction must go through (guard must NOT fire)
    assert g("player_scores", 8, 42, 42, was_anchored=False) is False
    assert g("player_scores", 8, 42, 42) is False               # default: no evidence -> no block
    # this run DID anchor (market 55 != pure 23) -> a real update, let it through
    assert g("player_sot_ge", 50, 23, 55, was_anchored=True) is False
    # existing is itself near the bare model (not anchored) -> nothing to protect
    assert g("player_sot_ge", 24, 23, 24, was_anchored=True) is False
    # non-player archetypes and missing existing are never guarded
    assert g("win", 70, 40, 40, was_anchored=True) is False
    assert g("player_scores", None, 8, 8, was_anchored=True) is False


def test_team_scores_reroutes_to_team_goals():
    """'Will <TEAM> score a goal (excluding own goals)' must price as the team scoring >=1
    (~40-75%), not as an unknown PLAYER (~8%) — the DR Congo miss."""
    from jumpprob.normalize.teams import canonical
    teams = {canonical("England"), canonical("DR Congo")}
    q = parse_question("Will DR Congo score a goal (excluding own goals) in regulation "
                       "(90 minutes + stoppage time)?", teams=teams)
    assert q.archetype == "team_goals_ge" and q.threshold == 1
    # a real player is untouched
    p = parse_question("Will Harry Kane (England) score a goal (excluding own goals)?", teams=teams)
    assert p.archetype == "player_scores"
    # and the answer is a sane team-scoring probability
    import pandas as pd
    from jumpprob.pipeline import Pipeline
    from jumpprob.submit.runner import Runner
    r = Runner(Pipeline(), pd.read_parquet("data/processed/player_match.parquet"), mc=None)
    prob = r.answer(q, r.context("England", "DR Congo"))
    assert 0.25 < prob < 0.85, prob
    """The submission-time guard that surfaces a model data-gap OR a book/question mismatch:
    a large pure-model vs de-vigged-book gap (the Haaland 7-vs-52 signature). Flags >= min_pts,
    biggest first; small gaps and the empty frame are clean."""
    import pandas as pd
    from jumpprob.submit.audit import book_vs_model_divergence
    df = pd.DataFrame([
        {"market_id": "a", "match": "X", "archetype": "player_scores", "question": "q",
         "sub_pure": 7, "sub_market": 52, "sub_a": 43},     # Haaland-class data gap, |gap|=45
        {"market_id": "b", "match": "X", "archetype": "win", "question": "q",
         "sub_pure": 48, "sub_market": 46, "sub_a": 46},    # |gap|=2 -> not flagged
        {"market_id": "c", "match": "X", "archetype": "total_ge", "question": "q",
         "sub_pure": 40, "sub_market": 64, "sub_a": 62},    # |gap|=24
    ])
    flagged = book_vs_model_divergence(df, min_pts=20)
    assert list(flagged.market_id) == ["a", "c"]           # >=20pt gaps, biggest first
    assert int(flagged.iloc[0].book_gap) == 45
    assert flagged.iloc[0].logit_gap > flagged.iloc[1].logit_gap   # scale-aware ordering tracks
    assert book_vs_model_divergence(df.iloc[:0], 20).empty        # empty frame is safe


def test_resolve_player_robust_name_join():
    """The book often carries a fuller name than the platform subject (middle names,
    initials) — the Haaland bug: 'Erling Haaland' missed the book key 'erling braut haaland'
    and silently lost the 52% scorer anchor. resolve_player must bridge that, while NEVER
    guessing on an ambiguous (same-surname) collision."""
    from jumpprob.ingest.odds_api import resolve_player
    book = {"erling braut haaland": {"player_scores": 0.52}, "martin odegaard": {"player_scores": 0.18}}
    # exact, middle-name, initial, and team-qualifier subjects all resolve to the same entry
    assert resolve_player("Martin Odegaard", book)["player_scores"] == 0.18
    assert resolve_player("Erling Haaland", book)["player_scores"] == 0.52          # middle name bridged
    assert resolve_player("E Haaland", book)["player_scores"] == 0.52               # initial bridged
    assert resolve_player("Erling Haaland (Norway)", book)["player_scores"] == 0.52 # team qualifier
    # a different player / single token / empty book -> {} (fall back to the model)
    assert resolve_player("Lionel Messi", book) == {}
    assert resolve_player("Haaland", book) == {}                                    # one token: too risky
    assert resolve_player("Erling Haaland", {}) == {}
    # AMBIGUOUS surname collision must NOT guess -> {}
    ambig = {"ousmane diomande": {"player_scores": 0.1}, "oumar diomande": {"player_scores": 0.2}}
    assert resolve_player("O Diomande", ambig) == {}                                # 'o' prefixes both
    assert resolve_player("Ousmane Diomande", ambig)["player_scores"] == 0.1        # full first disambiguates
    # NICKNAME alias: the platform's 'Trezeguet' is the book's registered 'mahmoud hassan'
    nick = {"mahmoud hassan": {"player_scores": 0.22}, "mohamed salah": {"player_scores": 0.5}}
    assert resolve_player("Mahmoud Trezeguet (Egypt)", nick)["player_scores"] == 0.22
    # HISPANIC DOUBLE SURNAME: the paternal surname sits before a trailing maternal one
    dbl = {"luis fernando diaz marulanda": {"player_scores": 0.31},
           "luis javier suarez charris": {"player_scores": 0.28}}
    assert resolve_player("Luis Díaz (Colombia)", dbl)["player_scores"] == 0.31
    assert resolve_player("Luis Suárez (Colombia)", dbl)["player_scores"] == 0.28
    # a colliding surname in ANY position must still refuse to guess
    coll = {"luis fernando diaz marulanda": {"player_scores": 0.31},
            "luis diaz romero": {"player_scores": 0.1}}
    assert resolve_player("Luis Díaz", coll) == {}


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


def test_total_count_markets_from_real_data():
    """Total shots (real team_half_counts.shots, NOT SoT/0.37), total offsides, and
    both-team cards parse and answer off the count model."""
    import pandas as pd
    from jumpprob.pipeline import Pipeline
    from jumpprob.submit.runner import Runner

    assert parse_question("Will there be 20 or more total shots (on and off target)?").archetype == "total_shots_ge"
    assert parse_question("Will there be 3 or more offside calls?").archetype == "offsides_total_ge"
    assert parse_question("Will both teams receive at least one card?").archetype == "both_teams_cards"
    pipe = Pipeline()
    assert "shots" in pipe.count_model.params          # real total-shots family was fit
    r = Runner(pipe, pd.read_parquet("data/processed/player_match.parquet"), mc=None)
    ctx = r.context("Brazil", "Japan")
    for q in ["Will there be 20 or more total shots (on and off target)?",
              "Will there be 3 or more offside calls?",
              "Will both teams receive at least one card?"]:
        p = r.answer(parse_question(q), ctx)
        assert p is not None and 0.0 < p < 1.0, q
    # modeling total shots from real data is not the SoT/0.37 underestimate (true ratio 0.315)
    cm = pipe.count_model
    mu_real = cm.mu("shots", 1, 1.5, 1.3) + cm.mu("shots", 2, 1.5, 1.3)
    mu_est = (cm.mu("shots_on_target", 1, 1.5, 1.3) + cm.mu("shots_on_target", 2, 1.5, 1.3)) / 0.37
    assert mu_real > mu_est


def test_event_and_any_player_markets():
    """Stoppage goals (lambda-scaled share), sub base rates, and any-player markets
    (team distribution x concentration) parse and behave monotonically."""
    import pandas as pd
    from jumpprob.pipeline import Pipeline
    from jumpprob.submit.runner import Runner
    for text, arch in [
        ("Will a goal be scored in first-half stoppage (added) time?", "goal_1h_stoppage"),
        ("Will a goal be scored in second-half stoppage time?", "goal_2h_stoppage"),
        ("Will a substitution be made before halftime?", "sub_before_ht"),
        ("Will a substitute score a goal in regulation (90 minutes + stoppage time)?", "substitute_scores"),
        ("Will any player record 2 or more shots on target?", "any_player_sot_ge"),
        ("Will any player score more than 1 goal (excluding own goals)?", "any_player_goals_ge"),
    ]:
        assert parse_question(text).archetype == arch, text
    r = Runner(Pipeline(), pd.read_parquet("data/processed/player_match.parquet"), mc=None)
    brace_q = parse_question("Will any player score more than 1 goal (excluding own goals)?")
    strong = r.answer(brace_q, r.context("Brazil", "Haiti"))
    weak = r.answer(brace_q, r.context("Haiti", "Qatar"))
    assert strong > weak                          # more expected goals -> brace more likely
    ctx = r.context("Brazil", "Japan")
    assert (r.answer(parse_question("Will a goal be scored in second-half stoppage time?"), ctx) >
            r.answer(parse_question("Will a goal be scored in first-half stoppage (added) time?"), ctx))


def test_player_sot_threshold_not_the_one_plus_line():
    """Regression for the Brobbey 60%-off-a-66%-1+line bug. 'N or more shots on target'
    must be P(>=N), strictly below the '1 or more' (book Over-0.5) line. The book SoT anchor
    (only ever the 0.5 line) must NOT be stamped raw onto a >=2 question; it is converted to
    P(>=2) via the Poisson intensity (see test_player_multi_sot_poisson_from_one_plus) so the
    >=2 line still tracks the book's player ranking instead of falling to a flat model."""
    import pandas as pd
    from jumpprob.pipeline import Pipeline
    from jumpprob.submit.runner import Runner
    from jumpprob.normalize.teams import canonical
    from jumpprob.ingest.odds_api import norm_player

    teams = {canonical("Netherlands"), canonical("Morocco")}
    t1 = "Will Brian Brobbey have at least 1 shot on target in regulation (90 minutes + stoppage time)?"
    t2 = "Will Brian Brobbey have 2 or more shots on target in regulation (90 minutes + stoppage time)?"
    q1, q2 = parse_question(t1, teams=teams), parse_question(t2, teams=teams)
    assert q1.archetype == q2.archetype == "player_sot_ge"
    assert (q1.threshold or 1) == 1 and q2.threshold == 2

    r = Runner(Pipeline(), pd.read_parquet("data/processed/player_match.parquet"), mc=None)
    ctx = r.context("Netherlands", "Morocco")
    p1, p2 = r.answer(q1, ctx), r.answer(q2, ctx)        # P(>=1 SoT) vs P(>=2 SoT)
    assert p1 is not None and p2 is not None
    assert p2 < p1 - 0.03, (p1, p2)                      # was IDENTICAL before the fix

    # book anchor = the Over-0.5 (>=1 SoT) price; anchors the >=1 line, dropped for >=2
    book = 0.662
    pp = {(canonical("Netherlands"), canonical("Morocco")):
          {norm_player("Brian Brobbey"): {"player_sot_ge": book}}}
    rows = [{"id": "m1", "lobby_id": "L", "match": {"name": "Netherlands vs Morocco"}, "question": t1},
            {"id": "m2", "lobby_id": "L", "match": {"name": "Netherlands vs Morocco"}, "question": t2}]
    by_q = {o["question"]: o["p_bot_a"] for o in r.predictions_for(rows, player_props=pp)[0]}
    assert by_q[t1] > p1 + 0.05                          # >=1 IS pulled up toward the 0.66 book
    assert by_q[t2] < by_q[t1] - 0.15                    # >=2 is strictly, materially below >=1
    assert by_q[t2] < 0.45                               # not the 0.66 1+ line stamped on
    # >=2 is anchored to the Poisson conversion of the 0.66 book line (mu=-ln(1-0.66)=1.09 ->
    # P>=2=0.295), blended at w_market -> well above a flat undifferentiated model, below P>=1.
    assert by_q[t2] > 0.15                               # NOT dropped to a flat ~17% prior


def test_player_multi_sot_poisson_from_one_plus():
    """The fix for the flat-SoT under-pricing that lost R32 (Mbappe 2+ SoT priced 19 like
    every player while the crowd ranked him 63, out=1 -> the single biggest loss). The book
    only quotes the 1+ (Over-0.5) line; we derive P(>=k SoT) from it via the Poisson intensity
    mu=-ln(1-P(>=1)), re-using the book's player ranking for the higher thresholds."""
    from jumpprob.submit.runner import book_sot_ge_from_one_plus as f

    # k<=1 -> None (caller uses the raw 1+ line directly)
    assert f(0.80, 1) is None and f(0.80, 0) is None
    # monotone in the threshold: P(>=1) > P(>=2) > P(>=3)
    elite = 0.834                                        # Mbappe's live R32 1+ line
    p2, p3 = f(elite, 2), f(elite, 3)
    assert elite > p2 > p3 > 0.0
    # elite 1+ line yields a MATERIALLY higher P(>=2) than a fringe one (the whole point:
    # the crowd differentiates, the flat model did not)
    fringe = 0.25
    assert f(elite, 2) > f(fringe, 2) + 0.3
    # live R32 numbers track the crowd (vs our flat 17-19, crowd 32-63):
    assert 0.52 < f(0.834, 2) < 0.56                     # Mbappe -> ~0.54 (crowd 63, out=1)
    assert 0.27 < f(0.658, 2) < 0.31                     # Sorloth -> ~0.29 (crowd 33, out=0)
    assert 0.12 < f(0.478, 2) < 0.16                     # Isak -> ~0.14 (crowd 32, out=0)
    # monotone in the 1+ line too: a higher 1+ line -> higher P(>=2)
    assert f(0.90, 2) > f(0.70, 2) > f(0.50, 2) > f(0.30, 2)
    # degenerate inputs stay in [0,1] without blowing up
    assert 0.0 <= f(0.999999, 2) <= 1.0 and 0.0 <= f(0.000001, 2) <= 1.0


def test_uncertain_markets_routed_to_llm_shadow():
    """The 2026-06-29 recovered uncertain markets are wired to the LLM as BLEND
    (logged in shadow now, promotable to live via a gate weight). Market-anchored
    goal markets stay EXCLUDE so the LLM can never move them."""
    from jumpprob.agents import panel as P
    for a in ["substitute_scores", "goal_2h_stoppage", "any_player_goals_ge",
              "total_shots_ge", "both_teams_cards", "cards_post_hb2", "sub_before_ht"]:
        assert P.route(a)[1] == P.BLEND, a
    assert P.route("win")[1] == P.EXCLUDE          # sharp, market-priced
    assert P.route("advance")[1] == P.EXCLUDE      # derived from the sharp 90-min line
    # shadow contract: with no gate weight, the expert is logged but never blended
    panel = {"m1": {"p_ensemble": 0.90, "archetype": "substitute_scores"}}
    p_final, info = P.apply_expert("m1", "substitute_scores", 0.32, panel, gate={})
    assert p_final == 0.32 and info["expert"] == 0.9


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


def test_r16_review_ship_now_fixes():
    """Adversarial-review SHIP-NOW set: grid templates registered as goal family (anchored
    pass-through, no base dilution); leads_at_any_point includes the ET term; the risk gate
    bypasses HALF-scoped questions (its bases are full-match quantities)."""
    import pandas as pd
    from jumpprob.pipeline import Pipeline
    from jumpprob.submit.runner import Runner, GOAL_ARCHS
    for a in ("pen_shootout", "total_exact", "goal_each_half", "leads_at_any_point"):
        assert a in GOAL_ARCHS, a
    r = Runner(Pipeline(), pd.read_parquet("data/processed/player_match.parquet"), mc=None)
    ctx = r.context("France", "Croatia")
    # leads now = first-goal + 0.05 comeback + ET term: strictly above the old +0.03 form
    la = r.answer(parse_question("Will Croatia hold a lead at any point in the match "
                                 "(excluding a penalty shootout)?"), ctx)
    import numpy as np
    lt, lo = (ctx.lam_b, ctx.lam_a)
    old = (lt / (lt + lo)) * (1 - np.exp(-(lt + lo))) + 0.03
    assert la > old + 0.015
    # half-scoped questions BYPASS the risk gate (its bases are full-match quantities):
    # spy on _risk — it must fire for the full-match phrasing, never for the halftime one
    calls = []
    orig = r._risk
    r._risk = lambda a, p: (calls.append(a), orig(a, p))[1]
    r.risk_on = True
    mk_row = lambda q: [{"id": "m1", "lobby_id": "L", "match": {"name": "France vs Croatia"}, "question": q}]
    r.predictions_for(mk_row("At halftime, will both teams have at least 1 shot on target?"))
    assert calls == []                                   # half-scoped: gate bypassed
    r.predictions_for(mk_row("Will both teams have at least 1 shot on target?"))
    assert "both_teams_sot_ge" in calls                  # full-match: gate still applies
    r._risk = orig
