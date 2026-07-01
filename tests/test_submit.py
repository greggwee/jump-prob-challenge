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


def test_anchor_regression_guard():
    """The Diallo 50->24 KO-lock guard: a run whose book prop vanished must not revert a
    previously anchored player prop to the bare model, but genuine updates still pass."""
    from jumpprob.submit.audit import is_anchor_regression as g
    # Diallo: existing anchored 50, this run has no book (market==pure 23) -> KEEP (guard fires)
    assert g("player_sot_ge", 50, 23, 24) is True
    # this run DID anchor (market 55 != pure 23) -> a real update, let it through
    assert g("player_sot_ge", 50, 23, 55) is False
    # existing is itself near the bare model (not anchored) -> nothing to protect
    assert g("player_sot_ge", 24, 23, 24) is False
    # non-player archetypes and missing existing are never guarded
    assert g("win", 70, 40, 40) is False
    assert g("player_scores", None, 8, 8) is False
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
