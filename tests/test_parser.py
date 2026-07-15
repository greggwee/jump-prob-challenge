"""Archetype-parser correctness — the classifications every gate depends on."""

from jumpprob.submit.parser import parse_question


def test_team_sot_phrased_as_player_is_rerouted():
    # 'Will <team> have at least N shots on target' reads as a player prop; with the
    # match teams known it must re-route to team_sot_ge (the live regression).
    q = "Will Haiti have at least 1 shot on target in the second half?"
    assert parse_question(q).archetype == "player_sot_ge"            # bare parse (offline)
    p = parse_question(q, teams={"Haiti"})
    assert p.archetype == "team_sot_ge" and p.half == 2


def test_real_player_sot_unchanged():
    p = parse_question("Will Viktor Gyökeres have at least 1 shot on target?",
                       teams={"Sweden", "Netherlands"})
    assert p.archetype == "player_sot_ge" and p.subject == "Viktor Gyökeres"


def test_both_teams_sot_precedes_player():
    # the fixed greedy-regex bug: 'both teams' must not be swallowed into a player prop
    p = parse_question("Will both teams have at least 1 shot on target in the first half?")
    assert p.archetype == "both_teams_sot_ge"


def test_player_vs_team_threshold_phrasing():
    # the live path always passes the match teams; "N or more" / "at least N" both default
    # to player_sot_ge and re-route to team_sot_ge when the subject is one of the teams.
    from jumpprob.normalize.teams import canonical
    teams = {canonical("Croatia"), canonical("Brazil")}
    assert parse_question("Will Croatia have 6 or more shots on target?", teams=teams).archetype == "team_sot_ge"
    assert parse_question("Will Son have at least 2 shots on target?", teams=teams).archetype == "player_sot_ge"


def test_no_edge_archetypes_parse():
    assert parse_question("Will Panama be caught offside 2 or more times?").archetype == "offsides_ge"
    assert parse_question("Will England commit more fouls than Croatia?").archetype == "fouls_gt"
    assert parse_question("Will Panama receive more cards than Ghana?").archetype == "cards_gt"


def test_player_ga_and_scores():
    assert parse_question("Will Son Heung-min score or assist a goal (excluding own goals)?").archetype == "player_ga"
    assert parse_question("Will Kane score a goal (excluding own goals)?").archetype == "player_scores"


def test_unparseable_returns_none():
    assert parse_question("Some unrelated sentence about nothing.") is None


def test_knockout_regulation_suffix_is_stripped():
    # the platform appends "...in regulation (90 minutes + stoppage time)" on knockout
    # questions; our models are 90-minute, so the base archetype must still match.
    assert parse_question("Will France win in regulation (90 minutes + stoppage time)?").archetype == "win"
    assert parse_question("Will Brazil score 2 or more goals in regulation (90 minutes + stoppage time)?").archetype == "team_goals_ge"
    assert parse_question("Will there be 4 or more total cards shown in regulation (90 minutes + stoppage time)?").archetype == "cards_ge"
    assert parse_question("Will there be 8 or more total shots on target in regulation (90 minutes + stoppage time)?").archetype == "total_sot_ge"


def test_win_phrasings_and_margin_disambiguation():
    assert parse_question("Will Argentina win?").archetype == "win"
    assert parse_question("Will Argentina win the match?").archetype == "win"
    # "win by N goals" is a MARGIN market, NOT a win — must not be mis-classified as win
    m = parse_question("Will the United States win by 2 or more goals?")
    assert m is None or m.archetype != "win"


def test_ht_leading_and_full_match_corners_phrasings():
    assert parse_question("Will Argentina be ahead at halftime?").archetype == "ht_leading"
    assert parse_question("At halftime, will Spain be winning?").archetype == "ht_leading"
    assert parse_question("Will Ivory Coast have more corner kicks than Norway?").archetype == "corners_gt"


def test_own_goal_and_outside_box_parse():
    # the two novel MEX v ECU markets (were unparsed -> dropped)
    assert parse_question("Will an own goal be scored in regulation "
                          "(90 minutes + stoppage time)?").archetype == "own_goal"
    assert parse_question("Will a goal be scored from outside the penalty area in "
                          "regulation (90 minutes + stoppage time)?").archetype == "goal_outside_box"


def test_r16_templates_parse():
    # the 6 new R16 question types caught by the P5 template audit (2026-07-04)
    assert parse_question("Will the match be decided by a penalty shootout?").archetype == "pen_shootout"
    q = parse_question("Will exactly 1 goal be scored in regulation (90 minutes + stoppage time)?")
    assert q.archetype == "total_exact" and q.threshold == 1
    q2 = parse_question("Will the match finish with exactly 2 total goals in regulation (90 minutes + stoppage time)?")
    assert q2.archetype == "total_exact" and q2.threshold == 2
    q3 = parse_question("Will Paraguay hold a lead at any point in the match (excluding a penalty shootout)?")
    assert q3.archetype == "leads_at_any_point" and q3.subject == "Paraguay"
    assert parse_question("Will there be more total cards than total goals in regulation "
                          "(90 minutes + stoppage time)?").archetype == "cards_gt_goals"
    q4 = parse_question("Will Christian Pulisic (United States) play the entire match in regulation "
                        "(90 minutes + stoppage time)?")
    assert q4.archetype == "plays_full_match" and "Pulisic" in q4.subject
    assert parse_question("Will at least one goal be scored in each half in regulation "
                          "(90 minutes + stoppage time)?").archetype == "goal_each_half"


def test_full_match_sot_comparison_phrasing():
    # 'more shots on target than' with NO half (the knockout regulation suffix is stripped)
    # must parse like the corners equivalent — it was silently DROPPED (MEX v ECU 12/15) since
    # only half-scoped SoT comparisons had a pattern.
    q = parse_question("Will Mexico have more shots on target than Ecuador in "
                       "regulation (90 minutes + stoppage time)?")
    assert q is not None and q.archetype == "sot_gt" and q.half is None
    assert parse_question("Will Mexico have more shots on target than Ecuador?").archetype == "sot_gt"
    # precedence preserved: the half-scoped forms still carry their half
    assert parse_question("Will Mexico have more shots on target than Ecuador in the second half?").half == 2
    assert parse_question("In the first half, will Mexico have more shots on target than Ecuador?").half == 1


def test_halftime_both_teams_sot_is_first_half_scoped():
    # the platform's single worst category (-7.8 RBP): "At halftime, ..." fell to the
    # no-half patterns and priced as FULL-MATCH (~.88) vs first-half truth (~.60).
    q = parse_question("At halftime, will both teams have at least 1 shot on target?")
    assert q.archetype == "both_teams_sot_ge" and q.half == 1 and q.threshold == 1
    q2 = parse_question("At halftime, will both teams have 2 or more shots on target?")
    assert q2.archetype == "both_teams_sot_ge" and q2.half == 1 and q2.threshold == 2
    # full-match + second-half phrasings unchanged
    assert parse_question("Will both teams have at least 1 shot on target?").half is None
    assert parse_question("Will both teams have at least 1 shot on target in the second half?").half == 2


def test_any_team_player_sot_is_team_scoped():
    # "any Portugal player" must NOT price as a single unknown player (flat 15)
    q = parse_question("Will any Portugal player have 2 or more shots on target in "
                       "regulation (90 minutes + stoppage time)?")
    assert q.archetype == "any_team_player_sot_ge" and q.subject == "Portugal" and q.threshold == 2
    # the match-level phrasing is untouched
    assert parse_question("Will any player record 2 or more shots on target?").archetype == "any_player_sot_ge"


def test_qf_wave2_phrasings_route_correctly():
    """2026-07-10 QF phrasings: substitute score-or-assist must NOT be swallowed by player_ga
    (was live at 15 vs crowd ~35-40); 'total cards' without 'shown'; 0-0 HT; HT substitution."""
    assert parse_question("Will a substitute score or assist a goal (excluding own goals) in "
                          "regulation (90 minutes + stoppage time)?").archetype == "substitute_ga"
    assert parse_question("Will a substitute score a goal (excluding own goals) in regulation "
                          "(90 minutes + stoppage time)?").archetype == "substitute_scores"
    q = parse_question("Will there be 4 or more total cards in regulation (90 minutes + stoppage time)?")
    assert q.archetype == "cards_ge" and q.threshold == 4
    assert parse_question("Will the match be 0-0 at halftime?").archetype == "ht_scoreless"
    assert parse_question("Will either team make a substitution at halftime?").archetype == "sub_at_ht"


def test_sf_phrasings_route_correctly():
    """SF (FRA/ESP) phrasings 2026-07-11: mid-hydration-window goal, offsides-combined, ET variant."""
    assert parse_question("Will a goal be scored after the first hydration break but before the "
                          "second hydration break?").archetype == "goal_between_hbs"
    q = parse_question("Will there be 4 or more offsides (both teams combined) in regulation "
                       "(90 minutes + stoppage time)?")
    assert q.archetype == "offsides_total_ge" and q.threshold == 4
    assert parse_question("Will the match be tied at the end of regulation (90 minutes + stoppage "
                          "time) and go to extra time?").archetype == "extra_time"


def test_sf2_phrasings_route_correctly():
    """ENG/ARG SF2 phrasings (2026-07-15): total-SoT-combined, lead-at-halftime, the
    card-before-goal competing-risks variant, and either-half stoppage goal."""
    q = parse_question("Will there be 8 or more total shots on target (both teams combined) in "
                       "regulation (90 minutes + stoppage time)?")
    assert q.archetype == "total_sot_ge" and q.threshold == 8
    assert parse_question("Will England lead at halftime?").archetype == "ht_leading"
    assert parse_question("Will a card be shown before the first goal of the match in regulation "
                          "(90 minutes + stoppage time)?").archetype == "first_card_before_goal"
    assert parse_question("Will a goal be scored during first- or second-half stoppage time in "
                          "regulation?").archetype == "goal_stoppage"
