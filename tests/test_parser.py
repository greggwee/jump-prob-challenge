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
