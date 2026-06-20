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
    assert parse_question("Will Croatia have 6 or more shots on target?").archetype == "team_sot_ge"
    assert parse_question("Will Son have at least 2 shots on target?").archetype == "player_sot_ge"


def test_no_edge_archetypes_parse():
    assert parse_question("Will Panama be caught offside 2 or more times?").archetype == "offsides_ge"
    assert parse_question("Will England commit more fouls than Croatia?").archetype == "fouls_gt"
    assert parse_question("Will Panama receive more cards than Ghana?").archetype == "cards_gt"


def test_player_ga_and_scores():
    assert parse_question("Will Son Heung-min score or assist a goal (excluding own goals)?").archetype == "player_ga"
    assert parse_question("Will Kane score a goal (excluding own goals)?").archetype == "player_scores"


def test_unparseable_returns_none():
    assert parse_question("Some unrelated sentence about nothing.") is None
