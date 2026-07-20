"""Integration tests against the real downloaded data.

These require data/raw to be populated (run scripts/fetch_data.py first) and
are skipped otherwise — unit tests stay green without network access.
"""

from pathlib import Path

import pandas as pd
import pytest

from jumpprob.ingest import results as results_mod
from jumpprob.ingest import statsbomb as sb_mod
from jumpprob.normalize.teams import canonical

DATA_PRESENT = (
    Path("data/raw/martj42/results.csv").exists()
    and Path("data/raw/statsbomb/matches_wc2022.json").exists()
)

pytestmark = pytest.mark.skipif(not DATA_PRESENT, reason="raw data not fetched yet")


@pytest.fixture(scope="module")
def results() -> pd.DataFrame:
    return results_mod.load_results()


@pytest.fixture(scope="module")
def sb_matches() -> pd.DataFrame:
    return sb_mod.load_matches()


def test_results_pass_validation(results):
    # load_results raises on validation failure; reaching here means it passed
    assert len(results) > 40_000  # corpus has 48k+ matches
    assert results["date"].is_monotonic_increasing


def test_goalscorers_minutes_sane():
    gs = results_mod.load_goalscorers()
    assert gs["minute"].between(1, 125).all()
    # second-half share of goals: lore says ~54-56%; assert a sane band and
    # record the actual value for the half-split parameter (assumption A2)
    second_half_share = (gs["minute"] > 45).mean()
    assert 0.50 < second_half_share < 0.62


def test_statsbomb_match_counts(sb_matches):
    # load_matches raises if any tournament deviates from its known match count
    assert len(sb_matches) == 64 + 64 + 51 + 51 + 32 + 52


def test_every_statsbomb_team_resolves(results, sb_matches):
    """Entity resolution: every StatsBomb team name must canonicalize to a name
    that exists in the martj42 corpus. Failure output lists the gaps to add to
    ALIASES — this test IS the alias-table completeness check."""
    known = set(results["home_team"]) | set(results["away_team"])
    sb_teams = set(sb_matches["home_team"]) | set(sb_matches["away_team"])
    unresolved = sorted(t for t in sb_teams if canonical(t) not in known)
    assert not unresolved, f"add aliases for: {unresolved}"


def test_statsbomb_scores_agree_with_martj42(results, sb_matches):
    """Cross-validation between independent sources: for WC 2022
    group-stage matches, scores must agree exactly after name resolution.
    Knockouts excluded: martj42 records 90'/120' conventions differ."""
    wc = sb_matches[(sb_matches.tournament == "wc2022") & (sb_matches.stage == "Group Stage")].copy()
    wc["home_c"] = wc["home_team"].map(canonical)
    wc["away_c"] = wc["away_team"].map(canonical)

    merged = wc.merge(
        results[["date", "home_team", "away_team", "home_score", "away_score"]],
        left_on=["date", "home_c", "away_c"],
        right_on=["date", "home_team", "away_team"],
        suffixes=("_sb", "_m42"),
    )
    assert len(merged) >= 40, f"only matched {len(merged)}/48 group matches by key — name/date drift"
    mismatched = merged[
        (merged["home_score_sb"] != merged["home_score_m42"])
        | (merged["away_score_sb"] != merged["away_score_m42"])
    ]
    assert mismatched.empty, mismatched[["date", "home_c", "away_c"]].to_string()


def test_elo_sanity_against_known_world(results):
    """Top of our Elo table should look like the real elite. We don't pin exact
    values (formula details differ slightly from eloratings.net) but the top-10
    must be dominated by consensus elite nations."""
    from jumpprob.features.elo import current_ratings

    top10 = set(current_ratings(results).head(10).index)
    elite = {
        "Spain", "Argentina", "France", "England", "Brazil",
        "Portugal", "Netherlands", "Germany", "Italy", "Belgium",
        "Uruguay", "Colombia", "Croatia", "Morocco", "Japan",
    }
    overlap = len(top10 & elite)
    assert overlap >= 7, f"Elo top10 looks wrong: {sorted(top10)}"
