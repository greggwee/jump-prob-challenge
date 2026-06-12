"""Unit tests for the Elo engine on synthetic data — no network needed."""

import pandas as pd
import pytest

from jumpprob.features.elo import (
    HOME_ADV,
    INITIAL_RATING,
    compute_elo,
    current_ratings,
    goal_multiplier,
    k_factor,
)


def _frame(rows):
    df = pd.DataFrame(
        rows,
        columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_k_factor_categories():
    assert k_factor("FIFA World Cup") == 60
    assert k_factor("FIFA World Cup qualification") == 40
    assert k_factor("UEFA Euro") == 50
    assert k_factor("Copa América") == 50
    assert k_factor("Friendly") == 20
    assert k_factor("King's Cup") == 30  # unknown -> minor tournament


def test_goal_multiplier():
    assert goal_multiplier(0) == 1.0
    assert goal_multiplier(1) == 1.0
    assert goal_multiplier(2) == 1.5
    assert goal_multiplier(3) == pytest.approx(14 / 8)
    assert goal_multiplier(5) == pytest.approx(16 / 8)


def test_zero_sum_and_direction():
    """Winner gains exactly what loser drops; equal teams at neutral venue split 50/50 expectation."""
    df = _frame([("2020-01-01", "A", "B", 1, 0, "Friendly", True)])
    ratings = current_ratings(df)
    assert ratings["A"] + ratings["B"] == pytest.approx(2 * INITIAL_RATING)
    # equal ratings, neutral: We = 0.5, K=20, G=1 -> delta = 10
    assert ratings["A"] == pytest.approx(INITIAL_RATING + 10)


def test_home_advantage_shrinks_home_win_reward():
    home = _frame([("2020-01-01", "A", "B", 1, 0, "Friendly", False)])
    neutral = _frame([("2020-01-01", "A", "B", 1, 0, "Friendly", True)])
    gain_home = current_ratings(home)["A"] - INITIAL_RATING
    gain_neutral = current_ratings(neutral)["A"] - INITIAL_RATING
    assert gain_home < gain_neutral  # home win was more expected -> smaller reward
    assert HOME_ADV == 100.0


def test_point_in_time_property():
    """elo_home/elo_away are PRE-match ratings: match 1 must show 1500 for both,
    and match 2 must show exactly the post-match-1 ratings."""
    df = _frame(
        [
            ("2020-01-01", "A", "B", 2, 0, "Friendly", True),
            ("2020-02-01", "A", "B", 0, 0, "Friendly", True),
        ]
    )
    out = compute_elo(df)
    assert out.loc[0, "elo_home"] == INITIAL_RATING
    assert out.loc[0, "elo_away"] == INITIAL_RATING
    # K=20, G(2)=1.5, We=0.5 -> delta = 15
    assert out.loc[1, "elo_home"] == pytest.approx(1515.0)
    assert out.loc[1, "elo_away"] == pytest.approx(1485.0)


def test_draw_moves_ratings_toward_each_other():
    df = _frame(
        [
            ("2020-01-01", "A", "B", 3, 0, "Friendly", True),  # A jumps ahead
            ("2020-02-01", "A", "B", 1, 1, "Friendly", True),  # draw: A overperformed expectation? No — A is now favourite, draw is underperformance
        ]
    )
    ratings = current_ratings(df)
    out = compute_elo(df)
    # after match 1 A > B; the draw must pull A down and B up
    assert ratings["A"] < out.loc[1, "elo_home"]
    assert ratings["B"] > out.loc[1, "elo_away"]
