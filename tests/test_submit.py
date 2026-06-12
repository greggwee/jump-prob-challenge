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
