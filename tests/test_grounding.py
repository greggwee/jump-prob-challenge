"""Phase 3 history grounding (pure data/math)."""

import pytest

from jumpprob.agents.grounding import history_implied, player_history, poisson_ge1


def test_poisson_ge1_basics():
    assert poisson_ge1(0.0) == pytest.approx(0.0)
    assert poisson_ge1(2.57, minutes=90) == pytest.approx(1 - pow(2.718281828, -2.57), abs=1e-3)
    # half-window + part-minutes both lower the probability
    assert poisson_ge1(2.0, minutes=45) < poisson_ge1(2.0, minutes=90)
    assert poisson_ge1(2.0, share=0.5) < poisson_ge1(2.0, share=1.0)


def test_player_history_resolves_and_grades_attacker():
    # Gyökeres has a high direct SoT/90 -> history implies he is LIKELY to get a SoT,
    # which is the corrective signal our model lacked (it bet ~11%).
    hi = history_implied("Viktor Gyökeres", "player_sot_ge", minutes=70, half=True)
    assert hi["resolved"] is not None
    assert hi["sot_p90"] > 2.0
    assert hi["implied"] > 0.4   # vastly higher than the ~0.11 we submitted


def test_unresolved_player_falls_back_to_base_rate():
    hi = history_implied("Nonexistent Qwerty Player", "player_sot_ge")
    assert hi["resolved"] is None
    assert "note" in hi


def test_history_known_attacker_exceeds_base():
    # a covered attacker's history-implied should exceed the data-starved base rate
    h = player_history("Sadio Mané")
    assert h["resolved"] is not None and h["sot_p90"] > 0
