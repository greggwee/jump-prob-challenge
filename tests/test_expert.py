"""Phase L0: the out-of-band expert caller — pure parts (no live API call).

The live call path was verified manually (Gyokeres SoT 0.11 -> 0.44). Here we
test the prompt construction, the Pydantic schema, and graceful degradation when
no key is present — all offline so the suite stays fast and key-free.
"""

import pytest

from jumpprob.agents import expert
from jumpprob.agents.expert import Forecast, build_system, build_user


def test_forecast_schema_bounds():
    f = Forecast(probability=0.42, rationale="elite volume")
    assert f.probability == pytest.approx(0.42)
    with pytest.raises(Exception):
        Forecast(probability=1.5)  # out of [0,1]


def test_system_prompt_pre_match_and_strict():
    for role in ("tactical", "lineup"):
        s = build_system(role)
        assert "PRE-MATCH" in s
        assert "STRICT" in s


def test_lineup_system_mentions_minutes_and_lineup():
    s = build_system("lineup")
    assert "minutes" in s.lower()
    assert "lineup" in s.lower()
    assert "regress" in s.lower()  # the anti-underpricing instruction


def test_user_prompt_injects_baseline_and_grounding():
    g = {"resolved": "Viktor Gyökeres", "sot_p90": 2.57, "implied": 0.65, "minutes": 2400}
    u = build_user("Will X have 1+ SoT in the second half?", model_pct=0.11, grounding=g)
    assert "11%" in u                 # our baseline prior is shown
    assert "2.57" in u                # the track record is injected
    assert "Viktor Gyökeres" in u
    assert "YES" in u


def test_user_prompt_without_grounding():
    u = build_user("Will there be over 9.5 corners?", model_pct=0.5, grounding=None)
    assert "50%" in u
    assert "Track record" not in u    # no grounding block when None


def test_no_key_yields_none(monkeypatch):
    # force the no-key path: client() -> None, so forecast_one returns None cleanly.
    monkeypatch.setattr(expert, "_client", None, raising=False)
    monkeypatch.setattr(expert, "_no_key", True, raising=False)
    assert expert.client() is None
    assert expert.forecast_one("Will X score?", 0.3, "claude-opus-4-8") is None


def test_panel_forecast_all_none_is_none(monkeypatch):
    monkeypatch.setattr(expert, "_client", None, raising=False)
    monkeypatch.setattr(expert, "_no_key", True, raising=False)
    res = expert.panel_forecast("Will X score?", 0.3, role="tactical")
    assert res["p_ensemble"] is None
    assert res["n"] == 0
    assert all(v is None for v in res["models"].values())


def test_panel_selects_lineup_panel_for_lineup_role():
    assert expert.LINEUP_PANEL == ["claude-opus-4-8", "claude-sonnet-4-6"]
    assert "claude-haiku-4-5" in expert.PANEL  # full panel is 3-wide
