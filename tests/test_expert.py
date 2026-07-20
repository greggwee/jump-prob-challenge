"""The out-of-band expert caller — pure parts (no live API call).

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


def test_forecast_review_fields_default_inert():
    # the review fields are additive: an old-style forecast (no factors) is still valid
    # and reads as "agrees with the model" (none evidence, empty factor list).
    f = Forecast(probability=0.3)
    assert f.evidence_strength == "none" and f.factors == []
    f2 = Forecast(probability=0.8, evidence_strength="hard",
                  factors=[expert.Factor(name="starter ruled out", direction="down",
                                         evidence="official team sheet")])
    assert f2.evidence_strength == "hard" and f2.factors[0].direction == "down"


def test_agg_evidence_takes_strongest():
    assert expert.agg_evidence(["none", "soft", "hard"]) == "hard"
    assert expert.agg_evidence(["none", "soft"]) == "soft"
    assert expert.agg_evidence([]) == "none"
    assert expert.agg_evidence(["bogus", "none"]) == "none"   # unknown -> none


def test_parse_evidence_strength_from_web_text():
    assert expert._parse_evidence_strength("...\nPROBABILITY: 0.2\nEVIDENCE_STRENGTH: hard") == "hard"
    assert expert._parse_evidence_strength("EVIDENCE_STRENGTH: SOFT") == "soft"
    assert expert._parse_evidence_strength("no marker here") == "none"


def test_review_role_system_prompt_frames_as_reviewer():
    s = build_system("lineup")
    assert "REVIEWING" in s
    assert "evidence_strength" in s            # the hard/soft discipline is stated
    assert "UNCHANGED" in s                    # the "no concrete reason -> don't move" rule


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


# --- web-search forecast (Tier 3) — pure parts ---

def test_parse_prob_fraction_percent_and_last():
    assert expert._parse_prob("...\nPROBABILITY: 0.34") == pytest.approx(0.34)
    assert expert._parse_prob("PROBABILITY: 42%") == pytest.approx(0.42)
    assert expert._parse_prob("PROBABILITY: 70") == pytest.approx(0.70)   # >1 -> /100
    assert expert._parse_prob("PROBABILITY: 0.9\nPROBABILITY: 0.30") == pytest.approx(0.30)  # last wins
    assert expert._parse_prob("no probability here") is None
    assert expert._parse_prob("PROBABILITY: 0.99") == pytest.approx(0.97)  # clipped


def test_forecast_web_no_key(monkeypatch):
    monkeypatch.setattr(expert, "_client", None, raising=False)
    monkeypatch.setattr(expert, "_no_key", True, raising=False)
    prob, text = expert.forecast_web("Will X score?", 0.3, {"matchup": "A vs B"})
    assert prob is None and text == ""


def test_web_search_tool_shape():
    assert expert.WEB_SEARCH_TOOL["type"] == "web_search_20260209"


def test_web_panel_default_one_wide():
    assert expert.WEB_PANEL == ["claude-opus-4-8"]  # 1-wide default for budget


def test_web_forecast_no_key(monkeypatch):
    monkeypatch.setattr(expert, "_client", None, raising=False)
    monkeypatch.setattr(expert, "_no_key", True, raising=False)
    res = expert.web_forecast("Will X have 1+ SoT?", 0.2, {"matchup": "A vs B"})
    assert res["p_ensemble"] is None and res["n"] == 0
