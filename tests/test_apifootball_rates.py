"""Tests for API-Football rate aggregation (pure / no network).
Covers the join's real failure modes: search returns several players, a player has
stats across club + national team, penalties inflate goals, zero-minute cameos."""

import pytest

from jumpprob.ingest.apifootball import (
    _profile_full_name, _search_token, players_to_rates_af)
from jumpprob.normalize.players import fold_name


class TestProfileFullName:
    def test_clean_firstlast_when_name_is_abbrev(self):
        # 'R. Mahrez' nickname adds nothing -> clean first+last
        pl = {"name": "R. Mahrez", "firstname": "Riyad Karim", "lastname": "Mahrez"}
        assert _profile_full_name(pl) == "Riyad Karim Mahrez"

    def test_appends_distinct_nickname_surname(self):
        # Trézéguet: nickname surname is in `name`, not first/last -> must be joinable
        pl = {"name": "Trézéguet", "firstname": "Mahmoud Ahmed Ibrahim", "lastname": "Hassan"}
        folded = fold_name(_profile_full_name(pl))   # how the join sees it
        assert "trezeguet" in folded and "mahmoud" in folded

    def test_falls_back_to_name_when_no_first_last(self):
        assert _profile_full_name({"name": "Pelé"}) == "Pelé"


def _stat(league_id, league_name, minutes, shots, on, goals, pens=0, assists=0,
          key=0, pos="Attacker", apps=10):
    return {"league": {"id": league_id, "name": league_name},
            "games": {"minutes": minutes, "position": pos, "appearences": apps},
            "shots": {"total": shots, "on": on},
            "goals": {"total": goals, "assists": assists},
            "penalty": {"scored": pens}, "passes": {"key": key}}


def _payload(query, hits):
    # hits: list of (player_name, [statistics...])
    return {"query": query,
            "response": [{"player": {"name": nm}, "statistics": st} for nm, st in hits]}


class TestSearchToken:
    def test_uses_longest_token_min4(self):
        assert _search_token("Sadio Mané") == "sadio"        # both >=4, longest
        assert _search_token("Riyad Mahrez") == "mahrez"
        assert _search_token("Xi Wu") == "xiwu"              # no token >=4 -> folded


class TestRatesAggregation:
    def test_per90_and_sot(self):
        p = _payload("Akram Afif",
                     [("Akram Afif", [_stat(305, "Qatar Stars League", 1800, 60, 24, 12, 2, 8, 30)])])
        df = players_to_rates_af([p])
        r = df.iloc[0]
        assert r["minutes"] == 1800
        assert r["sot_p90"] == pytest.approx(24 / (1800 / 90))   # 1.2 — the new field
        assert r["shots_p90"] == pytest.approx(3.0)
        assert r["npg_p90"] == pytest.approx((12 - 2) / 20)       # penalties removed

    def test_prefers_hole_league_over_national_team(self):
        # player has a national-team line with more minutes, but we want the club league
        p = _payload("Salem Al-Dawsari", [("Salem Al-Dawsari", [
            _stat(1, "World Cup Qualification", 2000, 5, 2, 1),     # NT, most minutes
            _stat(307, "Saudi Pro League", 1500, 45, 20, 10)])])   # club
        df = players_to_rates_af([p])
        assert df.iloc[0]["league"] == "Saudi Pro League"
        assert df.iloc[0]["minutes"] == 1500

    def test_picks_max_minutes_when_no_hole_league(self):
        p = _payload("X Y", [("X Y", [
            _stat(99, "Some Cup", 200, 3, 1, 0),
            _stat(98, "Some League", 1600, 30, 12, 8)])])
        assert players_to_rates_af([p]).iloc[0]["minutes"] == 1600

    def test_disambiguates_search_hits_by_name_overlap(self):
        # search 'Mané' returns two players; pick the token-overlapping one
        p = _payload("Sadio Mané", [
            ("Wilfried Mané", [_stat(40, "Championship", 900, 5, 1, 0)]),
            ("Sadio Mané", [_stat(307, "Saudi Pro League", 1700, 50, 22, 14)])])
        df = players_to_rates_af([p])
        assert df.iloc[0]["player_name"] == "Sadio Mané"

    def test_rejects_when_no_name_overlap(self):
        p = _payload("Akram Afif", [("Completely Different", [_stat(307, "x", 900, 5, 1, 0)])])
        assert players_to_rates_af([p]).empty

    def test_robust_to_empty_and_zero_minutes(self):
        assert players_to_rates_af([]).empty
        assert players_to_rates_af([_payload("A B", [])]).empty
        # zero-minute cameo -> dropped (no rate signal)
        p = _payload("A B", [("A B", [_stat(40, "Championship", 0, 0, 0, 0)])])
        assert players_to_rates_af([p]).empty
