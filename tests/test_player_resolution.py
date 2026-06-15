"""Tests for player-name resolution and Understat rate aggregation (Phase 1).
Pure / no network — covers the join's real-world failure modes."""

import pytest

from jumpprob.ingest.understat import players_to_rates
from jumpprob.normalize.players import build_index, fold_name, resolve, tokens


class TestFoldName:
    def test_accents_stripped(self):
        assert fold_name("Viktor Gyökeres") == "viktor gyokeres"
        assert fold_name("Kylian Mbappé") == "kylian mbappe"
        assert fold_name("Mário Figueira Fernandes") == "mario figueira fernandes"

    def test_punctuation_and_case(self):
        assert fold_name("N'Golo Kanté") == "ngolo kante"
        assert fold_name("  SON   Heung-min ") == "son heung min"   # hyphen -> space, collapse
        assert fold_name("O'Brien") == "obrien"

    def test_empty_and_garbage(self):
        assert fold_name("") == ""
        assert fold_name("   ") == ""
        assert fold_name("!!!") == ""


class TestResolve:
    def setup_method(self):
        self.exact, self.tok = build_index(
            ["Heung-Min Son", "Viktor Gyökeres", "Cody Gakpo", "Bruno Fernandes"])

    def test_exact_after_folding(self):
        assert resolve("Viktor Gyokeres", self.exact, self.tok) == ("Viktor Gyökeres", "exact")

    def test_token_order_independent(self):
        # market 'Son Heung-min' vs Understat 'Heung-Min Son'
        name, method = resolve("Son Heung-min", self.exact, self.tok)
        assert name == "Heung-Min Son" and method == "tokenset"

    def test_miss(self):
        assert resolve("Lionel Messi", self.exact, self.tok) == (None, "none")
        assert resolve("", self.exact, self.tok) == (None, "none")

    def test_no_false_positive_on_partial(self):
        # 'Fernandes' alone must NOT resolve to 'Bruno Fernandes' (avoid wrong join)
        assert resolve("Fernandes", self.exact, self.tok)[0] is None

    def test_subset_match_middle_name(self):
        # market 'Jonathan David' vs Understat 'Jonathan Christian David'
        exact, tok = build_index(["Jonathan Christian David", "Riyad Mahrez"])
        name, method = resolve("Jonathan David", exact, tok)
        assert name == "Jonathan Christian David" and method == "subset"

    def test_subset_rejected_when_ambiguous(self):
        # two candidates both contain {david} superset -> ambiguous -> no match
        exact, tok = build_index(["Jonathan Christian David", "David Luiz Moreira"])
        assert resolve("David", exact, tok)[0] is None  # 1 token anyway
        # 'Jonathan David' is subset of only one -> still resolves
        assert resolve("Jonathan David", exact, tok)[1] == "subset"


class TestRatesAggregation:
    def _player(self, name, time, shots, goals, npg, xg, npxg, assists, xa, kp, pos="F"):
        return {"player_name": name, "position": pos, "team_title": "X", "games": "10",
                "time": str(time), "shots": str(shots), "goals": str(goals), "npg": str(npg),
                "xG": str(xg), "npxG": str(npxg), "assists": str(assists), "xA": str(xa),
                "key_passes": str(kp)}

    def test_per90_math(self):
        df = players_to_rates([[self._player("A B", 900, 30, 9, 8, 10.0, 8.0, 4, 5.0, 20)]])
        r = df.iloc[0]
        assert r["minutes"] == 900
        assert r["shots_p90"] == pytest.approx(30 / (900 / 90))   # 3.0
        assert r["xG_p90"] == pytest.approx(1.0)
        assert r["xA_p90"] == pytest.approx(0.5)

    def test_aggregates_same_player_across_sources(self):
        # same player in two league-seasons -> volumes sum, per-90 over total minutes
        s1 = [self._player("Erling Haaland", 900, 20, 10, 10, 12.0, 12.0, 2, 2.0, 8)]
        s2 = [self._player("Erling Haaland", 900, 40, 20, 18, 24.0, 22.0, 4, 4.0, 12)]
        df = players_to_rates([s1, s2])
        assert len(df) == 1
        r = df.iloc[0]
        assert r["minutes"] == 1800 and r["goals"] == 30
        assert r["npg_p90"] == pytest.approx(28 / (1800 / 90))    # 1.4

    def test_robust_to_bad_and_empty(self):
        assert players_to_rates([]).empty
        assert players_to_rates([[]]).empty
        # missing/garbage numeric fields -> treated as 0, no crash
        bad = [{"player_name": "C D", "time": "x", "shots": None}]
        df = players_to_rates([bad])
        assert len(df) == 1 and df.iloc[0]["shots"] == 0.0

    def test_blank_names_dropped(self):
        df = players_to_rates([[self._player("", 500, 5, 1, 1, 1, 1, 1, 1, 1)]])
        assert df.empty
