"""End-to-end prediction pipeline: fixture -> calibrated bot probabilities.

predict_fixture(team_a, team_b) produces every archetype the parser knows,
using models fit on all data up to today. Anchor probabilities (de-vigged
odds), when supplied, are blended per bot profile:
    Bot A "anchor": w_anchor = 0.85 on anchored markets (backtest-fitted)
    Bot B "alpha":  w_anchor = 0.50 (model-leaning, decorrelated from field)
Unanchored markets are model-only for both bots.

Calibration: Platt per archetype fit from the Phase 2 backtest settlements
(data/processed/backtest_phase2_hl730.csv).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from jumpprob.features.elo import compute_elo, current_ratings
from jumpprob.ingest import results as results_mod
from jumpprob.models.calibration import apply_calibration, fit_calibration_table, submit_int
from jumpprob.models.count_model import fit_count_model, p_a_gt_b, p_ge, p_sum_ge
from jumpprob.models.goal_model import fit_goal_model
from jumpprob.models.market_anchor import blend
from jumpprob.normalize.teams import canonical
from jumpprob.simulate import markets

BOT_WEIGHTS = {"A": 0.85, "B": 0.50}
PENALTY_BASE = 0.293  # measured P(>=1 pen/match), 6 tournaments (A6)


class Pipeline:
    def __init__(self):
        self.results_elo = compute_elo(results_mod.load_results())
        today = self.results_elo["date"].max() + pd.Timedelta(days=1)
        self.ratings = current_ratings(self.results_elo)
        self.goal_model = fit_goal_model(self.results_elo, today, 730.0)
        self.calibration = self._load_calibration()
        self.count_model = self._fit_counts()

    def _load_calibration(self):
        path = Path("data/processed/backtest_phase2_hl730.csv")
        if not path.exists():
            return {}
        return fit_calibration_table(pd.read_csv(path))

    def _fit_counts(self):
        path = Path("data/processed/team_half_counts.parquet")
        if not path.exists():
            return None
        th = pd.read_parquet(path)
        th["team_c"] = th["team"].map(canonical)
        th["cards"] = th["yellows"] + th["reds"]
        th["offsides"] = th["pass_offsides"] + th["offside_events"]
        # attach lambdas from current goal model via Elo at match date (coarse:
        # current ratings; count covariate sensitivity to Elo drift is low)
        rows = []
        for (mid, team), g in th.groupby(["match_id", "team_c"]):
            opp = th[(th.match_id == mid) & (th.team_c != team)]["team_c"].iloc[0]
            ra = self.ratings.get(team, 1500.0)
            rb = self.ratings.get(opp, 1500.0)
            la, lb = self.goal_model.intensities(ra, rb)
            for r in g.itertuples():
                rows.append({"period": r.period, "lam_team": la, "lam_opp": lb,
                             "referee": r.referee, "corners": r.corners, "cards": r.cards,
                             "offsides": r.offsides, "shots_on_target": r.shots_on_target,
                             "fouls": r.fouls, "shots": r.shots})  # shots = total (on+off), real data
        return fit_count_model(pd.DataFrame(rows))

    def predict_fixture(
        self,
        team_a: str,
        team_b: str,
        a_home_ground: bool = False,
        referee: str | None = None,
        anchors: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Returns a frame: archetype, p_model, p_bot_a, p_bot_b, sub_a, sub_b."""
        a, b = canonical(team_a), canonical(team_b)
        ra, rb = self.ratings.get(a), self.ratings.get(b)
        if ra is None or rb is None:
            raise KeyError(f"unknown team: {a if ra is None else b}")

        probs = markets.answer_all(self.goal_model, ra, rb, a_home_ground)
        la, lb = self.goal_model.intensities(ra, rb, a_home_ground)
        probs["penalty_awarded"] = PENALTY_BASE

        if self.count_model is not None:
            cm = self.count_model
            def pmf(fam, half, lt, lo):
                return cm.pmf(fam, cm.mu(fam, half, lt, lo, referee if fam == "cards" else None))
            probs["corners_ht_a_gt_b"] = p_a_gt_b(pmf("corners", 1, la, lb), pmf("corners", 1, lb, la))
            probs["cards_2h_ge_2"] = p_sum_ge(pmf("cards", 2, la, lb), pmf("cards", 2, lb, la), 2)
            probs["a_offsides_ge_2"] = p_sum_ge(pmf("offsides", 1, la, lb), pmf("offsides", 2, la, lb), 2)
            probs["b_offsides_ge_2"] = p_sum_ge(pmf("offsides", 1, lb, la), pmf("offsides", 2, lb, la), 2)
            probs["a_sot_gt_b_2h"] = p_a_gt_b(pmf("shots_on_target", 2, la, lb), pmf("shots_on_target", 2, lb, la))

        anchors = anchors or {}
        rows = []
        for arch, p in probs.items():
            p_cal = apply_calibration(p, arch, self.calibration)
            row = {"archetype": arch, "p_model": round(p_cal, 4)}
            for bot, w in BOT_WEIGHTS.items():
                p_bot = blend([anchors[arch], p_cal], [w, 1 - w]) if arch in anchors else p_cal
                row[f"p_bot_{bot}"] = round(p_bot, 4)
                row[f"sub_{bot}"] = submit_int(p_bot)
            rows.append(row)
        return pd.DataFrame(rows)
