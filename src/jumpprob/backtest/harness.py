"""Phase 2 backtest: replay six tournaments point-in-time.

For each tournament: fit the goal model on all internationals strictly before
its first match (Elo features are pre-match by construction), predict every
group-stage match's archetype probabilities, settle against actual outcomes,
and compare Brier against a climatological baseline.

Group stage only at this phase: martj42 knockout scores can include extra time,
which would corrupt 90-minute settlement (flagged in PLAN.md cleaning rules).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from jumpprob.features.elo import compute_elo
from jumpprob.ingest import results as results_mod
from jumpprob.ingest import statsbomb as sb_mod
from jumpprob.models.goal_model import fit_goal_model
from jumpprob.normalize.teams import canonical
from jumpprob.simulate import markets

ARCHETYPES = [
    "win_a", "win_b", "draw", "total_ge_2", "total_ge_3",
    "btts", "btts_and_3plus", "a_scores_2h", "b_scores_2h", "a_scores_1h",
]


@dataclass
class BacktestData:
    results_elo: pd.DataFrame
    fixtures: pd.DataFrame          # group-stage fixtures with elo + scores
    half_goals: pd.DataFrame        # per (date, team) goals by half, reconciled matches only


def _resolve_goal_team_convention(gs: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Determine whether goalscorers.csv credits own goals to the scorer's team
    or the benefiting team, by reconciling per-match goal sums with final
    scores under both conventions and keeping the one that reconciles better.
    Returns goals with a `credited_team` column under the winning convention.
    """
    sample = gs.merge(
        results[["date", "home_team", "away_team", "home_score", "away_score"]],
        on=["date", "home_team", "away_team"],
        how="inner",
    )

    def reconcile_rate(credit_other_on_og: bool) -> tuple[float, pd.DataFrame]:
        g = sample.copy()
        if credit_other_on_og:
            other = np.where(g["team"] == g["home_team"], g["away_team"], g["home_team"])
            g["credited_team"] = np.where(g["own_goal"], other, g["team"])
        else:
            g["credited_team"] = g["team"]
        per = g.groupby(["date", "home_team", "away_team", "credited_team"]).size().rename("n").reset_index()
        per_h = per[per["credited_team"] == per["home_team"]].set_index(["date", "home_team", "away_team"])["n"]
        match_scores = g.drop_duplicates(["date", "home_team", "away_team"]).set_index(
            ["date", "home_team", "away_team"]
        )[["home_score"]]
        joined = match_scores.join(per_h).fillna(0)
        return float((joined["home_score"] == joined["n"]).mean()), g

    rate_as_is, g_as_is = reconcile_rate(False)
    rate_flipped, g_flipped = reconcile_rate(True)
    return g_flipped if rate_flipped > rate_as_is else g_as_is


def load_backtest_data() -> BacktestData:
    results = results_mod.load_results()
    results_elo = compute_elo(results)
    gs = results_mod.load_goalscorers()

    sb = sb_mod.load_matches()
    sb["home_c"] = sb["home_team"].map(canonical)
    sb["away_c"] = sb["away_team"].map(canonical)
    group = sb[sb["stage"] == "Group Stage"].copy()

    # Join StatsBomb fixtures to martj42 rows tolerating (a) home/away order
    # flips between the sources and (b) +/-1 day date drift (evening kickoffs
    # cross midnight UTC). martj42 orientation becomes canonical: team A = its
    # home_team. StatsBomb contributes the stage filter and match_id only.
    m42 = results_elo[
        ["date", "home_team", "away_team", "home_score", "away_score", "neutral", "elo_home", "elo_away"]
    ]
    candidates = []
    for offset in (0, 1, -1):
        shifted = m42.assign(join_date=m42["date"] + pd.Timedelta(days=offset))
        for left_keys in (["date", "home_c", "away_c"], ["date", "away_c", "home_c"]):
            candidates.append(
                group.merge(
                    shifted,
                    left_on=left_keys,
                    right_on=["join_date", "home_team", "away_team"],
                    suffixes=("_sb", ""),
                )
            )
    fixtures = pd.concat(candidates, ignore_index=True).drop_duplicates(subset="match_id")
    n_lost = len(group) - len(fixtures)
    if n_lost > len(group) * 0.03:
        raise ValueError(f"{n_lost}/{len(group)} group fixtures failed to join martj42 — investigate")

    goals = _resolve_goal_team_convention(gs, results_elo)
    # keep only fully reconciled matches: credited home goals == home_score AND
    # credited away goals == away_score
    goals["credited_home"] = goals["credited_team"] == goals["home_team"]
    agg = goals.groupby(["date", "home_team", "away_team"]).agg(
        h_goals=("credited_home", "sum"),
        n_goals=("credited_home", "size"),
        home_score=("home_score", "first"),
        away_score=("away_score", "first"),
    )
    ok = agg[(agg["h_goals"] == agg["home_score"]) & (agg["n_goals"] - agg["h_goals"] == agg["away_score"])]
    goals_ok = goals.merge(ok.reset_index()[["date", "home_team", "away_team"]], on=["date", "home_team", "away_team"])

    half = (
        goals_ok.assign(second_half=goals_ok["minute"] > 45)
        .groupby(["date", "home_team", "away_team", "credited_team", "second_half"])
        .size()
        .rename("goals")
        .reset_index()
    )
    return BacktestData(results_elo, fixtures, half)


def run_backtest(half_life_days: float = 730.0) -> pd.DataFrame:
    data = load_backtest_data()
    rows = []
    for tournament, fx in data.fixtures.groupby("tournament"):
        as_of = fx["date"].min()
        model = fit_goal_model(data.results_elo, as_of, half_life_days)
        base = _climatology(data.results_elo, as_of)

        for m in fx.itertuples():
            a_home_ground = not m.neutral
            probs = markets.answer_all(model, m.elo_home, m.elo_away, a_home_ground)
            outcomes = _settle(m, data.half_goals)
            for arch in ARCHETYPES:
                if outcomes.get(arch) is None:
                    continue
                rows.append(
                    {
                        "tournament": tournament,
                        "date": m.date,
                        "fixture": f"{m.home_team} v {m.away_team}",
                        "archetype": arch,
                        "p_model": probs[arch],
                        "p_base": base[arch] if arch in base else _elo_base(arch, m, base),
                        "outcome": outcomes[arch],
                    }
                )
    return pd.DataFrame(rows)


def _climatology(results_elo: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    """Base rates from competitive internationals in the 8 years before as_of."""
    df = results_elo[
        (results_elo["date"] < as_of)
        & (results_elo["date"] >= as_of - pd.Timedelta(days=365 * 8))
        & (~results_elo["tournament"].str.contains("Friendly"))
    ]
    total = df["home_score"] + df["away_score"]
    return {
        "draw": float((df["home_score"] == df["away_score"]).mean()),
        "total_ge_2": float((total >= 2).mean()),
        "total_ge_3": float((total >= 3).mean()),
        "btts": float(((df["home_score"] > 0) & (df["away_score"] > 0)).mean()),
        "btts_and_3plus": float(((df["home_score"] > 0) & (df["away_score"] > 0) & (total >= 3)).mean()),
        # P(team scores in a half): Poisson thinning of the climatological
        # scoring rate: 1 - (1 - P(scores at all))^share  (share from markets.py)
        "a_scores_2h": _half_base(float((df["home_score"] > 0).mean()), markets.SECOND_HALF_SHARE),
        "b_scores_2h": _half_base(float((df["away_score"] > 0).mean()), markets.SECOND_HALF_SHARE),
        "a_scores_1h": _half_base(float((df["home_score"] > 0).mean()), 1 - markets.SECOND_HALF_SHARE),
    }


def _half_base(p_any: float, share: float) -> float:
    return float(1.0 - (1.0 - p_any) ** share)


def _elo_base(arch: str, m, base: dict[str, float]) -> float:
    """Baseline for win questions: Elo expected score with the climatological
    draw share carved out proportionally."""
    dr = m.elo_home - m.elo_away + (0.0 if m.neutral else 100.0)
    we = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
    p_draw = base["draw"]
    if arch == "win_a":
        return we * (1 - p_draw)
    if arch == "win_b":
        return (1 - we) * (1 - p_draw)
    raise KeyError(arch)


def _settle(m, half_goals: pd.DataFrame) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    total = m.home_score + m.away_score
    out["win_a"] = float(m.home_score > m.away_score)
    out["win_b"] = float(m.away_score > m.home_score)
    out["draw"] = float(m.home_score == m.away_score)
    out["total_ge_2"] = float(total >= 2)
    out["total_ge_3"] = float(total >= 3)
    out["btts"] = float(m.home_score > 0 and m.away_score > 0)
    out["btts_and_3plus"] = float(out["btts"] and total >= 3)

    hg = half_goals[
        (half_goals["date"] == m.date)
        & (half_goals["home_team"] == m.home_team)
        & (half_goals["away_team"] == m.away_team)
    ]
    if hg.empty and total > 0:
        # no reconciled scorer data for this match -> half questions unsettleable
        out["a_scores_2h"] = out["b_scores_2h"] = out["a_scores_1h"] = None
    else:
        def scored(team: str, second: bool) -> float:
            sel = hg[(hg["credited_team"] == team) & (hg["second_half"] == second)]
            return float(sel["goals"].sum() > 0)

        out["a_scores_2h"] = scored(m.home_team, True)
        out["b_scores_2h"] = scored(m.away_team, True)
        out["a_scores_1h"] = scored(m.home_team, False)
    return out


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    def brier(p, y):
        return float(np.mean((p - y) ** 2))

    rows = []
    for arch, g in bt.groupby("archetype"):
        rows.append(
            {
                "archetype": arch,
                "n": len(g),
                "brier_model": brier(g["p_model"], g["outcome"]),
                "brier_base": brier(g["p_base"], g["outcome"]),
                "edge_vs_base": brier(g["p_base"], g["outcome"]) - brier(g["p_model"], g["outcome"]),
                "base_rate": float(g["outcome"].mean()),
                "mean_p_model": float(g["p_model"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("edge_vs_base", ascending=False)
