"""Six-tournament point-in-time backtest: v2 MC engine vs v1 analytic vs base.

Point-in-time discipline: per evaluated tournament, the goal model is fit
as-of its first match on results only; the count model AND the dynamics are
fit on event data from tournaments that finished before it (wc2018 is
training-only). Posterior-predictive covariance from the same as-of fit.

Usage:
  python scripts/backtest_v2.py                  # full v2
  python scripts/backtest_v2.py --no-coupling    # ablations
  python scripts/backtest_v2.py --no-pp
  python scripts/backtest_v2.py --no-frailty
  python scripts/backtest_v2.py --draws 50000
Outputs: data/processed/backtest_v2[_<tag>].csv + summary table.
"""

import sys

import numpy as np
import pandas as pd

from jumpprob.backtest.harness import ARCHETYPES, _settle, load_backtest_data
from jumpprob.models.count_model import fit_count_model
from jumpprob.models.dynamics import fit_dynamics
from jumpprob.models.goal_model import fit_goal_model
from jumpprob.models.uncertainty import beta_covariance
from jumpprob.normalize.teams import canonical
from jumpprob.simulate import markets as mk
from jumpprob.simulate.mc_engine import MCEngine, answer

ORDER = ["wc2018", "euro2020", "wc2022", "afcon2023", "copa2024", "euro2024"]
COUPLED = "--no-coupling" not in sys.argv
if "--goal-only" in sys.argv:
    COUPLED = {"goal"}  # production config: goal coupling only (see ablations)
USE_PP = "--no-pp" not in sys.argv
USE_FRAILTY = "--no-frailty" not in sys.argv
DRAWS = int(sys.argv[sys.argv.index("--draws") + 1]) if "--draws" in sys.argv else 50_000
TAG = ("_go" if isinstance(COUPLED, set) else
       "" if (COUPLED and USE_PP and USE_FRAILTY) else
       "_nc" * (not COUPLED) + "_npp" * (not USE_PP) + "_nf" * (not USE_FRAILTY))

# v1<->v2 archetype mapping: (v2 archetype, side, threshold, half)
V1_TO_V2 = {
    "win_a": ("win", "a", None, None), "win_b": ("win", "b", None, None),
    "draw": ("draw", None, None, None),
    "total_ge_2": ("total_ge", None, 2, None), "total_ge_3": ("total_ge", None, 3, None),
    "btts": ("btts", None, None, None), "btts_and_3plus": ("btts_and_total_ge", None, 3, None),
    "a_scores_2h": ("team_scores_half", "a", None, 2), "b_scores_2h": ("team_scores_half", "b", None, 2),
    "a_scores_1h": ("team_scores_half", "a", None, 1),
}
COUNT_ARCHS = {
    "corners_ht_a_gt_b": ("corners_gt", "a", None, 1),
    "cards_2h_ge_2": ("cards_ge", None, 2, 2),
    "offsides_a_ge_2": ("offsides_ge", "a", 2, None),
    "sot_a_gt_b_2h": ("sot_gt", "a", None, 2),
}
TIMING_ARCHS = {
    "ht_tied": ("ht_tied", None, None, None),
    "h2_more_goals_than_h1": ("h2_more_goals_than_h1", None, None, None),
}

data = load_backtest_data()
stream = pd.read_parquet("data/processed/event_stream.parquet")
th = pd.read_parquet("data/processed/team_half_counts.parquet")
th["team_c"] = th["team"].map(canonical)
th["cards"] = th.yellows + th.reds
th["offsides"] = th.pass_offsides + th.offside_events

rows = []
for i, t in enumerate(ORDER):
    if i == 0:
        continue
    fx = data.fixtures[data.fixtures["tournament"] == t]
    if fx.empty:
        continue
    as_of = fx["date"].min()
    gm = fit_goal_model(data.results_elo, as_of, 730.0)
    cov = beta_covariance(gm, data.results_elo, as_of) if USE_PP else None

    # point-in-time training sets from prior tournaments
    train_t = ORDER[:i]
    sub_stream = stream[stream["tournament"].isin(train_t)]
    # goal offsets from fixtures' point-in-time elo (group matches; knockouts fall back to 0)
    offs = {}
    fxall = data.fixtures.set_index("match_id")
    for mid in sub_stream["match_id"].unique():
        if mid in fxall.index:
            r = fxall.loc[mid]
            la, lb = gm.intensities(r["elo_home"], r["elo_away"], not r["neutral"])
            offs[(mid, r["home_team_sb"])] = float(np.log(la))
            offs[(mid, r["away_team_sb"])] = float(np.log(lb))
    dyn = fit_dynamics(sub_stream, goal_offsets=offs)
    # scale calibration: quick pilot on this tournament's fixture lambdas
    for _ in range(2):
        eng0 = MCEngine(gm, None, dyn, n_draws=8_000, coupled=COUPLED, frailty=False, seed=3)
        rats = []
        for m in fx.head(12).itertuples():
            la, lb = gm.intensities(m.elo_home, m.elo_away, not m.neutral)
            res = eng0.simulate(m.elo_home, m.elo_away, not m.neutral)
            rats.append((res.score_a.mean() + res.score_b.mean()) / (la + lb))
        dyn.scale["goal"] = round(dyn.scale.get(
            "goal", float(np.exp(np.asarray(dyn.alpha["goal"])[np.minimum(np.arange(90) // 15, 5)]).sum())
        ) * float(np.mean(rats)), 4)
    # count model trained on prior tournaments (same rows as backtest_counts)
    th_train = th[th["tournament"].isin(train_t)][
        ["match_id", "team_c", "period", "corners", "cards", "offsides", "shots_on_target", "fouls"]
    ].merge(
        data.fixtures[["match_id", "elo_home", "elo_away", "neutral", "home_team", "referee"]],
        on="match_id")
    crows = []
    for r in th_train.itertuples():
        is_a = r.team_c == r.home_team
        la, lb = gm.intensities(r.elo_home, r.elo_away, not r.neutral)
        crows.append({"period": r.period, "lam_team": la if is_a else lb,
                      "lam_opp": lb if is_a else la, "referee": r.referee,
                      "corners": r.corners, "cards": r.cards, "offsides": r.offsides,
                      "shots_on_target": r.shots_on_target, "fouls": r.fouls})
    cm = fit_count_model(pd.DataFrame(crows))
    # calibrate COUNT-family scales too (goal-only calibration leaves families
    # with asymmetric state betas ~10% inflated — caught by the SoT archetype
    # regressing in the first verification run)
    from jumpprob.simulate.mc_engine import COUNT_FAMILY
    eng0 = MCEngine(gm, cm, dyn, n_draws=8_000, coupled=COUPLED, frailty=False, seed=5)
    fam_ratios = {f: [] for f in dyn.alpha if f != "goal"}
    for m in fx.head(12).itertuples():
        la, lb = gm.intensities(m.elo_home, m.elo_away, not m.neutral)
        res0 = eng0.simulate(m.elo_home, m.elo_away, not m.neutral)
        for f in res0.counts:
            cf = COUNT_FAMILY[f]
            mu = sum(cm.mu(cf, h, la, lb) + cm.mu(cf, h, lb, la) for h in (1, 2))
            fam_ratios[f].append(res0.counts[f].sum(axis=(1, 2)).mean() / mu)
    for f, rs in fam_ratios.items():
        if rs:
            prev = dyn.scale.get(f, float(np.exp(np.asarray(dyn.alpha[f])[
                np.minimum(np.arange(90) // 15, 5)]).sum() * np.exp(dyn.beta[f][2])))
            dyn.scale[f] = round(prev * float(np.mean(rs)), 4)
    eng = MCEngine(gm, cm, dyn, n_draws=DRAWS, coupled=COUPLED,
                   frailty=USE_FRAILTY, beta_cov=cov, seed=42)

    th_t = th[th["tournament"] == t]
    for m in fx.itertuples():
        res = eng.simulate(m.elo_home, m.elo_away, not m.neutral)
        v1_probs = mk.answer_all(gm, m.elo_home, m.elo_away, not m.neutral)
        outcomes = _settle(m, data.half_goals)
        # count + timing settlements from event tables
        g = th_t[th_t["match_id"] == m.match_id]
        a1 = g[(g.team_c == m.home_team) & (g.period == 1)]
        a2 = g[(g.team_c == m.home_team) & (g.period == 2)]
        b1 = g[(g.team_c == m.away_team) & (g.period == 1)]
        b2 = g[(g.team_c == m.away_team) & (g.period == 2)]
        if len(a1) and len(a2) and len(b1) and len(b2):
            a1, a2, b1, b2 = a1.iloc[0], a2.iloc[0], b1.iloc[0], b2.iloc[0]
            outcomes["corners_ht_a_gt_b"] = float(a1.corners > b1.corners)
            outcomes["cards_2h_ge_2"] = float(a2.cards + b2.cards >= 2)
            outcomes["offsides_a_ge_2"] = float(a1.offsides + a2.offsides >= 2)
            outcomes["sot_a_gt_b_2h"] = float(a2.shots_on_target > b2.shots_on_target)
            outcomes["ht_tied"] = float(a1.goals == b1.goals)
            outcomes["h2_more_goals_than_h1"] = float((a2.goals + b2.goals) > (a1.goals + b1.goals))

        for arch, (v2a, side, thr, hf) in {**V1_TO_V2, **COUNT_ARCHS, **TIMING_ARCHS}.items():
            y = outcomes.get(arch)
            if y is None:
                continue
            p2 = answer(res, v2a, side, threshold=thr, half=hf)
            p1 = v1_probs.get(arch)
            rows.append({"tournament": t, "match_id": m.match_id, "archetype": arch,
                         "p_v2": p2, "p_v1": p1, "outcome": y})

bt = pd.DataFrame(rows)
out = f"data/processed/backtest_v2{TAG}.csv"
bt.to_csv(out, index=False)

bt["sq2"] = (bt.p_v2 - bt.outcome) ** 2
have_v1 = bt.dropna(subset=["p_v1"]).copy()
have_v1["sq1"] = (have_v1.p_v1 - have_v1.outcome) ** 2
s = bt.groupby("archetype").agg(n=("sq2", "size"), brier_v2=("sq2", "mean"),
                                base_rate=("outcome", "mean"))
s = s.join(have_v1.groupby("archetype")["sq1"].mean().rename("brier_v1"))
s["v2_minus_v1"] = s.brier_v1 - s.brier_v2
print(f"config: coupled={COUPLED} pp={USE_PP} frailty={USE_FRAILTY} draws={DRAWS}")
print(s.round(4).to_string())
print(f"\noverall v2 Brier: {bt.sq2.mean():.4f}")
print(f"overall v1 Brier (same settlements): {have_v1.sq1.mean():.4f} | "
      f"v2 on those: {have_v1.sq2.mean():.4f} | edge {have_v1.sq1.mean() - have_v1.sq2.mean():+.4f}")
print(f"saved {out}")
