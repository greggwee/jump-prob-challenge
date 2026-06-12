"""Estimate game-state dynamics from the StatsBomb event cache (reproducible).

1. Build/load the minute event stream (314 matches).
2. Fit alpha (minute profile) + beta (score-state multipliers) per family —
   feedback-safe full likelihood for goals (offsets = goal-model lambdas),
   conditional multinomial for everything else.
3. Calibrate simulation normalizers (scale) by pilot simulation so realized
   means match the models' lambdas/mus, and residual gamma frailty k per count
   family so simulated var/mean matches the measured team-half var/mean.
4. Write data/processed/minute_dynamics.json and print every measured value.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.ingest.event_stream import build_stream
from jumpprob.models.dynamics import Dynamics, fit_dynamics
from jumpprob.simulate.mc_engine import COUNT_FAMILY, MCEngine

STREAM_PATH = Path("data/processed/event_stream.parquet")
OUT = Path("data/processed/minute_dynamics.json")

stream = pd.read_parquet(STREAM_PATH) if STREAM_PATH.exists() else build_stream()
print(f"stream: {len(stream)} events, {stream.match_id.nunique()} matches")
print(stream.family.value_counts().to_string())

# goal offsets: log lambda per (match, team) from the current v1 goal model
from jumpprob.pipeline import Pipeline
pipe = Pipeline()
from jumpprob.normalize.teams import canonical
offsets = {}
fixtures = stream.drop_duplicates("match_id")[["match_id", "home_team", "away_team"]]
for r in fixtures.itertuples():
    h, a = canonical(r.home_team), canonical(r.away_team)
    rh = pipe.ratings.get(h, 1500.0)
    ra = pipe.ratings.get(a, 1500.0)
    la, lb = pipe.goal_model.intensities(rh, ra)
    offsets[(r.match_id, r.home_team)] = float(np.log(la))
    offsets[(r.match_id, r.away_team)] = float(np.log(lb))

dyn = fit_dynamics(stream, goal_offsets=offsets)
print("\nalpha (minute-profile, 15-min buckets, bucket0=0):")
for f, a in dyn.alpha.items():
    print(f"  {f}: {a}")
print("beta (state multipliers e^beta for [trail2+, trail1, level, lead1, lead2+]):")
for f, b in dyn.beta.items():
    print(f"  {f}: {[round(float(np.exp(x)), 3) for x in b]}")

# implied half splits from alpha (verification vs measured A2/A3)
from jumpprob.models.dynamics import BUCKET
for f in dyn.alpha:
    w = np.exp(np.asarray(dyn.alpha[f])[BUCKET])
    print(f"  {f} 2H share (alpha-implied): {w[45:].sum() / w.sum():.3f}")

# --- pilot-sim calibration: scale (means) then frailty (variance) ----------
print("\ncalibrating scales by pilot simulation...")
ref = pipe.results_elo.tail(400)[["elo_home", "elo_away"]].to_numpy()
ref = ref[:: max(1, len(ref) // 25)]  # ~25 reference fixtures

for iteration in range(3):
    eng = MCEngine(pipe.goal_model, pipe.count_model, dyn, n_draws=20_000, seed=7, frailty=False)
    ratios = {f: [] for f in dyn.alpha}
    for eh, ea in ref:
        la, lb = pipe.goal_model.intensities(eh, ea)
        res = eng.simulate(eh, ea)
        ratios["goal"].append((res.score_a.mean() + res.score_b.mean()) / (la + lb))
        for f in res.counts:
            cf = COUNT_FAMILY[f]
            mu = sum(pipe.count_model.mu(cf, h, la, lb) + pipe.count_model.mu(cf, h, lb, la) for h in (1, 2))
            ratios[f].append(res.counts[f].sum(axis=(1, 2)).mean() / mu)
    for f, rs in ratios.items():
        r = float(np.mean(rs))
        prev = dyn.scale.get(f, float(np.exp(np.asarray(dyn.alpha[f])[BUCKET]).sum()
                                      * np.exp(dyn.beta[f][2])))
        dyn.scale[f] = round(prev * r, 4)
    print(f"  iter {iteration}: ratios {{'goal': {np.mean(ratios['goal']):.4f}}} (target 1.0)")

# frailty: residual overdispersion after coupling
th = pd.read_parquet("data/processed/team_half_counts.parquet")
th["cards"] = th.yellows + th.reds
th["offsides"] = th.pass_offsides + th.offside_events
emp_vm = {"corner": th.corners, "card": th.cards, "foul": th.fouls,
          "sot": th.shots_on_target, "offside": th.offsides}
emp_vm = {f: (s.var() / s.mean(), s.mean()) for f, s in emp_vm.items()}

eng = MCEngine(pipe.goal_model, pipe.count_model, dyn, n_draws=20_000, seed=11, frailty=False)
sim_vals = {f: [] for f in emp_vm}
for eh, ea in ref:
    res = eng.simulate(eh, ea)
    for f in res.counts:
        sim_vals[f].append(res.counts[f].reshape(eng.n, 4))
for f, (v_emp, mu_emp) in emp_vm.items():
    arr = np.concatenate(sim_vals[f], axis=0).ravel()
    v_sim = arr.var() / arr.mean()
    if v_emp > v_sim + 0.02:
        k = float(mu_emp / (v_emp - v_sim))
    else:
        k = float("inf")
    dyn.frailty_k[f] = round(k, 3) if np.isfinite(k) else 1e9
    print(f"  {f}: emp var/mean {v_emp:.2f}, coupled-sim {v_sim:.2f} -> frailty k {dyn.frailty_k[f]}")

dyn.to_json(OUT)
print(f"\nwritten {OUT}")
