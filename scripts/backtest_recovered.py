"""Backtest the 2026-06-28 drop-audit recoveries on settled WC26 matches.

The recovered archetypes (first_goal_match, win_margin, clean_sheet, draw,
score_both_halves, half_total_ge, h2_more_goals_than_h1, and the goal hydration-
break windows) were *dropped* at submission time, so the platform never recorded
an outcome for them. We reconstruct the actual outcomes from martj42 (final score
+ goal minutes, which run through the live WC26 dates) and score the new handlers
against them.

What this validates: do the recovered handlers carry *match-specific skill* —
i.e. beat climatology (the base rate)? That is the prerequisite for beating the
noisy field we're scored against. It is a CONSERVATIVE floor: it uses the PURE
model grid (no odds anchor), while the live system anchors goals to the book
(strictly sharper), and it compares to the base rate, not the (unavailable) crowd.

NOT backtestable here (stated, not hidden):
  - player-prop market anchor — needs historical closing prop odds we never stored;
  - card/corner/offside hydration windows + red_card — martj42 has only goals.

  python scripts/backtest_recovered.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import glob

from jumpprob.ingest import odds_api
from jumpprob.normalize.teams import canonical
from jumpprob.pipeline import Pipeline
from jumpprob.submit.parser import ParsedQuestion
from jumpprob.submit.runner import CODE_MAP, Runner

CODE_MAP = {**CODE_MAP, "COD": "DR Congo", "CPV": "Cape Verde", "DRC": "DR Congo"}


def load_closing_anchors() -> dict:
    """{frozenset(home,away): MatchAnchor} from each match's CLOSING line — the
    latest on-disk snapshot that still listed it (after kickoff it leaves the feed).
    This reconstructs the anchor the live system would have submitted with."""
    out = {}
    for snap in sorted(glob.glob(str(odds_api.RAW_DIR / "odds_*.json"))):  # chronological
        try:
            import json
            events = json.loads(open(snap, encoding="utf-8").read())
            for (h, a), anc in odds_api.build_anchors(events).items():
                out[frozenset([h, a])] = anc        # latest snapshot wins = closing line
        except Exception:
            continue
    return out


def code2canon(tok: str) -> str:
    tok = tok.strip()
    return canonical(CODE_MAP.get(tok, tok))


def parse_minute(raw) -> tuple[int, bool]:
    """(base_minute, is_first_half). '45+2'->(45,True), '90+3'->(90,False)."""
    s = str(raw).strip()
    base = int(s.split("+")[0]) if "+" in s else int(float(s))
    return base, base <= 45


def load_martj42():
    res = pd.read_csv("data/raw/martj42/results.csv")
    res = res[(res["date"] >= "2026-06-01")].dropna(subset=["home_score", "away_score"]).copy()
    res["hk"], res["ak"] = res["home_team"].map(canonical), res["away_team"].map(canonical)
    res["pair"] = res.apply(lambda x: frozenset([x["hk"], x["ak"]]), axis=1)
    gs = pd.read_csv("data/raw/martj42/goalscorers.csv")
    gs = gs[gs["date"] >= "2026-06-01"].copy()
    gs["tk"] = gs["team"].map(canonical)
    return res, gs


def match_goals(gs: pd.DataFrame, hk: str, ak: str, date: str) -> list[dict]:
    """Credited goals for one match: (minute, is_h1, credited_team). Own goals
    are credited to the opponent of the listed team."""
    g = gs[(gs["date"] == date) &
           (gs["tk"].isin([hk, ak]))]
    out = []
    for r in g.itertuples():
        base, is_h1 = parse_minute(r.minute)
        team = r.tk
        if bool(getattr(r, "own_goal", False)):
            team = ak if team == hk else hk
        out.append({"min": base, "h1": is_h1, "team": team})
    return out


def derive_outcomes(hk, ak, hs, asc, goals) -> dict[tuple[str, str | None], float]:
    """Actual 0/1 outcome per (archetype, subject-canonical) from final score+goals."""
    score = {hk: int(hs), ak: int(asc)}
    h1 = [g for g in goals if g["h1"]]
    h2 = [g for g in goals if not g["h1"]]
    first = min(goals, key=lambda g: g["min"])["team"] if goals else None
    o: dict[tuple[str, str | None], float] = {
        ("draw", None): float(hs == asc),
        ("half_total_ge", None): float(len(h1) >= 2),     # "first half produce 2+"
        ("h2_more_goals_than_h1", None): float(len(h2) > len(h1)),
        ("goal_pre_hb1", None): float(any(g["min"] <= 22 for g in goals)),     # break at 22'
        ("goal_post_hb2", None): float(any(g["min"] > 67 for g in goals)),     # 2nd break at 45+22=67'
    }
    for t, opp in ((hk, ak), (ak, hk)):
        o[("clean_sheet", t)] = float(score[opp] == 0)
        o[("win_margin", t)] = float(score[t] - score[opp] >= 2)
        o[("first_goal_match", t)] = float(first == t)
        o[("score_both_halves", t)] = float(any(g["team"] == t for g in h1) and
                                            any(g["team"] == t for g in h2))
    return o


def model_question(arch, subject, threshold=None, half=None):
    return ParsedQuestion(arch, subject=subject, threshold=threshold, half=half)


def main():
    settled = pd.read_parquet("data/processed/settled_with_crowd.parquet")
    res, gs = load_martj42()
    by_pair = {r.pair: r for r in res.itertuples()}
    anchors = load_closing_anchors()

    runner = Runner(Pipeline(), pd.read_parquet("data/processed/player_match.parquet"), mc=None)

    def predict(arch, ctx, subject, thr=None, half=None):
        return runner.answer(model_question(arch, subject, thr, half), ctx)

    rows = []
    joined = n_anc = 0
    for mname in settled["match"].unique():
        parts = mname.split(" vs ")
        if len(parts) != 2:
            continue
        ta, tb = code2canon(parts[0]), code2canon(parts[1])
        rec = by_pair.get(frozenset([ta, tb]))
        if rec is None:
            continue
        hk, ak, hs, asc = rec.hk, rec.ak, rec.home_score, rec.away_score  # oriented to martj42 home/away
        goals = match_goals(gs, hk, ak, rec.date)
        # goalscorers/score consistency guard (own-goal credit, missing scorers)
        if sum(1 for g in goals if g["team"] == hk) != int(hs) or \
           sum(1 for g in goals if g["team"] == ak) != int(asc):
            continue
        anc = anchors.get(frozenset([ta, tb]))
        try:
            ctx_pure = runner.context(ta, tb)
            ctx_anc = runner.context(ta, tb, anchor=anc) if anc is not None else None
        except KeyError:
            continue
        joined += 1
        n_anc += ctx_anc is not None and ctx_anc.anchored
        outs = derive_outcomes(hk, ak, hs, asc, goals)
        for (arch, subj), y in outs.items():
            thr = 2 if arch in ("win_margin", "half_total_ge") else None
            half = 1 if arch == "half_total_ge" else None
            p_pure = predict(arch, ctx_pure, subj, thr, half)
            p_anc = predict(arch, ctx_anc, subj, thr, half) if (ctx_anc is not None and ctx_anc.anchored) else None
            if p_pure is None:
                continue
            rows.append({"match": mname, "archetype": arch, "subject": subj, "y": y,
                         "p_pure": float(p_pure),
                         "p_anc": float(p_anc) if p_anc is not None else np.nan})

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"backtested {joined} settled WC26 matches ({n_anc} with a closing-line anchor) | "
          f"{len(df)} recovered-archetype questions\n")

    def report(col, label, mask=None):
        d = df if mask is None else df[mask]
        d = d.dropna(subset=[col])

        def agg(g):
            base = g["y"].mean()
            mb = float(((g[col] - g["y"]) ** 2).mean())
            cb = float(((base - g["y"]) ** 2).mean())
            return pd.Series({"n": len(g), "base_rate": base, "mean_p": g[col].mean(),
                              "brier": mb, "clim": cb, "skill": cb - mb})
        tab = d.groupby("archetype").apply(agg).sort_values("skill", ascending=False)
        print(f"=== {label}: model vs climatology (base rate) on real WC26 outcomes ===")
        print(tab.round(4).to_string())
        d = d.merge(d.groupby("archetype")["y"].mean().rename("b"), on="archetype")
        mb, cb = float(((d[col] - d["y"]) ** 2).mean()), float(((d["b"] - d["y"]) ** 2).mean())
        print(f"POOLED  model {mb:.4f}  vs climatology {cb:.4f}  skill {cb - mb:+.4f}  (n={len(d)})\n")

    report("p_pure", "PURE MODEL (conservative floor — no odds anchor)")
    report("p_anc", "ANCHORED to closing line (what the live system submits)", mask=df["p_anc"].notna())
    print("skill>0 => the handler carries match-specific information (the prerequisite for beating the field).")
    print("Compared to the base rate, not the (unavailable) crowd; player-prop anchor + card/corner/")
    print("offside hydration windows aren't here (no stored prop odds / no minute data for those families).")
    df.to_csv("data/processed/backtest_recovered.csv", index=False)


if __name__ == "__main__":
    main()
