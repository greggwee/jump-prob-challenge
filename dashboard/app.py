"""Probability Cup dashboard.

Run:  .venv\\Scripts\\python -m streamlit run dashboard/app.py

Tabs: Bot status (live API) | Match explorer | v1 vs v2 shadow | Backtests.
The refresh button snapshots /predictions and /results to
data/processed/api_snapshot.json (rate-limit friendly: manual, cached).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SNAPSHOT = PROCESSED / "api_snapshot.json"
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"

st.set_page_config(page_title="Probability Cup", layout="wide")
st.title("Jump Trading Probability Cup — bot dashboard")


def load_csv(name: str) -> pd.DataFrame | None:
    p = PROCESSED / name
    return pd.read_csv(p) if p.exists() else None


def refresh_snapshot():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from jumpprob.submit.api_client import SportsPredictClient
    import os
    os.chdir(ROOT)
    c = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
    preds = c.my_predictions(LOBBY)
    results = c.results(LOBBY)
    SNAPSHOT.write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "predictions": preds, "results": results}))


with st.sidebar:
    st.header("Data")
    if st.button("Refresh from API"):
        with st.spinner("fetching..."):
            refresh_snapshot()
        st.success("snapshot updated")
    if SNAPSHOT.exists():
        snap = json.loads(SNAPSHOT.read_text())
        st.caption(f"API snapshot: {snap['fetched_at'][:19]}Z")
    else:
        snap = None
        st.caption("no API snapshot yet — click refresh")

tab_entry, tab_results, tab_status, tab_match, tab_shadow, tab_back = st.tabs(
    ["⭐ Bot B manual entry", "📊 Settled results", "Bot status",
     "Match explorer", "v1 vs v2 shadow", "Backtests"])

# ---------------- Settled results ----------------
with tab_results:
    if snap is None:
        st.info("Click 'Refresh from API' to load settled results.")
    else:
        res = pd.DataFrame(snap.get("results") or snap["predictions"])
        res = res.dropna(subset=["brier_score"]).copy()
        if res.empty:
            st.warning("No settled questions yet — results appear as matches complete.")
        else:
            # recover binary outcome from our own Brier: y=1 iff (p-1)^2 == brier
            res["p"] = res["probability_submitted" if "probability_submitted" in res else "probability"] / 100.0
            amb = np.isclose(res["p"], 0.5)
            res["outcome"] = np.where(
                np.isclose((res["p"] - 1) ** 2, res["brier_score"], atol=1e-6), 1, 0).astype(float)
            res.loc[amb, "outcome"] = np.nan  # p=50 -> unrecoverable

            # attribute to match + archetype via the matchday sheet (by market_id)
            md = load_csv("matchday_predictions.csv")
            if md is not None:
                res = res.merge(md[["market_id", "match", "archetype"]], on="market_id", how="left")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Settled questions", len(res))
            c2.metric("Our mean Brier", f"{res.brier_score.mean():.4f}")
            c3.metric("vs coin-flip (0.25)", f"{0.25 - res.brier_score.mean():+.4f}")
            won = res["outcome"].mean()
            c4.metric("YES-outcome rate", f"{won:.0%}" if not np.isnan(won) else "—")

            # field-relative: API hides field Brier; let the user paste total RBP
            st.caption("The API does not expose the field average. Paste your total "
                       "Relative Brier Points from the profile page for a field comparison.")
            rbp = st.number_input("Total RBP (from SportsPredict profile)", value=0.0, step=1.0)
            if rbp != 0.0:
                # field_avg_brier = our_total_brier + RBP/100  (sum over settled, mult=1 group stage)
                field_avg = res.brier_score.sum() / len(res) + rbp / 100.0 / len(res)
                st.metric("Implied field mean Brier", f"{field_avg:.4f}",
                          delta=f"{field_avg - res.brier_score.mean():+.4f} (you − field, lower is better)")

            if "created_date" in res:
                res["created_date"] = pd.to_datetime(res["created_date"])
                res = res.sort_values("created_date")
            res["running_brier"] = res.brier_score.expanding().mean()
            st.subheader("Running mean Brier")
            st.line_chart(res.reset_index(drop=True)["running_brier"])

            if "archetype" in res and res["archetype"].notna().any():
                st.subheader("Brier by archetype (where we are strong / weak)")
                by_arch = res.groupby("archetype").agg(
                    n=("brier_score", "size"), brier=("brier_score", "mean"),
                    yes_rate=("outcome", "mean")).sort_values("brier")
                st.dataframe(by_arch.round(4), use_container_width=True)
            if "match" in res and res["match"].notna().any():
                st.subheader("Brier by match")
                by_match = res.groupby("match").agg(
                    n=("brier_score", "size"), brier=("brier_score", "mean")).sort_values("brier")
                st.dataframe(by_match.round(4), use_container_width=True)

            # Phase 7 validation: pure-model vs odds-anchored Brier on settled markets
            anc = load_csv("odds_anchor_run.csv")
            if anc is not None:
                j = res.merge(anc[["market_id", "sub_pure", "sub_a"]], on="market_id", how="inner")
                j = j.dropna(subset=["outcome"])
                if len(j):
                    j["brier_pure"] = (j["sub_pure"] / 100 - j["outcome"]) ** 2
                    j["brier_anchored"] = (j["sub_a"] / 100 - j["outcome"]) ** 2
                    st.subheader(f"Odds-anchor A/B on {len(j)} settled markets")
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Pure-model Brier", f"{j.brier_pure.mean():.4f}")
                    cc2.metric("Odds-anchored Brier", f"{j.brier_anchored.mean():.4f}")
                    cc3.metric("Anchor edge", f"{j.brier_pure.mean() - j.brier_anchored.mean():+.4f}")
                else:
                    st.caption("Odds-anchor A/B: accumulating — today's anchored markets "
                               "have not settled yet.")

            with st.expander("All settled questions"):
                cols = [c for c in ("match", "question", "archetype", "probability_submitted",
                                    "brier_score", "outcome") if c in res]
                st.dataframe(res[cols].sort_values("brier_score"),
                             use_container_width=True, hide_index=True)

# ---------------- Bot B manual entry ----------------
with tab_entry:
    st.subheader("Upcoming matches — Bot B (v2) values to enter")
    st.caption("Recompute with:  `.venv\\Scripts\\python scripts\\prep_bot_b.py`")
    bb = load_csv("bot_b_manual.csv")
    if bb is None:
        st.info("Run scripts/prep_bot_b.py to generate the entry sheet.")
    else:
        bb["kickoff"] = pd.to_datetime(bb["kickoff"], utc=True)
        now = pd.Timestamp.now(tz="UTC")
        upcoming = bb[bb["kickoff"] > now].copy()
        if upcoming.empty:
            st.warning("No open matches ahead in the sheet — rerun prep_bot_b.py.")
        only_diff = st.checkbox("Show only questions where Bot B differs from Bot A", value=False)
        st.caption(f"{upcoming['match'].nunique()} upcoming matches · "
                   f"sheet built for {bb['match'].nunique()} matches total")
        for match, g in sorted(upcoming.groupby("match"), key=lambda kv: kv[1]["kickoff"].min()):
            ko = g["kickoff"].min()
            mins = int((ko - now).total_seconds() // 60)
            when = f"{mins // 60}h {mins % 60}m" if mins >= 60 else f"{mins}m"
            with st.expander(f"**{match}**  ·  closes in {when}  ·  {ko.strftime('%a %H:%M')}Z", expanded=mins < 360):
                show = g[g["diff"].abs() >= 1] if only_diff else g
                view = show[["question", "sub_a", "sub_b", "diff"]].rename(
                    columns={"question": "Question", "sub_a": "Bot A (live)",
                             "sub_b": "Bot B — ENTER THIS", "diff": "Δ"})
                st.dataframe(
                    view.style.apply(
                        lambda r: ["background-color: #ffe08a; color: #000"
                                   if abs(r["Δ"]) >= 3 else "" for _ in r], axis=1),
                    use_container_width=True, hide_index=True)
                st.caption("Highlighted rows = Bot B differs from the live Bot A value by ≥3 points.")

# ---------------- Bot status ----------------
with tab_status:
    if snap is None:
        st.info("Click 'Refresh from API' to load live bot status.")
    else:
        preds = pd.DataFrame(snap["predictions"])
        settled = preds.dropna(subset=["brier_score"]) if "brier_score" in preds else pd.DataFrame()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predictions", len(preds))
        c2.metric("Settled", len(settled))
        c3.metric("Open", int((preds.get("market_status") == "open").sum()))
        if len(settled):
            c4.metric("Cumulative Brier (mean)", f"{settled.brier_score.mean():.4f}")
            settled = settled.copy()
            settled["created_date"] = pd.to_datetime(settled["created_date"])
            settled = settled.sort_values("created_date")
            settled["running_brier"] = settled.brier_score.expanding().mean()
            st.subheader("Running mean Brier (settled questions)")
            st.line_chart(settled.set_index(settled.index)["running_brier"])
            st.subheader("Settled questions")
            st.dataframe(settled[["question", "probability", "brier_score"]]
                         .sort_values("brier_score"), use_container_width=True)
        else:
            c4.metric("Cumulative Brier", "—")
            st.caption("No settled questions yet.")
        with st.expander("All open predictions"):
            st.dataframe(preds[["question", "probability", "market_status"]],
                         use_container_width=True)

# ---------------- Match explorer ----------------
with tab_match:
    v1 = load_csv("matchday_predictions.csv")
    v2 = load_csv("shadow_v2.csv")
    if v1 is None:
        st.info("Run scripts/run_matchday.py first.")
    else:
        match = st.selectbox("Match", sorted(v1["match"].unique()))
        sub = v1[v1["match"] == match][["market_id", "question", "archetype", "p_bot_b"]] \
            .rename(columns={"p_bot_b": "v1 (submitted)"})
        if v2 is not None:
            sub = sub.merge(v2[["market_id", "p_v2"]].rename(columns={"p_v2": "v2 (shadow)"}),
                            on="market_id", how="left")
        st.dataframe(sub.drop(columns="market_id").set_index("question").round(3),
                     use_container_width=True)
        chart = sub.set_index("question")[[c for c in ("v1 (submitted)", "v2 (shadow)") if c in sub]]
        st.bar_chart(chart)

# ---------------- v1 vs v2 ----------------
with tab_shadow:
    v2 = load_csv("shadow_v2.csv")
    if v2 is None:
        st.info("Run scripts/shadow_v2.py first.")
    else:
        sim = v2[v2["v2_simulated"]].copy()
        sim["diff"] = sim.p_v2 - sim.p_v1
        c1, c2 = st.columns(2)
        c1.metric("Markets simulated by v2", len(sim))
        c2.metric("Mean |v2 − v1|", f"{sim['diff'].abs().mean():.4f}")
        st.subheader("v1 vs v2 (each point = one market)")
        st.scatter_chart(sim.rename(columns={"p_v1": "v1", "p_v2": "v2"}),
                         x="v1", y="v2", color="archetype", size=20)
        st.subheader("Largest divergences")
        st.dataframe(sim.reindex(sim["diff"].abs().sort_values(ascending=False).index)
                     [["match", "question", "archetype", "p_v1", "p_v2", "diff"]].head(20).round(3),
                     use_container_width=True)
        st.subheader("Mean divergence by archetype")
        st.bar_chart(sim.groupby("archetype")["diff"].mean())

# ---------------- Backtests ----------------
with tab_back:
    go = load_csv("backtest_v2_go.csv")
    if go is not None:
        go["sq2"] = (go.p_v2 - go.outcome) ** 2
        have = go.dropna(subset=["p_v1"]).copy()
        have["sq1"] = (have.p_v1 - have.outcome) ** 2
        s = go.groupby("archetype").agg(n=("sq2", "size"), v2=("sq2", "mean"))
        s = s.join(have.groupby("archetype")["sq1"].mean().rename("v1"))
        s["v2_minus_v1"] = (s.v1 - s.v2).round(4)
        st.subheader("Six-tournament backtest (production v2 config vs v1)")
        st.dataframe(s.round(4), use_container_width=True)
        st.bar_chart(s[["v1", "v2"]])
    bf = load_csv("blend_fit.csv")
    if bf is not None:
        bf["sq"] = (bf.p - bf.outcome) ** 2
        st.subheader("Anchor blend weight (Brier vs w, 0 = pure model, 1 = pure anchor)")
        st.line_chart(bf.groupby("w")["sq"].mean())
    st.caption("Full analysis: PLAN.md §9, V2-FINDINGS.md, docs/METHODS.md")
