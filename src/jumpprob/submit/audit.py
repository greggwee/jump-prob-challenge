"""Submission-time audits — cheap sanity checks that surface SILENT failures every run
instead of post-settlement.

Two failure modes these catch, both amplified by the w_market=0.85 anchor:
  * model data-gaps — a player/market the structural model prices badly (e.g. Haaland's
    scoring rate not loaded -> model 7% while the book prices 52%);
  * book-market mismatch — the de-vigged book line does not actually match the platform
    question (scope: regulation vs ET; threshold; semantics), so anchoring to it is wrong.

A large gap between the pure MODEL and the de-vigged BOOK is the shared signature of both,
so we flag it for review. Reporting only — never changes a submission.
"""

from __future__ import annotations

import math

import pandas as pd


_PLAYER_ARCHS = frozenset({"player_scores", "player_ga", "player_sot_ge"})


def is_anchor_regression(archetype: str, existing_pct, sub_pure, sub_market,
                         min_gap: int = 5, was_anchored: bool | None = None) -> bool:
    """Guard against a props-less run REVERTING a previously book-anchored player prop back to
    the bare model. The Diallo 50->24 lock: his SoT line vanished from the near-KO snapshot
    (line pulled, or listed as just 'Amad' which the resolver rejects), so that run priced him
    at the bare model (24) and OVERWROTE the anchored 50 already submitted, then it locked.

    True when: (a) it's a player prop, (b) THIS run has no book anchor (sub_market == sub_pure
    within 1pt), (c) the EXISTING live submission is materially different (>= min_gap) from the
    bare model, AND (d) the ledger CONFIRMS the market was book-anchored on a previous run
    (was_anchored). (d) is load-bearing: 'existing differs from the model' alone cannot tell a
    protected book anchor (Diallo 50 vs model 24) from a STALE BAD submission the current model
    wants to correct — the guard's first live version kept a mis-parsed 'DR Congo scores' at 8%
    while the fixed pipeline wanted 42 (crowd 37, out=YES, -0.45 edge, the single biggest R32
    loss). No ledger evidence of a prior anchor -> never block a correction."""
    if archetype not in _PLAYER_ARCHS or existing_pct is None:
        return False
    if not was_anchored:                      # None/False: no proof it was ever book-anchored
        return False
    this_run_no_book = abs(int(round(sub_market)) - int(round(sub_pure))) <= 1
    existing_anchored = abs(int(round(existing_pct)) - int(round(sub_pure))) >= int(min_gap)
    return this_run_no_book and existing_anchored


def previously_anchored_markets(ledger_path="data/processed/clv_ledger.csv") -> set:
    """Market_ids whose most recent ledger row was book-anchored (sub_market != sub_pure).
    Feeds is_anchor_regression's was_anchored evidence; missing/empty ledger -> empty set."""
    try:
        led = pd.read_csv(ledger_path)
        last = led.sort_values("ts").groupby("market_id").last()
        return set(last.index[(last.sub_market - last.sub_pure).abs() >= 2])
    except Exception:
        return set()


def book_vs_model_divergence(df: pd.DataFrame, min_pts: int = 20) -> pd.DataFrame:
    """Flag rows where the de-vigged BOOK line disagrees with the pure MODEL by >= min_pts
    integer probability points. Expects columns `sub_pure` (model %) and `sub_market` (book %);
    returns the flagged rows sorted by |gap| descending, with added `book_gap` (book - model)
    and `logit_gap` (the same gap in log-odds, which is scale-aware: a 20pt gap at 5->25% is a
    bigger disagreement than at 45->65%). Empty frame if nothing crosses the threshold."""
    if df.empty:
        return df.assign(book_gap=pd.Series(dtype=int), logit_gap=pd.Series(dtype=float))
    d = df.copy()
    d["book_gap"] = d["sub_market"].astype(int) - d["sub_pure"].astype(int)

    def _lg(row) -> float:
        a = min(0.999, max(0.001, row["sub_pure"] / 100.0))
        b = min(0.999, max(0.001, row["sub_market"] / 100.0))
        return abs(math.log(b / (1 - b)) - math.log(a / (1 - a)))

    d["logit_gap"] = d.apply(_lg, axis=1).round(2)
    flagged = d[d["book_gap"].abs() >= int(min_pts)]
    return flagged.reindex(flagged["book_gap"].abs().sort_values(ascending=False).index)
