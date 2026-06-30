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
