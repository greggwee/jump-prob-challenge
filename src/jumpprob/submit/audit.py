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
                         min_gap: int = 5) -> bool:
    """Guard against a props-less run REVERTING a previously book-anchored player prop back to
    the bare model. The Diallo 50->24 lock: his SoT line vanished from the near-KO snapshot
    (line pulled, or listed as just 'Amad' which the resolver rejects), so that run priced him
    at the bare model (24) and OVERWROTE the anchored 50 already submitted, then it locked.

    True when: (a) it's a player prop, (b) THIS run has no book anchor (sub_market == sub_pure
    within 1pt), and (c) the EXISTING live submission is materially anchored (>= min_gap from
    the bare model). In that case keep the existing anchored value rather than regress it.
    Conservative by design — a genuinely-benched player is rare and the divergence audit + a
    fresh book still catch it; silently reverting a good anchor to a data-gapped model is worse."""
    if archetype not in _PLAYER_ARCHS or existing_pct is None:
        return False
    this_run_no_book = abs(int(round(sub_market)) - int(round(sub_pure))) <= 1
    existing_anchored = abs(int(round(existing_pct)) - int(round(sub_pure))) >= int(min_gap)
    return this_run_no_book and existing_anchored


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
