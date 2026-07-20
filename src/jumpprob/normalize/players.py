"""Canonical player-name resolution for joining sources (Understat <-> markets).

Player names differ across sources by accent (Gyokeres/Gyökeres), order
(Son Heung-min / Heung-Min Son), punctuation (N'Golo), and case. `fold_name`
normalises accents/punctuation/case; token-set matching handles word order.
`resolve` returns (matched_name, method) where method is exact / tokenset / none,
so callers can flag fuzzy joins for the mis-resolution gate (review L3).
"""

from __future__ import annotations

import re
import unicodedata


def ascii_lower(name: str) -> str:
    """Accent-stripped lowercase, preserving punctuation and word breaks.

    Deliberately weaker than :func:`fold_name`: the player-rate tables are keyed
    on this form, and it is used for substring token matching where keeping
    hyphens and apostrophes intact is what makes the match specific.
    """
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()


def fold_name(name: str) -> str:
    """Accent-, punctuation-, case-insensitive normal form. Hyphens -> spaces so
    'Heung-min' and 'Heung min' tokenise identically."""
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    n = n.lower()
    n = n.replace("'", "").replace(".", "")   # join intra-word: N'Golo->ngolo, C.->c
    n = n.replace("-", " ")                     # hyphen -> token break
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def tokens(name: str) -> frozenset:
    """Order-independent token set (so 'Son Heung-min' == 'Heung-Min Son')."""
    return frozenset(t for t in fold_name(name).split() if t)


def build_index(names) -> tuple[dict, dict]:
    """Index a source's canonical names: (folded->name, tokenset->name).
    First spelling wins on collision (deterministic)."""
    exact: dict[str, str] = {}
    tok: dict[frozenset, str] = {}
    for nm in names:
        exact.setdefault(fold_name(nm), nm)
        tok.setdefault(tokens(nm), nm)
    return exact, tok


def resolve(query: str, exact: dict, tok: dict) -> tuple[str | None, str]:
    """Resolve a query name against an index. Tiers, most-conservative first:
    exact fold -> order-independent token-set -> strict subset (handles middle
    names, e.g. 'Jonathan David' vs 'Jonathan Christian David') accepted only when
    a SINGLE candidate matches (ambiguous -> reject, avoiding wrong joins).
    Returns (matched_name | None, method)."""
    f = fold_name(query)
    if not f:
        return None, "none"
    if f in exact:
        return exact[f], "exact"
    t = tokens(query)
    if t in tok:
        return tok[t], "tokenset"
    if len(t) >= 2:  # one name's tokens strictly contained in the other's
        cands = [name for ts, name in tok.items() if t < ts or ts < t]
        if len(cands) == 1:
            return cands[0], "subset"
    return None, "none"
