"""Polymarket (crowd) prices for WC 2026 matches — the FIELD proxy (Phase B).

A sharp book line estimates the TRUE probability; a prediction-market price
estimates a CROWD's belief (with crowd biases). The SportsPredict field is a
crowd, so Polymarket's per-match prices proxy the field on the markets it covers
better than the sharp line does. Public Gamma API, no auth.

fetch_match_crowd() -> {(canonical_home, canonical_away): MatchCrowd}, where the
3-way (home/draw/away) crowd probabilities come from the event's binary markets,
normalized. volume and bid-ask spread gauge liquidity (= the crowd's precision).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from jumpprob.normalize.teams import canonical

GAMMA = "https://gamma-api.polymarket.com"
HDR = {"User-Agent": "Mozilla/5.0 research"}


@dataclass
class MatchCrowd:
    home: str
    away: str
    p_home: float
    p_draw: float
    p_away: float
    volume: float
    spread: float | None
    raw_sum: float  # pre-normalization sum of Yes prices (sanity: ~1.0)


def _events(max_events: int = 600) -> list[dict]:
    out, offset = [], 0
    while offset < max_events:
        r = requests.get(f"{GAMMA}/events",
                         params={"closed": "false", "limit": 100, "offset": offset,
                                 "order": "volume", "ascending": "false"},
                         headers=HDR, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        offset += 100
        if len(batch) < 100:
            break
    return out


def _prices(market: dict) -> float | None:
    p = market.get("outcomePrices")
    if isinstance(p, str):
        p = json.loads(p)
    return float(p[0]) if p else None  # Yes price


def fetch_match_crowd(max_events: int = 600) -> dict[tuple[str, str], MatchCrowd]:
    out: dict[tuple[str, str], MatchCrowd] = {}
    for e in _events(max_events):
        title = e.get("title", "")
        low = title.lower()
        sep = " vs. " if " vs. " in low else " vs " if " vs " in low else None
        if sep is None:
            continue
        parts = title.split(sep[0] if False else sep)
        if len(parts) != 2:
            continue
        a_raw, b_raw = parts[0].strip(), parts[1].strip()
        ca, cb = canonical(a_raw), canonical(b_raw)
        markets = e.get("markets") or []
        ph = pd = pa = None
        vol, spreads = 0.0, []
        for m in markets:
            q = (m.get("question") or "").lower()
            git = (m.get("groupItemTitle") or "").lower()
            yes = _prices(m)
            if yes is None:
                continue
            vol += float(m.get("volumeNum") or m.get("volume") or 0)
            ba, bb = m.get("bestAsk"), m.get("bestBid")
            if ba and bb:
                spreads.append(float(ba) - float(bb))
            label = git or q
            if "draw" in label or "tie" in label:
                pd = yes
            elif a_raw.lower() in label:
                ph = yes
            elif b_raw.lower() in label:
                pa = yes
        if ph and pd and pa:
            s = ph + pd + pa
            out[(ca, cb)] = MatchCrowd(ca, cb, ph / s, pd / s, pa / s, vol,
                                       sum(spreads) / len(spreads) if spreads else None, s)
    return out


def crowd_for(crowd: dict[tuple[str, str], MatchCrowd], home: str, away: str) -> MatchCrowd | None:
    """Look up a fixture in either orientation."""
    return crowd.get((home, away)) or _flip(crowd.get((away, home)))


def _flip(mc: MatchCrowd | None) -> MatchCrowd | None:
    if mc is None:
        return None
    return MatchCrowd(mc.away, mc.home, mc.p_away, mc.p_draw, mc.p_home, mc.volume, mc.spread, mc.raw_sum)
