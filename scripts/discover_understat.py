"""Phase / #2 feasibility probe: what per-player data does Understat expose, and
in what shape? Understat embeds JSON inside <script> tags as JSON.parse('...').
"""

import json
import re

import requests

HDR = {"User-Agent": "Mozilla/5.0"}


def _embedded(html: str, var: str):
    """Understat pages hold data as: var X = JSON.parse('<hex-escaped json>');"""
    m = re.search(var + r"\s*=\s*JSON\.parse\('(.+?)'\)", html, re.S)
    if not m:
        return None
    raw = m.group(1).encode().decode("unicode_escape")
    return json.loads(raw)


# 1. league players (season aggregates) — the bulk per-player rate source
r = requests.get("https://understat.com/league/EPL/2024", headers=HDR, timeout=20)
print(f"league page HTTP {r.status_code}")
players = _embedded(r.text, "playersData") or _embedded(r.text, "var playersData")
if players:
    print(f"  playersData: {len(players)} players; sample keys = {sorted(players[0].keys())}")
    p = players[0]
    print(f"  example: {p.get('player_name')} games={p.get('games')} min={p.get('time')} "
          f"shots={p.get('shots')} xG={p.get('xG')} goals={p.get('goals')} "
          f"assists={p.get('assists')} npg={p.get('npg')} key_passes={p.get('key_passes')}")
else:
    print("  could not parse playersData (structure may have changed)")

# 2. a player's per-shot history (for SoT-rate derivation) — optional richness
if players:
    pid = players[0]["id"]
    rp = requests.get(f"https://understat.com/player/{pid}", headers=HDR, timeout=20)
    shots = _embedded(rp.text, "shotsData")
    if shots:
        on_target = sum(1 for s in shots if s.get("result") in ("Goal", "SavedShot"))
        print(f"\n  player {players[0]['player_name']}: {len(shots)} career shots, "
              f"~{on_target} on target ({on_target/max(len(shots),1):.0%} SoT rate)")
        print(f"  shot keys = {sorted(shots[0].keys())}")
