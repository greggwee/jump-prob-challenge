"""Phase L2 live op — generate the expert panel for the IMMINENT slate, out of band.

Budget-aware: finds matches with open markets kicking off within --within-hours,
refreshes the model numbers for just those (run_matchday_odds dry --matches), preps
GATE-ONLY archetypes (the only ones that can move a submission — no spend on shadow
markets), runs the panel, and writes data/processed/ai_panel.json. If nothing is
imminent it exits without spending a cent.

Schedule a few hours before kickoffs (e.g. twice daily). The submit daemon
(scheduler_tick -> run_matchday_odds) hot-reads the panel and blends iff
AI_EXPERT_ENABLE=1 and a per-archetype gate entry exists.

  python scripts/ai_panel_live.py --within-hours 12
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone

from jumpprob.submit.api_client import SportsPredictClient

EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
PY = sys.executable

# Force the expert OFF for the model-number refresh: the panel must ground on the
# PURE/anchored model number, never on an already-blended one (no feedback loop).
_ENV = {**os.environ, "AI_EXPERT_ENABLE": "0"}


def _run(cmd: list[str]) -> None:
    print(">", " ".join(cmd), flush=True)
    r = subprocess.run([PY, *cmd], env=_ENV)
    if r.returncode:
        sys.exit(r.returncode)


def imminent_matches(within_hours: float) -> list[str]:
    """Names of matches with open markets kicking off within the window."""
    client = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
    now = datetime.now(timezone.utc)
    out = []
    for m in client.matches(EVENT, LOBBY):
        if m.get("open_market_count", 0) <= 0:
            continue
        ko = datetime.fromisoformat(m["opening_time"].replace("Z", "+00:00"))
        hrs = (ko - now).total_seconds() / 3600.0
        if 0 <= hrs <= within_hours:
            out.append(m["name"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--within-hours", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    names = imminent_matches(args.within_hours)
    if not names:
        print(f"no open markets kicking off within {args.within_hours:.0f}h — nothing to do (no spend).")
        return
    print(f"imminent slate ({len(names)}): {names}")

    # refresh model numbers for just these matches (dry; no PATCH), then panel them.
    _run(["scripts/run_matchday_odds.py", "--matches", "||".join(names)])
    _run(["scripts/ai_panel_prep.py", "--source", "live", "--gate-only"])
    _run(["scripts/ai_panel_write.py",
          "--prep", "data/processed/ai_panel_prep.json",
          "--out", "data/processed/ai_panel.json", "--workers", str(args.workers)])
    print("live panel ready -> data/processed/ai_panel.json")


if __name__ == "__main__":
    main()
