"""Phase L2 Tier-3 — near-kickoff WEB-SEARCH pass for the imminent match.

~T-60 (when lineups publish) this refreshes ONLY the gated player props
(player_ga / player_sot_ge) of a match about to kick off, using a web-search
panelist that pulls the confirmed lineup, recent form, and injury/pundit news
(past the models' Jan-2026 cutoff). The result is MERGED into ai_panel.json
(other matches' daily entries untouched) and logged to a forward-test ledger.

Budget discipline (each web call ~$0.2-0.4):
  * gated archetypes only (the ones that can move a submission),
  * one match at a time, inside a narrow near-kickoff window,
  * run ONCE per match (state file),
  * 1 model by default (--models 2 for a wider ensemble),
  * a hard --max-questions cap per run.

Poll this every ~10 min near kickoffs (Task Scheduler). Most runs no-op cheaply.
Force a specific match for testing with --match "Name".
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from jumpprob.agents import panel as P
from jumpprob.agents.expert import client, web_forecast
from jumpprob.submit.api_client import SportsPredictClient

EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
PY = sys.executable
LOG = Path("data/ai_panel_web.log")
STATE = Path("data/processed/ai_panel_web_state.json")
LEDGER = Path("data/processed/ai_panel_web_ledger.csv")
PREP = Path("data/processed/ai_panel_prep.json")
# expert OFF for the model-number refresh -> grounding anchors on the pure model
_ENV = {**os.environ, "AI_EXPERT_ENABLE": "0"}


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line)
    except Exception:
        pass


def _run(cmd: list[str]) -> None:
    log("> " + " ".join(cmd))
    r = subprocess.run([PY, *cmd], env=_ENV, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or ["<no output>"])[-1]
    log(f"  exit={r.returncode}: {tail}")
    if r.returncode:
        if r.stderr:
            log(f"  stderr: {r.stderr.strip().splitlines()[-1][:200]}")
        sys.exit(r.returncode)


def _state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            return {}
    return {}


def _targets(win_lo: float, win_hi: float, forced: str | None) -> list[str]:
    client_sp = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
    now = datetime.now(timezone.utc)
    done = _state()
    out = []
    for m in client_sp.matches(EVENT, LOBBY):
        if m.get("open_market_count", 0) <= 0:
            continue
        name = m["name"]
        ko = datetime.fromisoformat(m["opening_time"].replace("Z", "+00:00"))
        mins = (ko - now).total_seconds() / 60.0
        key = f"{name}|{ko.date()}"
        if forced:
            if name == forced:
                out.append(name)
        elif win_lo <= mins <= win_hi and key not in done:
            out.append(name)
    return out


def _mark_done(names: list[str]) -> None:
    client_sp = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
    st = _state()
    ko_by = {m["name"]: m["opening_time"] for m in client_sp.matches(EVENT, LOBBY)}
    for n in names:
        ko = ko_by.get(n)
        if ko:
            d = datetime.fromisoformat(ko.replace("Z", "+00:00")).date()
            st[f"{n}|{d}"] = datetime.now(timezone.utc).isoformat()
    STATE.write_text(json.dumps(st, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win-lo", type=float, default=35.0)
    ap.add_argument("--win-hi", type=float, default=80.0)
    ap.add_argument("--match", default=None, help="force a specific match (testing)")
    ap.add_argument("--models", type=int, default=1, help="web ensemble width (1=Opus, 2=+Sonnet)")
    ap.add_argument("--max-uses", type=int, default=3)
    ap.add_argument("--max-questions", type=int, default=12, help="budget cap per run")
    args = ap.parse_args()

    if client() is None:
        log("no ANTHROPIC_API_KEY — skip")
        return
    try:
        names = _targets(args.win_lo, args.win_hi, args.match)
    except Exception as e:
        log(f"matches fetch failed ({e}); skip")
        return
    if not names:
        log(f"no match in web window [{args.win_lo:.0f},{args.win_hi:.0f}]m — skip (no spend).")
        return
    log(f"web pass for: {names}")

    # refresh model numbers for just these matches, prep GATE-ONLY player props
    _run(["scripts/run_matchday_odds.py", "--matches", "||".join(names)])
    _run(["scripts/ai_panel_prep.py", "--source", "live", "--gate-only"])
    items = json.loads(PREP.read_text())
    items = [it for it in items if it["match"] in names][: args.max_questions]
    if not items:
        log("no gated player props in these matches — skip")
        if not args.match:
            _mark_done(names)
        return

    models = ["claude-opus-4-8", "claude-sonnet-4-6"][: max(1, args.models)]
    entries, rows, t0 = {}, [], time.time()
    for i, it in enumerate(items, 1):
        res = web_forecast(it["question"], it["model_pct"], it.get("grounding"),
                           models=models, max_uses=args.max_uses)
        ens = res["p_ensemble"]
        entries[str(it["market_id"])] = {
            "p_model": it["model_pct"], "p_ensemble": ens, "models": res["models"],
            "n": res["n"], "role": it["role"], "action": it["action"],
            "archetype": it["archetype"], "source": "web", "generated_at": time.time(),
        }
        rows.append({"ts": datetime.now(timezone.utc).isoformat(), "market_id": it["market_id"],
                     "match": it["match"], "archetype": it["archetype"],
                     "question": it["question"], "model_pct": it["model_pct"],
                     "web_prob": ens, "n_models": res["n"]})
        log(f"  {i}/{len(items)} {it['archetype']:14s} model={it['model_pct']:.2f} web={ens}")

    P.merge_panel(entries, P.PANEL_PATH, meta={"web": names, "n": len(entries)})
    import csv
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    if not args.match:                  # forced/test runs don't consume the once-per-match slot
        _mark_done(names)
    log(f"merged {len(entries)} web entries -> {P.PANEL_PATH}; ledger += {len(rows)} "
        f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
