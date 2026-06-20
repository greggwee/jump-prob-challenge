"""Phase L0 — run the multi-model panel over the prep file and CACHE it.

Reads data/processed/ai_panel_prep.json, polls the curated panel per item (out of
band, threaded over models), robust-combines (trim-median in log-odds), and writes
data/processed/ai_panel.json for jumpprob.agents.panel to hot-read.

Shadow by default: producing this cache changes NOTHING live until a role weight
(AI_TACTICAL_WEIGHT / AI_LINEUP_WEIGHT) is set. Use --limit for a smoke test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from jumpprob.agents import panel as P
from jumpprob.agents.expert import client, panel_forecast


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="data/processed/ai_panel_prep.json")
    ap.add_argument("--out", default=str(P.PANEL_PATH))
    ap.add_argument("--limit", type=int, default=0, help="cap items (smoke test)")
    args = ap.parse_args()

    if client() is None:
        raise SystemExit("no ANTHROPIC_API_KEY in env/.env — cannot run the panel")

    items = json.loads(Path(args.prep).read_text())
    if args.limit:
        items = items[: args.limit]

    entries: dict[str, dict] = {}
    t0 = time.time()
    for i, it in enumerate(items, 1):
        res = panel_forecast(it["question"], it["model_pct"], it["role"], it.get("grounding"))
        entries[str(it["market_id"])] = {
            "p_model": it["model_pct"], "p_ensemble": res["p_ensemble"],
            "models": res["models"], "n": res["n"], "role": it["role"],
            "action": it["action"], "archetype": it["archetype"],
            "generated_at": time.time(),
        }
        if i % 10 == 0 or i == len(items):
            ens = res["p_ensemble"]
            print(f"  {i}/{len(items)}  {it['archetype']:16s} model={it['model_pct']:.2f} "
                  f"ens={ens:.2f}" if ens is not None else
                  f"  {i}/{len(items)}  {it['archetype']:16s} model={it['model_pct']:.2f} ens=None")

    P.write_panel(entries, Path(args.out), meta={"prep": args.prep, "n": len(entries)})
    print(f"wrote {len(entries)} entries -> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
