"""Phase L2 live op — generate the expert panel for the CURRENT slate, out of band.

Chains the prep (from the latest odds_anchor_run.csv model numbers) and the panel
run, writing data/processed/ai_panel.json for the live daemon to hot-read. Schedule
this ONCE per slate, after the model numbers first exist and well before lock (the
LLM calls take a few minutes).

The daemon blends only if AI_EXPERT_ENABLE=1 AND a per-archetype gate entry exists
(data/processed/expert_gate.json); otherwise the fresh panel is pure shadow (logged
in odds_anchor_run.csv as p_expert/expert_action, submissions unchanged).
"""

from __future__ import annotations

import subprocess
import sys

PY = sys.executable


def _run(cmd: list[str]) -> None:
    print(">", " ".join(cmd), flush=True)
    r = subprocess.run([PY, *cmd])
    if r.returncode:
        sys.exit(r.returncode)


def main() -> None:
    _run(["scripts/ai_panel_prep.py", "--source", "live"])
    _run(["scripts/ai_panel_write.py",
          "--prep", "data/processed/ai_panel_prep.json",
          "--out", "data/processed/ai_panel.json", "--workers", "4"])
    print("live panel ready -> data/processed/ai_panel.json")


if __name__ == "__main__":
    main()
