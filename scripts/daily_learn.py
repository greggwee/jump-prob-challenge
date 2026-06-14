"""Daily learning refresh — run once a day by Windows Task Scheduler.

Between-matchday learning, the disciplined (not per-match) way:
  1. Refresh the international results feed -> Elo learns from newly-played
     matches (incl. World Cup results). The next per-kickoff submit run picks up
     the updated ratings automatically (Pipeline reloads results each run).
  2. Re-run the edge gate from the day's settlements -> the extremizing gate
     learns where we are (or are not) beating the field.

Deliberately does NOT retrain the structural model betas mid-tournament
(small-sample overfit risk) and cannot update count/dynamics models (StatsBomb
does not release 2026 event data live). All times logged in UTC.
"""

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from jumpprob.ingest import results as results_mod  # noqa: E402

LOG = ROOT / "data" / "learning.log"
PYEXE = ROOT / ".venv" / "Scripts" / "python.exe"
CREATE_NO_WINDOW = 0x08000000


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line)
    except Exception:
        pass


def main() -> None:
    # 1. refresh international results -> Elo learning
    try:
        before = len(results_mod.load_results())
    except Exception:
        before = 0
    try:
        results_mod.fetch(force=True)
        r = results_mod.load_results()
        log(f"results refreshed: {len(r)} matches (+{len(r) - before} new) through {r['date'].max().date()}")
    except Exception as e:
        log(f"results refresh FAILED: {e}")

    # 2. update edge gate from settlements -> extremizing-gate learning
    try:
        p = subprocess.run([str(PYEXE), "scripts/update_edge_gate.py"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=300,
                           creationflags=CREATE_NO_WINDOW)
        tail = " | ".join((p.stdout.strip().splitlines() or [""])[-2:])
        log(f"edge gate: {tail}")
        if p.returncode != 0 and p.stderr:
            log(f"  edge gate stderr: {p.stderr.strip().splitlines()[-1][:200]}")
    except Exception as e:
        log(f"edge gate FAILED: {e}")

    log("daily learn complete")


if __name__ == "__main__":
    main()
