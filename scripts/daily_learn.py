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

    def _run(label: str, script: str, want_line: str = "offsides"):
        try:
            p = subprocess.run([str(PYEXE), script],
                               cwd=str(ROOT), capture_output=True, text=True, timeout=300,
                               creationflags=CREATE_NO_WINDOW)
            lines = p.stdout.strip().splitlines() or [""]
            tail = next((ln for ln in lines if want_line in ln), lines[-1])
            log(f"{label}: {tail.strip()}")
            if p.returncode != 0 and p.stderr:
                log(f"  {label} stderr: {p.stderr.strip().splitlines()[-1][:200]}")
        except Exception as e:
            log(f"{label} FAILED: {e}")

    # 2. refit calibration from accumulating settled data — the offsides
    #    definition-gap correction sharpens as live offsides settle.
    _run("calibration", "scripts/fit_live_calibration.py", want_line="offsides")
    # 3. update edge gate from settlements -> extremizing-gate learning
    _run("edge gate", "scripts/update_edge_gate.py", want_line="edge_gate")
    # 4. Phase 1: rebuild settled+crowd dataset and refit asymmetric confidence
    #    calibration (floors confident-low bets on archetypes where that loses big).
    #    Uses the harvested crowd consensus (data/processed/crowd_consensus_raw.csv;
    #    refresh manually until the harvest is automated — new markets without crowd
    #    data are simply not yet in the fit). The fit self-gates on a no-regression
    #    LOO-match CV, so a refit can only tighten the floor set, never regress it.
    _run("settled dataset", "scripts/build_settled_dataset.py", want_line="wrote settled_with_crowd")
    _run("confidence", "scripts/fit_confidence.py", want_line="wrote confidence_gate")
    #    P1: refresh the per-archetype crowd prior (the submit-time catastrophic-deviation
    #    clamp's proxy) as newly harvested crowd rows accumulate.
    _run("crowd prior", "scripts/fit_crowd_prior.py", want_line="crowd_prior")
    # 5. forward shadow-scorer for the open quant-review hypotheses (de-extremize /
    #    A>B over-prediction). Append-only ledger, FIXED params -> validates on fresh
    #    settled data before either touches the live path. Logged, never auto-applied.
    _run("shadow hypotheses", "scripts/shadow_decalib.py", want_line="de-extremize")

    log("daily learn complete")


if __name__ == "__main__":
    main()
