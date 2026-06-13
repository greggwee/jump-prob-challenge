"""Near-kickoff submission tick — run every few minutes by Windows Task Scheduler.

Strategy: most ticks do nothing (a cheap SportsPredict /matches check costs no
odds credits). When an open match is inside the firing window before kickoff, it
runs a TARGETED odds-anchored submission for just that match's markets via Bot A,
capturing the latest (near-closing) line.

TIMEZONE: everything is UTC. The API gives kickoff (`closing_time`) in UTC, and
datetime.now(timezone.utc) is UTC regardless of the desktop's local time-zone
setting — so this works correctly no matter where you are travelling, as long as
the machine's absolute clock is right (NTP handles that; the zone offset does not
matter). No local time is ever used.

Window: fire when WIN_MIN <= minutes_to_kickoff <= WIN_MAX. WIN_MIN must exceed
the worst-case run time (~model load 2m + targeted submit ~30s) so the PATCH
lands before the market locks. A global throttle avoids back-to-back heavy runs.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)  # so .env and data/ relative paths resolve under Task Scheduler
sys.path.insert(0, str(ROOT / "src"))

from jumpprob.submit.api_client import SportsPredictClient  # noqa: E402

EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
WIN_MIN, WIN_MAX = 4.0, 16.0      # minutes-to-kickoff firing window
THROTTLE_MIN = 3.5                 # don't launch runs closer together than this
DRY = "--dry" in sys.argv
STATE = ROOT / "data" / "processed" / "scheduler_state.json"
LOG = ROOT / "data" / "scheduler.log"
PYEXE = ROOT / ".venv" / "Scripts" / "python.exe"
CREATE_NO_WINDOW = 0x08000000


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:  # under pythonw.exe (Task Scheduler) stdout is None — file log is enough
        print(line)
    except Exception:
        pass


def main() -> None:
    now = datetime.now(timezone.utc)
    client = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
    try:
        matches = client.matches(EVENT, LOBBY)
    except Exception as e:  # transient API/network — try again next tick
        log(f"matches fetch failed ({e}); skip")
        return

    near = []
    for m in matches:
        if m.get("open_market_count", 0) <= 0:
            continue
        ko = datetime.fromisoformat(m["closing_time"].replace("Z", "+00:00"))
        mins = (ko - now).total_seconds() / 60.0
        if WIN_MIN <= mins <= WIN_MAX:
            near.append((m["name"], round(mins, 1)))

    if not near:
        # quiet success; only log occasionally to avoid noise would be nicer, but
        # a one-line skip is cheap and useful for auditing while travelling.
        log(f"no match in [{WIN_MIN:.0f},{WIN_MAX:.0f}]m window; skip")
        return

    # global throttle
    last = None
    if STATE.exists():
        try:
            last = datetime.fromisoformat(json.loads(STATE.read_text())["last_run"])
        except Exception:
            last = None
    if last and (now - last).total_seconds() / 60.0 < THROTTLE_MIN:
        log(f"near {near} but throttled (<{THROTTLE_MIN}m since last run); skip")
        return

    names = "||".join(n for n, _ in near)
    if DRY:
        log(f"[DRY] would submit Bot A for: {near}")
        return

    log(f"near kickoff {near} -> targeted Bot A submission")
    STATE.write_text(json.dumps({"last_run": now.isoformat()}))
    try:
        r = subprocess.run(
            [str(PYEXE), "scripts/run_matchday_odds.py", "--live", "--matches", names],
            cwd=str(ROOT), capture_output=True, text=True, timeout=600,
            creationflags=CREATE_NO_WINDOW,
        )
        tail = (r.stdout.strip().splitlines() or ["<no output>"])[-1]
        log(f"submission exit={r.returncode}: {tail}")
        if r.returncode != 0 and r.stderr:
            log(f"  stderr: {r.stderr.strip().splitlines()[-1][:200]}")
    except subprocess.TimeoutExpired:
        log("submission timed out (>10m) — will retry next tick")


if __name__ == "__main__":
    main()
