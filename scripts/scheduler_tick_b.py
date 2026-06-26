"""Closing-line submission tick for gigabot2 (KEY_B) — the aggressive rank-chaser.

An ISOLATED mirror of scheduler_tick.py (Bot A): separate state file, separate log,
separate Windows task. It never touches Bot A's path. Fires submit_gigabot2.py once per
match, ~1.5-2.5 min before kickoff, to refresh the aggressive v2 sheet to the closing line.

  .venv\\Scripts\\pythonw.exe scripts\\scheduler_tick_b.py          # live
  .venv\\Scripts\\python.exe  scripts\\scheduler_tick_b.py --dry    # report only
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from jumpprob.submit.api_client import SportsPredictClient  # noqa: E402

EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
# STAGGERED a minute earlier than Bot A's [1.5, 2.5] so the two bots never run their
# heavy MC + odds-fetch at the same instant (avoids CPU contention and bursting the
# shared 60-req/min SportsPredict rate limit). gigabot2 submits ~T-3.5; Bot A ~T-2.
WIN_MIN, WIN_MAX = 3.0, 4.0
GLOBAL_THROTTLE_MIN = 1.0
DRY = "--dry" in sys.argv
STATE = ROOT / "data" / "processed" / "scheduler_state_b.json"   # isolated from Bot A
LOG = ROOT / "data" / "scheduler_b.log"
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


def _state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            return {}
    return {}


def main() -> None:
    now = datetime.now(timezone.utc)
    client = SportsPredictClient("SPORTSPREDICT_KEY_B", dry_run=True)   # read-only here
    try:
        matches = client.matches(EVENT, LOBBY)
    except Exception as e:
        log(f"matches fetch failed ({e}); skip")
        return

    st = _state()
    submitted = st.get("submitted", {})
    due = []
    for m in matches:
        if m.get("open_market_count", 0) <= 0:
            continue
        ko = datetime.fromisoformat(m["opening_time"].replace("Z", "+00:00"))
        mins = (ko - now).total_seconds() / 60.0
        key = f"{m['name']}|{ko.date()}"
        if WIN_MIN <= mins <= WIN_MAX and key not in submitted:
            due.append((m["name"], key, round(mins, 1)))

    if not due:
        log(f"no match due in [{WIN_MIN},{WIN_MAX}]m closing window; skip")
        return

    last = st.get("last_run")
    if last:
        try:
            if (now - datetime.fromisoformat(last)).total_seconds() / 60.0 < GLOBAL_THROTTLE_MIN:
                log(f"due {due} but throttled; retry next tick")
                return
        except Exception:
            pass

    names = "||".join(n for n, _, _ in due)
    if DRY:
        log(f"[DRY] would submit gigabot2 closing line for: {due}")
        return

    log(f"closing window {due} -> targeted gigabot2 submission")
    st["last_run"] = now.isoformat()
    STATE.write_text(json.dumps(st))
    try:
        r = subprocess.run(
            [str(PYEXE), "scripts/submit_gigabot2.py", "--live", "--matches", names],
            cwd=str(ROOT), capture_output=True, text=True, timeout=600,
            creationflags=CREATE_NO_WINDOW,
        )
        tail = (r.stdout.strip().splitlines() or ["<no output>"])[-1]
        log(f"gigabot2 submission exit={r.returncode}: {tail}")
        if r.returncode == 0:
            for _, key, _ in due:
                submitted[key] = now.isoformat()
            st["submitted"] = submitted
            STATE.write_text(json.dumps(st))
        elif r.stderr:
            log(f"  stderr: {r.stderr.strip().splitlines()[-1][:200]}")
    except subprocess.TimeoutExpired:
        log("gigabot2 submission timed out (>10m) — will retry next tick")


if __name__ == "__main__":
    main()
