"""Closing-line submission tick — run every minute by Windows Task Scheduler.

Strategy: every market already has a baseline submission (the initial slate + any
manual run), so the scheduler's job is just to refresh to the CLOSING line as late
as is safe. It fires ONCE per match, inside a narrow window a minute or two before
kickoff, then never again for that match. That:
  * lands the final submission ~1-2 min before kickoff (captures the closing line),
  * spends ONE odds fetch per match instead of 5-6 across a wide window — the Odds
    API free tier is small (4 credits/fetch) and was being burned through fast.

KICKOFF = opening_time (verified UTC). Markets LOCK at kickoff. WIN_MIN must exceed
the worst-case targeted run time (~25-40s = ~0.5 min) plus margin so the PATCH lands
before the lock — hence a 1.5-minute floor, not literally T-1 (a run started at T-1
could finish after the market has locked).

Once-per-match state is keyed by name+kickoff-date and set ONLY after a successful
run, so a transient failure is retried on the next minute's tick while the match is
still inside the window (free redundancy without extra fetches).

  .venv\\Scripts\\pythonw.exe scripts\\scheduler_tick.py          # live
  .venv\\Scripts\\python.exe  scripts\\scheduler_tick.py --dry    # report only
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
WIN_MIN, WIN_MAX = 1.5, 2.5    # minutes-to-kickoff closing window (poll every 1 min)
GLOBAL_THROTTLE_MIN = 1.0      # don't launch two heavy runs back-to-back
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
    client = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
    try:
        matches = client.matches(EVENT, LOBBY)
    except Exception as e:
        log(f"matches fetch failed ({e}); skip")
        return

    st = _state()
    submitted = st.get("submitted", {})            # {name|date: iso}
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

    # global throttle (avoid overlapping heavy runs if kickoffs cluster)
    last = st.get("last_run")
    if last:
        try:
            if (now - datetime.fromisoformat(last)).total_seconds() / 60.0 < GLOBAL_THROTTLE_MIN:
                log(f"due {due} but throttled (<{GLOBAL_THROTTLE_MIN}m since last run); retry next tick")
                return
        except Exception:
            pass

    names = "||".join(n for n, _, _ in due)
    if DRY:
        log(f"[DRY] would submit closing line for: {due}")
        return

    log(f"closing window {due} -> targeted Bot A submission")
    st["last_run"] = now.isoformat()
    STATE.write_text(json.dumps(st))
    try:
        r = subprocess.run(
            [str(PYEXE), "scripts/run_matchday_odds.py", "--live", "--matches", names],
            cwd=str(ROOT), capture_output=True, text=True, timeout=600,
            creationflags=CREATE_NO_WINDOW,
        )
        tail = (r.stdout.strip().splitlines() or ["<no output>"])[-1]
        log(f"submission exit={r.returncode}: {tail}")
        if r.returncode == 0:                      # mark done ONLY on success (else retried next tick)
            for _, key, _ in due:
                submitted[key] = now.isoformat()
            st["submitted"] = submitted
            STATE.write_text(json.dumps(st))
        elif r.stderr:
            log(f"  stderr: {r.stderr.strip().splitlines()[-1][:200]}")
    except subprocess.TimeoutExpired:
        log("submission timed out (>10m) — will retry next tick")


if __name__ == "__main__":
    main()
