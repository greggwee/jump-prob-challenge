"""SportsPredict Model API client (docs: sportspredict.com/probabilitycup/api).

Two bots = two API keys = two separate leaderboard entries (verified in docs).
Keys come from env: SPORTSPREDICT_KEY_A (anchor bot), SPORTSPREDICT_KEY_B
(alpha bot). Every client method respects dry_run: log, don't POST.

Rate limit: 60 req/min/IP shared across both bots — the client sleeps to stay
under 50/min. Batch submissions (<=50) keep round-trips minimal.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from jumpprob.submit.env import load_dotenv

BASE = "https://api.sportspredict.com/api/v1"
LOG_DIR = Path("data/submissions")


class SportsPredictClient:
    def __init__(self, key_env: str, dry_run: bool = True):
        load_dotenv()
        self.key = os.environ.get(key_env, "")
        self.key_env = key_env
        self.dry_run = dry_run
        self._last_request = 0.0
        if not self.key and not dry_run:
            raise RuntimeError(f"set {key_env} for live submission")

    # -- plumbing ----------------------------------------------------------
    def _throttle(self):
        wait = 60.0 / 50.0 - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _get(self, path: str, **params):
        self._throttle()
        resp = requests.get(f"{BASE}{path}", params=params,
                            headers={"Authorization": f"Bearer {self.key}"}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict):
        if self.dry_run:
            self._log("DRY-POST", path, body)
            return {"dry_run": True, "path": path, "body": body}
        self._throttle()
        resp = requests.post(f"{BASE}{path}", json=body,
                             headers={"Authorization": f"Bearer {self.key}"}, timeout=30)
        resp.raise_for_status()
        out = resp.json()
        self._log("POST", path, {"body": body, "response": out})
        return out

    def _patch(self, path: str, body: dict):
        if self.dry_run:
            self._log("DRY-PATCH", path, body)
            return {"dry_run": True}
        self._throttle()
        resp = requests.patch(f"{BASE}{path}", json=body,
                              headers={"Authorization": f"Bearer {self.key}"}, timeout=30)
        resp.raise_for_status()
        out = resp.json()
        self._log("PATCH", path, {"body": body, "response": out})
        return out

    def _log(self, verb: str, path: str, payload: dict):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat()
        with (LOG_DIR / f"{self.key_env}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": stamp, "verb": verb, "path": path, "payload": payload}) + "\n")

    # -- API surface -------------------------------------------------------
    def events(self, limit: int = 20):
        return self._get("/events", limit=limit)

    def lobbies(self, event_id: str):
        return self._get("/lobbies", event_id=event_id)

    def join_lobby(self, lobby_id: str):
        return self._post(f"/lobbies/{lobby_id}/join", {})

    def matches(self, event_id: str, lobby_id: str):
        return self._get("/matches", event_id=event_id, lobby_id=lobby_id)

    def markets(self, lobby_id: str, match_id: str | None = None):
        params = {"lobby_id": lobby_id}
        if match_id:
            params["match_id"] = match_id
        return self._get("/markets", **params)

    def submit_batch(self, predictions: list[dict]):
        """predictions: [{market_id, lobby_id, probability(int 1-99)}, ...] <= 50 per call."""
        out = []
        for i in range(0, len(predictions), 50):
            out.append(self._post("/predictions/batch", {"predictions": predictions[i: i + 50]}))
        return out

    def update_prediction(self, prediction_id: str, probability: int):
        return self._patch(f"/predictions/{prediction_id}", {"probability": probability})

    def my_predictions(self, lobby_id: str):
        return self._get("/predictions", lobby_id=lobby_id)

    def results(self, lobby_id: str):
        return self._get("/results", lobby_id=lobby_id)
