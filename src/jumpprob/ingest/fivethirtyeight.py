"""FiveThirtyEight SPI match forecasts (archived; model discontinued 2023).

spi_matches.csv contains pre-match prob1/prob2/probtie + projected scores for
~2016-2023 including FIFA World Cup 2018 and 2022. We use these as a
market-quality external forecast to (a) fit ensemble blend weights on real
tournament data and (b) benchmark our model against a strong public model.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# 538 shut down in 2023; projects.fivethirtyeight.com now serves HTML and the
# GitHub data repo was pruned to a README. The Wayback Machine snapshot of
# 2023-04-27 (post-WC2022) is the canonical surviving copy ("id_" = raw bytes).
URL = (
    "https://web.archive.org/web/20230427025826id_/"
    "https://projects.fivethirtyeight.com/soccer-api/international/spi_matches_intl.csv"
)
URL_FALLBACK = (
    "https://web.archive.org/web/20230427025826/"
    "https://projects.fivethirtyeight.com/soccer-api/international/spi_matches_intl.csv"
)

RAW_DIR = Path("data/raw/fivethirtyeight")


def fetch(raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "spi_matches.csv"
    if force or not dest.exists():
        for url in (URL, URL_FALLBACK):
            try:
                resp = requests.get(url, timeout=300)
                resp.raise_for_status()
                # discontinued endpoints can return HTML with HTTP 200 —
                # validate it parses as the expected CSV before accepting
                head = resp.content[:200].lstrip()
                if not head.startswith(b"season") and b"date" not in head:
                    continue
                dest.write_bytes(resp.content)
                (raw_dir / "spi_matches.meta.json").write_text(
                    json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(), "url": url})
                )
                break
            except requests.RequestException:
                continue
        else:
            raise RuntimeError("could not fetch SPI matches from any source")
    return dest


def load_world_cup_forecasts(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "spi_matches.csv", parse_dates=["date"])
    # final tournaments only (qualifiers excluded by exact league names);
    # the 2023-04 snapshot covers WC 2022 and Euro 2020 from our backtest set
    finals = ["FIFA World Cup", "European Championships", "Copa America", "African Cup of Nations"]
    wc = df[df["league"].isin(finals)].copy()
    wc = wc.dropna(subset=["prob1", "prob2", "probtie"])
    cols = ["date", "team1", "team2", "prob1", "prob2", "probtie", "proj_score1", "proj_score2"]
    return wc[[c for c in cols if c in wc.columns]].reset_index(drop=True)
