"""Ingest international match results from the martj42 dataset (GitHub mirror).

Source: https://github.com/martj42/international_results — results.csv (all
internationals since 1872), goalscorers.csv (scorer + minute per goal),
shootouts.csv. Raw files are stored append-only under data/raw/martj42/ with a
fetch timestamp so backtests can reconstruct what was known when.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = ["results.csv", "goalscorers.csv", "shootouts.csv"]

RAW_DIR = Path("data/raw/martj42")


def fetch(raw_dir: Path = RAW_DIR, force: bool = False) -> dict[str, Path]:
    """Download the three CSVs. Skips download if already present unless force."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in FILES:
        dest = raw_dir / name
        if force or not dest.exists():
            resp = requests.get(f"{BASE_URL}/{name}", timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            meta = {"fetched_at": datetime.now(timezone.utc).isoformat(), "url": f"{BASE_URL}/{name}"}
            (raw_dir / f"{name}.meta.json").write_text(json.dumps(meta))
        paths[name] = dest
    return paths


def load_results(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and validate results.csv.

    Columns: date, home_team, away_team, home_score, away_score, tournament,
    city, country, neutral.
    """
    df = pd.read_csv(raw_dir / "results.csv", parse_dates=["date"])
    # Drop matches with missing scores (scheduled/abandoned fixtures appear at dataset tail).
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)

    # Exact duplicates (same date, teams AND scores) are double entries of one
    # match (e.g. Gibraltar-Cayman 2026-06-06 entered under two city spellings)
    # -> drop. Same-day same-fixture rows with DIFFERENT scores are genuine
    # doubleheaders (e.g. Tahiti-New Caledonia 1974-02-17) -> keep both.
    df = df.drop_duplicates(subset=["date", "home_team", "away_team", "home_score", "away_score"])

    _validate_results(df)
    return df.sort_values("date").reset_index(drop=True)


def load_goalscorers(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load goalscorers.csv: date, home_team, away_team, team, scorer, minute, own_goal, penalty."""
    df = pd.read_csv(raw_dir / "goalscorers.csv", parse_dates=["date"])
    df = df.dropna(subset=["minute"]).copy()
    df["minute"] = df["minute"].astype(float)
    for col in ("own_goal", "penalty"):
        df[col] = df[col].astype(bool)
    return df


def _validate_results(df: pd.DataFrame) -> None:
    # After exact-dupe removal, same-fixture-same-day rows should be a handful
    # of historical doubleheaders at most; a spike means upstream data breakage.
    doubleheaders = df.duplicated(subset=["date", "home_team", "away_team"]).sum()
    if doubleheaders > 5:
        raise ValueError(f"{doubleheaders} same-day same-fixture rows — upstream data problem?")
    if (df["home_score"] < 0).any() or (df["away_score"] < 0).any():
        raise ValueError("negative scores present")
    if df["date"].min().year < 1872:
        raise ValueError("dates before 1872 — parsing bug?")
    span_ok = df["date"].max() > pd.Timestamp("2025-01-01")
    if not span_ok:
        raise ValueError("dataset looks stale (no matches after 2025-01-01)")
