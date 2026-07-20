"""The live pricing pass: fetch odds, price every open market, submit Bot A.

  python scripts/run_matchday_odds.py            # dry-run: fetch, anchor, show before/after
  python scripts/run_matchday_odds.py --live     # PATCH Bot A to the anchored values

Flags:
  --live            actually submit; without it nothing leaves the machine
  --matches "A||B"  restrict to named matches, as the near-kickoff tick does
  --refresh-props   re-fetch player prop lines, which move once lineups drop

The run happens in three passes, so closing-line value stays measurable after
the fact: the pure model, the odds-anchored blend we actually submit, and the
pure closing line. All three are written to data/processed/odds_anchor_run.csv
and appended to the CLV ledger.

Everything after pricing is either a safety layer or an audit, and the audits
all print rather than block. Each exists because it once caught something: a
question the parser silently dropped, a player whose name failed to join to the
book, a book line that diverged wildly from the model because our rate table had
a hole in it.

The run is deliberately hard to kill. An odds-API outage falls back to the last
cached snapshot, a single malformed quote cannot abort the anchor build, one
market's submission failure cannot abort the batch, and stdout is forced to
UTF-8 because a print statement once did exactly that.
"""

from __future__ import annotations

import sys

# The scheduler runs this through a pipe with a cp1252 stdout; printing a player
# name outside that codepage ('Demirović') raised UnicodeEncodeError and killed a
# near-kickoff tick before it reached the submit loop. Force UTF-8 first, so no
# print can ever abort a submission.
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream is not None:
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json                                                        # noqa: E402
from dataclasses import dataclass                                  # noqa: E402
from pathlib import Path                                           # noqa: E402

import pandas as pd                                                # noqa: E402

from jumpprob.ingest import odds_api                               # noqa: E402
from jumpprob.models.dynamics import Dynamics                      # noqa: E402
from jumpprob.normalize.teams import canonical                     # noqa: E402
from jumpprob.pipeline import Pipeline                             # noqa: E402
from jumpprob.simulate.mc_engine import MCEngine                   # noqa: E402
from jumpprob.submit.api_client import SportsPredictClient         # noqa: E402
from jumpprob.submit.audit import (                                # noqa: E402
    book_vs_model_divergence, crowd_clamp, is_anchor_regression,
    load_crowd_prior, previously_anchored_markets,
)
from jumpprob.submit.constants import CODE_MAP, PLAYER_ARCHS       # noqa: E402
from jumpprob.submit.parser import parse_question                  # noqa: E402
from jumpprob.submit.runner import Runner                          # noqa: E402

EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"

PROCESSED = Path("data/processed")
RUN_SHEET = PROCESSED / "odds_anchor_run.csv"
CLV_LEDGER = PROCESSED / "clv_ledger.csv"
DIVERGENCE_LEDGER = PROCESSED / "divergence_audit.csv"
TEMPLATE_LEDGER = PROCESSED / "template_audit.csv"
LINEUP_STATUS = PROCESSED / "lineup_status.json"
EXPERT_OVERRIDES = PROCESSED / "expert_overrides.json"
LINEUP_BRANCH_OVERRIDES = PROCESSED / "lineup_branch_overrides.json"

LINEUP_FRESH_H = 12.0        # ignore a scraped lineup status older than this
CLAMP_BACKED_MIN_PTS = 2     # a book move this size marks a submission "backed"
MOVED_MIN_PTS = 3            # report anchor moves at least this large
DIVERGENCE_MIN_PTS = 20      # report book-vs-model gaps at least this large


@dataclass
class Options:
    live: bool = False
    matches: set[str] | None = None
    refresh_props: bool = False


def parse_args(argv: list[str]) -> Options:
    matches = None
    if "--matches" in argv:
        matches = set(argv[argv.index("--matches") + 1].split("||"))
    return Options(live="--live" in argv, matches=matches,
                   refresh_props="--refresh-props" in argv)


def read_json(path: Path, default):
    """Read an optional artefact, falling back to ``default`` if it is absent."""
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def fetch_anchors() -> dict:
    """De-vigged book consensus per fixture, falling back to the last snapshot.

    Quota exhaustion or an outage must never kill a submission run: a book
    anchor a few hours old still beats the pure model, and the
    anchor-regression guard protects previously anchored props even if the
    cache turns out to be empty too.
    """
    print("fetching live odds...")
    try:
        events = odds_api.fetch_events()
    except Exception as e:
        events = odds_api.load_latest()
        print(f"!! live odds fetch FAILED ({str(e)[:60]}) -> using the latest CACHED "
              f"snapshot ({len(events)} events). Check The Odds API quota.")
    anchors = odds_api.build_anchors(events)
    print(f"{len(anchors)} fixtures with book consensus")
    return anchors


def build_runner() -> Runner:
    print("loading models...")
    pipeline = Pipeline()
    players = pd.read_parquet(PROCESSED / "player_match.parquet")
    # The Monte-Carlo engine answers the timing markets. count_model is None
    # because timing needs only goals; per-family calibration auto-loads.
    dynamics = Dynamics.from_json(PROCESSED / "minute_dynamics.json")
    mc = MCEngine(pipeline.goal_model, None, dynamics, n_draws=50_000,
                  coupled={"goal"}, seed=20260613)
    runner = Runner(pipeline, players, mc=mc)
    print(f"calibration loaded: {runner.calib}")
    return runner


def fetch_open_markets(client: SportsPredictClient, opts: Options) -> list[dict]:
    rows = [m for m in client.markets(LOBBY)
            if m.get("status") == "open"
            and (opts.matches is None or m["match"]["name"] in opts.matches)]
    scope = f" (filtered to {opts.matches})" if opts.matches else ""
    print(f"{len(rows)} open markets{scope}")
    return [{**m, "match": m["match"]} for m in rows]


def fetch_player_props(opts: Options) -> dict:
    """Book player lines, by fixture.

    A full run fetches; a targeted scheduler tick reuses the cached snapshot,
    because per-event prop credits add up. The closing tick passes
    --refresh-props to top up just the due matches: lineups drop about an hour
    out, after the cached snapshot was taken, and the book reprices star props
    on rotation news.
    """
    try:
        if opts.matches is not None and opts.refresh_props:
            pairs = []
            for name in opts.matches:
                codes = [t.strip() for t in name.split(" vs ")]
                if len(codes) == 2:
                    pairs.append(tuple(canonical(CODE_MAP.get(c, c)) for c in codes))
            props = odds_api.refresh_props_targeted(pairs)
            print(f"player props: targeted refresh {pairs} -> {len(props)} fixtures")
        elif opts.matches is None:
            props = odds_api.fetch_player_props()
            print(f"player props: {len(props)} fixtures")
        else:
            props = odds_api.load_latest_props()
            print(f"player props: {len(props)} fixtures")
        return props
    except Exception as e:
        try:
            props = odds_api.load_latest_props()
            print(f"!! props fetch FAILED ({str(e)[:50]}) -> latest CACHED props "
                  f"({len(props)} fixtures)")
            return props
        except Exception:
            print(f"player props unavailable ({str(e)[:50]}) -> model-only")
            return {}


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


def price_three_ways(runner: Runner, market_rows: list[dict], anchors: dict,
                     props: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Price the slate at three market weights and join them into one frame.

    The pure model (no anchors) and the pure closing line (full market weight)
    bracket the blend we submit, which is what makes closing-line value
    measurable once the match settles.
    """
    pure, _ = runner.predictions_for(market_rows)
    anchored, skipped = runner.predictions_for(market_rows, anchors=anchors,
                                               player_props=props)
    blend_w = runner.w_market
    runner.w_market = 1.0
    at_close, _ = runner.predictions_for(market_rows, anchors=anchors,
                                         player_props=props)
    runner.w_market = blend_w

    pure_by_market = {r["market_id"]: r for r in pure}
    close_by_market = {r["market_id"]: r for r in at_close}
    rows = []
    for r in anchored:
        p = pure_by_market.get(r["market_id"], {})
        m = close_by_market.get(r["market_id"], {})
        rows.append({**r, "sub_pure": p.get("sub_a", r["sub_a"]),
                     "sub_market": m.get("sub_a", r["sub_a"])})
    df = pd.DataFrame(rows)
    if len(df):
        df["delta"] = df["sub_a"] - df["sub_pure"]
    return df, skipped


def report_template_audit(skipped: list[dict]) -> None:
    """Surface every question that did not price.

    Each one is a candidate new phrasing. The header-goal and any-player-brace
    drops sat invisible for days because this list was discarded, so a new
    knockout phrasing must be seen on the first run it appears, not after it
    settles against us.
    """
    if not skipped:
        return
    print(f"\n!! TEMPLATE AUDIT: {len(skipped)} open questions NOT priced "
          f"(new phrasing/market type?):")
    for s in skipped:
        print(f"    [{s['match']}] ({s['reason']}) {s['question'][:70]}")
    ledger = pd.DataFrame(skipped)
    ledger.insert(0, "ts", pd.Timestamp.now(tz="UTC").isoformat())
    ledger.to_csv(TEMPLATE_LEDGER, mode="a",
                  header=not TEMPLATE_LEDGER.exists(), index=False)


# --------------------------------------------------------------------------
# Safety layers
# --------------------------------------------------------------------------


def apply_crowd_clamp(df: pd.DataFrame) -> pd.DataFrame:
    """Pull any unbacked submission back toward the archetype's crowd prior.

    The measured risk curve over 336 settled crowd rows: deviations of 5 to 15
    points from the crowd are positive expectation, but 25-point-plus deviations
    ran at -0.095 per market and every single loss was a bug-class extreme.
    Book-anchored markets and fringe player lows — a validated edge — are exempt
    inside crowd_clamp itself. On the current pipeline this fires zero times; it
    is pure tail insurance.
    """
    prior = load_crowd_prior()
    backed = (df["sub_market"] - df["sub_pure"]).abs() >= CLAMP_BACKED_MIN_PTS
    clamped, fired = [], []
    for row, is_backed in zip(df.itertuples(), backed):
        value, info = crowd_clamp(row.archetype, row.sub_a, bool(is_backed), prior)
        clamped.append(value)
        if info and value != int(round(row.sub_a)):
            fired.append((row.match, row.archetype, info["was"], value,
                          info["proxy"], row.question[:44]))
    df["sub_a"] = clamped
    if fired:
        print(f"\n!! crowd-proxy clamp fired on {len(fired)} markets "
              f"(unbacked extreme deviations):")
        for match, arch, was, now, proxy, question in fired:
            print(f"    {was} -> {now} (crowd prior {proxy:.0f}) [{match}] {arch}: {question}")
    return df


def apply_expert_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Apply hand-written pins for specific markets, if any are staged.

    Optional and absent by default. This existed so a human could put hard
    evidence into one market without touching code. Applied *after* the crowd
    clamp, so a fat-fingered override still cannot produce a silent extreme, and
    re-applied every tick so pins survive re-pricing until the market settles.
    Format: {market_id: {"value": int, "why": str}}.
    """
    pins = read_json(EXPERT_OVERRIDES, {})
    if not pins:
        return df
    hits = df["market_id"].astype(str).map(pins.__contains__)
    if not hits.any():
        return df
    print(f"\n!! expert overrides applied to {int(hits.sum())} markets:")
    for row in df[hits].itertuples():
        pin = pins[str(row.market_id)]
        print(f"    {row.sub_a} -> {pin['value']}  ({pin['why'][:80]}) | {row.question[:44]}")
        df.loc[df.market_id == row.market_id, "sub_a"] = int(pin["value"])
    return df


def apply_lineup_branch_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Fire conditional pins whose lineup condition is now confirmed.

    A decision table made executable. Each rule fires only when the scraped
    lineup status matches, so judgement staged hours ahead — "if Kane is
    benched, the substitute-scores question moves up" — executes automatically
    at the closing tick with nobody at the desk. Applied after the static pins,
    so confirmed team news wins. Format:
    [{"player", "status", "market_id", "value", "why"}].
    """
    rules = read_json(LINEUP_BRANCH_OVERRIDES, [])
    if not rules:
        return df
    from jumpprob.ingest.odds_api import norm_player

    status_by_player = read_json(LINEUP_STATUS, {})
    fired = []
    for rule in rules:
        entry = status_by_player.get(norm_player(str(rule.get("player", ""))))
        if not entry:
            continue
        confirmed = str(entry.get("status", "")).lower().replace("benched", "bench")
        wanted = str(rule.get("status", "")).lower().replace("benched", "bench")
        if confirmed != wanted:
            continue
        ts = entry.get("ts")
        try:
            age_h = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(ts)).total_seconds() / 3600.0
            if ts and age_h > LINEUP_FRESH_H:
                continue
        except Exception:
            pass
        hit = df["market_id"].astype(str) == str(rule.get("market_id"))
        if hit.any():
            df.loc[hit, "sub_a"] = int(rule["value"])
            fired.append(rule)
    if fired:
        print(f"\n!! LINEUP-BRANCH overrides fired ({len(fired)}):")
        for rule in fired:
            print(f"    {rule['player']} {rule['status']} -> {rule['value']}  "
                  f"({str(rule.get('why', ''))[:70]})")
    return df


def report_lineup_caps(df: pd.DataFrame) -> None:
    """Show player props the runner capped for a confirmed benched or out player.

    The cap is applied inside the runner and is already in sub_a; it is printed
    here so a bad lineup scrape is loudly visible before it ships.
    """
    if "lineup" not in df.columns or not df["lineup"].notna().any():
        return
    capped = df[df["lineup"].notna()]
    print(f"\n!! LINEUP overrides applied to {len(capped)} player props "
          f"(benched/out, capped):")
    for row in capped.itertuples():
        print(f"    -> {row.sub_a} [{row.lineup}] | {row.question[:56]}")


# --------------------------------------------------------------------------
# Audits — all reporting only, none of them change a submission
# --------------------------------------------------------------------------


def report_moves(df: pd.DataFrame) -> None:
    moved = df[df["delta"].abs() >= MOVED_MIN_PTS]
    print(f"\nanswered {len(df)} markets; {len(moved)} moved >={MOVED_MIN_PTS} pts "
          f"by the odds anchor")
    print("\nbiggest moves (pure Bot A -> odds-anchored Bot A):")
    biggest = df.reindex(df["delta"].abs().sort_values(ascending=False).index).head(20)
    for row in biggest.itertuples():
        print(f"  {row.sub_pure:>2} -> {row.sub_a:<2} ({row.delta:+d})  "
              f"[{row.match}] {row.question[:46]}")


def report_name_join_misses(runner: Runner, priced: list[dict], props: dict) -> None:
    """Flag player props whose subject matched no book key despite book data.

    This is exactly how the Haaland miss (a middle name) and the
    Sørloth/Ødegaard misses (a dropped Nordic letter) were found. Surfacing them
    every run means the next spelling class is caught immediately rather than
    after settlement.
    """
    misses = []
    for row in priced:
        if row["archetype"] not in PLAYER_ARCHS:
            continue
        teams = runner.resolve_teams(row["match"], [row["question"]])
        book = (props.get(teams) or props.get((teams[1], teams[0])) or {}) if teams else {}
        if not book:
            continue                    # no book data for this match: uncovered, not a miss
        q = parse_question(row["question"], teams={canonical(t) for t in teams})
        subject = q.subject if q else None
        if subject and not odds_api.resolve_player(subject, book):
            misses.append((row["match"], subject, row["archetype"]))
    if misses:
        print(f"\n!! {len(misses)} player props have book data but NO name match "
              f"(possible new spelling miss -> extend norm_player/_TRANSLIT or "
              f"resolve_player):")
        for match, subject, arch in misses[:15]:
            print(f"    [{match}] {subject!r}  ({arch})")


def report_book_divergence(df: pd.DataFrame, stamp: str) -> None:
    """Flag markets where the book and the model disagree sharply.

    A large gap is either a model data-gap — Haaland priced 7 against a book's
    52, because his rate never loaded — or a book market that does not actually
    match the question's scope, threshold or semantics. At 85% market weight a
    bad anchor is expensive, so these surface every run and accumulate in a
    trend ledger.
    """
    diverging = book_vs_model_divergence(df, min_pts=DIVERGENCE_MIN_PTS)
    if not len(diverging):
        return
    print(f"\n!! {len(diverging)} markets where the BOOK diverges "
          f">={DIVERGENCE_MIN_PTS}pts from the model "
          f"(data-gap or possible market mismatch — review, esp. player props):")
    for row in diverging.head(20).itertuples():
        print(f"    model {row.sub_pure:>2} -> book {row.sub_market:<3} "
              f"final {row.sub_a:<3} ({row.book_gap:+d}, lg {row.logit_gap}) "
              f"[{row.match}] {row.archetype}: {row.question[:36]}")
    ledger = diverging[["market_id", "match", "archetype", "question", "sub_pure",
                        "sub_market", "sub_a", "book_gap", "logit_gap"]].copy()
    ledger.insert(0, "ts", stamp)
    ledger.to_csv(DIVERGENCE_LEDGER, mode="a",
                  header=not DIVERGENCE_LEDGER.exists(), index=False)


def append_clv_ledger(df: pd.DataFrame, stamp: str) -> None:
    """Record all three passes per market, so CLV is measurable after settlement."""
    ledger = df[["market_id", "match", "question", "archetype",
                 "sub_pure", "sub_market", "sub_a"]].copy()
    ledger.insert(0, "ts", stamp)
    ledger.to_csv(CLV_LEDGER, mode="a", header=not CLV_LEDGER.exists(), index=False)


# --------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------


def submit(client: SportsPredictClient, df: pd.DataFrame, prev_anchored: set) -> None:
    """PATCH each changed market, one at a time, tolerating individual failures."""
    existing_by_market = {p["market_id"]: p for p in client.my_predictions(LOBBY)}
    n_patch = n_new = n_fail = n_kept = 0
    failures = []
    for row in df.itertuples():
        existing = existing_by_market.get(row.market_id)
        value = int(row.sub_a)
        # Anchor-regression guard: a run whose book prop momentarily vanished
        # must not overwrite a previously anchored player prop with the bare
        # model. Prior-run ledger evidence is required — without that condition
        # the guard once preserved a stale mis-parsed price against the
        # pipeline's own correction, which cost us the biggest single loss of
        # the round of 32.
        if is_anchor_regression(row.archetype,
                                existing["probability"] if existing else None,
                                row.sub_pure, row.sub_market,
                                was_anchored=row.market_id in prev_anchored):
            n_kept += 1
            continue
        try:
            if existing is None:
                client.submit_batch([{"market_id": row.market_id, "lobby_id": LOBBY,
                                      "probability": value}])
                n_new += 1
            elif int(existing["probability"]) != value:
                client.update_prediction(existing["id"], value)
                n_patch += 1
        except Exception as e:          # one market must never abort the batch
            n_fail += 1
            failures.append((row.market_id, str(e)[:80]))
    unchanged = len(df) - n_patch - n_new - n_fail - n_kept
    print(f"\nLIVE: PATCHed {n_patch}, submitted {n_new} new, {n_fail} failed, "
          f"{n_kept} kept (anchor-regression guard), {unchanged} already current.")
    if failures:
        print("failures (re-run to retry these):")
        for market_id, err in failures[:10]:
            print(f"  {market_id}: {err}")


# --------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    opts = parse_args(argv)
    client = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=not opts.live)

    anchors = fetch_anchors()
    runner = build_runner()
    market_rows = fetch_open_markets(client, opts)
    props = fetch_player_props(opts)

    df, skipped = price_three_ways(runner, market_rows, anchors, props)
    report_template_audit(skipped)
    if not len(df):
        # Either no matches are open, or nothing on the slate parsed — in which
        # case the template audit above has already said so, loudly.
        print("\nnothing to price (no open markets). Done.")
        return 0

    df = apply_crowd_clamp(df)
    df.to_csv(RUN_SHEET, index=False)
    df = apply_expert_overrides(df)
    df = apply_lineup_branch_overrides(df)
    report_lineup_caps(df)

    # Snapshot which markets were anchored on a PREVIOUS run before appending
    # this one: the anchor-regression guard needs prior-run evidence, and
    # appending first would let this run's own props-less rows erase it.
    prev_anchored = previously_anchored_markets()

    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    append_clv_ledger(df, stamp)

    report_moves(df)
    priced = df.to_dict("records")
    report_name_join_misses(runner, priced, props)
    report_book_divergence(df, stamp)

    if not opts.live:
        print(f"\nDRY RUN — rerun with --live to PATCH Bot A. Sheet: {RUN_SHEET}")
    else:
        submit(client, df, prev_anchored)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
