"""LLM expert caller — the OUT-OF-BAND multi-model panel.

NEVER imported by the live daemon (it pulls anthropic + makes network calls).
Run from scripts/ai_panel_write.py, which caches the result to
data/processed/ai_panel.json for jumpprob.agents.panel to hot-read.

Each panelist gets the SAME grounded prompt — the question, our statistical prior
(model_pct) as the baseline to reason against, and the subject's TRACK RECORD —
and returns a single calibrated YES probability via structured output
(messages.parse + Pydantic, adaptive thinking). The panel is combined with a
robust trim-median in log-odds (jumpprob.agents.panel.combine_logodds), so a single
rogue model cannot set the price.

Graceful degradation: no ANTHROPIC_API_KEY -> client() is None -> forecasts are
None (the writer simply emits no entry; the daemon falls back to the pure model).
A single model erroring is dropped from the panel, not fatal.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from jumpprob.agents.panel import clip01, combine_logodds
from jumpprob.submit.env import load_dotenv

# Curated multi-MODEL panel (model diversity > prompt diversity). Anthropic-only
# today; cross-provider is additive (see plan §7). Haiku may end up shadow once
# the L1 forward-test scores each model — our data decides, not a leaderboard.
PANEL = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
LINEUP_PANEL = ["claude-opus-4-8", "claude-sonnet-4-6"]  # props fan out 2-wide (cost)


class Forecast(BaseModel):
    probability: float = Field(ge=0.0, le=1.0, description="P(answer is YES), 0-1")
    rationale: str = Field(default="", description="one-sentence justification")


_client = None
_no_key = False


def client():
    """Lazy Anthropic client; None if no key (so this module imports + tests offline)."""
    global _client, _no_key
    if _client is not None:
        return _client
    if _no_key:
        return None
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _no_key = True
        return None
    import anthropic
    _client = anthropic.Anthropic()
    return _client


def build_system(role: str) -> str:
    base = (
        "You are a sharp, well-calibrated football forecaster for the FIFA World Cup 2026. "
        "Reason strictly PRE-MATCH: never assume any result has occurred. Comparative "
        "questions are STRICT — a tie counts as NO. Return a single calibrated probability "
        "that the answer is YES."
    )
    if role == "lineup":
        return base + (
            " You specialise in player props. Weigh the player's historical per-90 rate, "
            "his expected minutes given the (confirmed, if provided) lineup, and the opponent's "
            "defensive strength. A rotated or obscure player should regress toward the base rate; "
            "a proven high-volume player must NOT be underpriced just because a naive model is shy."
        )
    return base + (
        " You reason tactically about team styles, the matchup, game-state and motivation for "
        "whole-match, half-specific and compound questions."
    )


def build_user(question: str, model_pct: float, grounding: dict | None = None) -> str:
    lines = [
        f"Question: {question}",
        f"A baseline statistical model estimates YES at {model_pct * 100:.0f}%.",
    ]
    if grounding:
        lines.append("Track record (evidence our model under-uses — weigh it):")
        for k, v in grounding.items():
            if v is not None and v != "":
                lines.append(f"  - {k}: {v}")
    lines.append(
        "Give your single best-calibrated probability (0-1) that the answer is YES, "
        "and a one-sentence rationale."
    )
    return "\n".join(lines)


def forecast_one(question: str, model_pct: float, model: str, role: str = "tactical",
                 grounding: dict | None = None, max_tokens: int = 2500) -> float | None:
    """One panelist's calibrated probability, clipped. None on no-key / error."""
    c = client()
    if c is None:
        return None
    try:
        r = c.messages.parse(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=build_system(role),
            messages=[{"role": "user", "content": build_user(question, model_pct, grounding)}],
            output_format=Forecast,
        )
        f = r.parsed_output
        if f is None:
            return None
        return clip01(float(f.probability))
    except Exception as e:  # provider hiccup, parse failure, rate limit — drop this panelist
        print(f"  ! {model} failed on {question[:46]!r}: {str(e)[:70]}")
        return None


def panel_forecast(question: str, model_pct: float, role: str = "tactical",
                   grounding: dict | None = None, models: list[str] | None = None) -> dict:
    """Fan out across the panel (one thread per model); robust-combine the results.
    Returns {'models': {model: prob|None}, 'p_ensemble': float|None, 'n': int}."""
    models = models or (LINEUP_PANEL if role == "lineup" else PANEL)
    out: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(models))) as ex:
        futs = {ex.submit(forecast_one, question, model_pct, m, role, grounding): m for m in models}
        for fu, m in futs.items():
            try:
                out[m] = fu.result()
            except Exception:
                out[m] = None
    probs = [p for p in out.values() if p is not None]
    return {"models": out, "p_ensemble": combine_logodds(probs), "n": len(probs)}
