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
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from jumpprob.agents.panel import clip01, combine_logodds
from jumpprob.submit.env import load_dotenv

# Server-side web search (Tier 3): lets a panelist pull confirmed lineups, recent
# form, and injury/pundit news for the specific match past the Jan-2026 cutoff.
# _20260209 auto-filters results before they hit context (token-efficient). max_uses
# caps the per-call search count for budget. NOT combined with structured output —
# web results carry citations, which 400 with output_config.format — so the probability
# is parsed from the final text instead.
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}

# Curated multi-MODEL panel (model diversity > prompt diversity). Anthropic-only
# today; cross-provider is additive (see plan §7). Haiku may end up shadow once
# the L1 forward-test scores each model — our data decides, not a leaderboard.
PANEL = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
LINEUP_PANEL = ["claude-opus-4-8", "claude-sonnet-4-6"]  # props fan out 2-wide (cost)

# Adaptive thinking is an Opus/Sonnet-tier feature; Haiku 4.5 returns 400
# ("adaptive thinking is not supported on this model"). Omit it for those.
_NO_ADAPTIVE_THINKING = {"claude-haiku-4-5"}


class Factor(BaseModel):
    """One nuance the statistical model is blind to — the reason to move off it."""
    name: str = Field(description="the factor, e.g. 'starter ruled out', 'manager change', 'derby intensity'")
    direction: str = Field(default="none", description="'up', 'down', or 'none' — which way it moves P(YES)")
    evidence: str = Field(default="", description="the concrete, checkable basis (source/fact); empty if none")


class Forecast(BaseModel):
    probability: float = Field(ge=0.0, le=1.0, description="your reviewed P(answer is YES), 0-1")
    evidence_strength: str = Field(
        default="none",
        description="'hard' = a verifiable match-specific fact moved you (confirmed XI, suspension, "
                    "injury/ruled out); 'soft' = opinion/form/narrative; 'none' = nothing the model misses",
    )
    factors: list[Factor] = Field(
        default_factory=list,
        description="the specific nuances that moved you off the model's estimate (empty if you agree)",
    )
    rationale: str = Field(default="", description="one-sentence justification")


_EVIDENCE_RANK = {"hard": 2, "soft": 1, "none": 0}


def agg_evidence(strengths) -> str:
    """Panel-level evidence strength = the strongest any panelist claimed (hard > soft > none)."""
    best = max((_EVIDENCE_RANK.get(str(s).lower().strip(), 0) for s in strengths), default=0)
    return {2: "hard", 1: "soft", 0: "none"}[best]


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
        "You are a sharp, well-calibrated football forecaster REVIEWING a statistical model's "
        "estimate for a FIFA World Cup 2026 question. The model already captures team strength "
        "(Elo), expected goals, and historical per-90 rates — do NOT re-derive those. Your job is "
        "to adjust its estimate ONLY for real, match-specific nuance the model is blind to, such as: "
        "a key player doubtful / suspended / rested or a confirmed lineup; recent form swings, "
        "injuries, or fatigue; locker-room or managerial controversy; a coaching or tactical change; "
        "underdog motivation or dead-rubber apathy. This list is not exhaustive. If you have no "
        "concrete reason the model is wrong, RETURN ITS ESTIMATE UNCHANGED — most questions need no "
        "move. Reason strictly PRE-MATCH (never assume a result). Comparative questions are STRICT (a "
        "tie counts as NO). Claim 'hard' evidence_strength ONLY for a verifiable, match-specific FACT "
        "(a confirmed team sheet, an official suspension or injury ruling) — never for a hunch or "
        "general narrative."
    )
    if role == "lineup":
        return base + (
            " This is a PLAYER prop. The single biggest thing the model misses is availability and "
            "minutes: whether the player actually starts, is benched, rotated, doubtful, or ruled out. "
            "Weigh his expected minutes and role against his per-90 track record and the opponent's "
            "defensive strength. A rotated or obscure player should regress toward the base rate; a "
            "proven high-volume starter must NOT be underpriced; a player confirmed to not start should "
            "drop hard toward zero."
        )
    return base + (
        " This is a team / match question. Adjust for matchup-specific tactics, game-state, and "
        "motivation the goal and count models can't see."
    )


def build_user(question: str, model_pct: float, grounding: dict | None = None) -> str:
    g = grounding or {}
    matchup = g.get("matchup")
    base = g.get("base_rate")
    lines = [f"Question: {question}"]
    if matchup:
        lines.append(f"Match: {matchup} (infer the subject's side and the opponent).")
    lines.append(f"Our statistical model estimates YES at {model_pct * 100:.0f}%.")
    if base is not None:
        lines.append(
            f"Empirical base rate for this exact market type: {base * 100:.0f}%. Treat this as "
            "the anchor — the track record and the matchup are reasons to move ABOVE or BELOW "
            "it, but most cases land near the base rate. Do not over-price a favourite."
        )
    if g.get("exp_goals"):
        lines.append(f"Our model's expected goals: {g['exp_goals']}.")
    if g.get("elo"):
        lines.append(f"Team strength (Elo): {g['elo']}.")
    if g.get("resolved"):
        lines.append(f"Player: {g['resolved']}"
                     + (f" ({g.get('position')}, {g.get('league')})" if g.get("position") else ""))
    prof = {k: v for k, v in g.items()
            if k not in ("matchup", "base_rate", "note", "resolved", "position", "league",
                         "exp_goals", "elo")
            and v is not None and v != ""}
    if prof:
        lines.append("Track record (per-90 club form — evidence our model under-uses):")
        for k, v in prof.items():
            lines.append(f"  - {k}: {v}")
    if g.get("note"):
        lines.append(f"  ({g['note']})")
    lines.append(
        "Review the model's estimate above. If real, match-specific nuance justifies a move, give your "
        "adjusted probability (0-1), set evidence_strength ('hard' ONLY for a verifiable fact such as a "
        "confirmed lineup / suspension / injury ruling, else 'soft' or 'none'), and list the factor(s) "
        "that moved you with their concrete evidence. If nothing concrete contradicts the model, return "
        "its estimate unchanged with an empty factor list and a one-sentence rationale."
    )
    return "\n".join(lines)


def forecast_one(question: str, model_pct: float, model: str, role: str = "tactical",
                 grounding: dict | None = None, max_tokens: int = 2500) -> "Forecast | None":
    """One panelist's reviewed forecast (probability clipped, plus evidence_strength +
    factors). None on no-key / error. Callers read `.probability` for the ensemble."""
    c = client()
    if c is None:
        return None
    try:
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=build_system(role),
            messages=[{"role": "user", "content": build_user(question, model_pct, grounding)}],
            output_format=Forecast,
        )
        if model not in _NO_ADAPTIVE_THINKING:
            kwargs["thinking"] = {"type": "adaptive"}
        r = c.messages.parse(**kwargs)
        f = r.parsed_output
        if f is None:
            return None
        f.probability = clip01(float(f.probability))
        return f
    except Exception as e:  # provider hiccup, parse failure, rate limit — drop this panelist
        print(f"  ! {model} failed on {question[:46]!r}: {str(e)[:70]}")
        return None


_PROB_RE = re.compile(r"PROBABILITY:\s*([0-9]*\.?[0-9]+)\s*(%?)", re.I)
_EVIDENCE_RE = re.compile(r"EVIDENCE_STRENGTH:\s*(hard|soft|none)", re.I)


def _parse_evidence_strength(text: str) -> str:
    """Pull the web reviewer's 'EVIDENCE_STRENGTH: <hard|soft|none>' line. Default 'none'."""
    m = _EVIDENCE_RE.findall(text or "")
    return m[-1].lower() if m else "none"


def _parse_prob(text: str) -> float | None:
    """Pull the final 'PROBABILITY: <x>' the model is instructed to emit. % -> fraction."""
    matches = _PROB_RE.findall(text or "")
    if not matches:
        return None
    val, pct = matches[-1]                       # last one = the final answer
    try:
        p = float(val)
    except ValueError:
        return None
    if pct or p > 1.0:
        p /= 100.0
    return clip01(p)


def forecast_web(question: str, model_pct: float, grounding: dict | None = None,
                 model: str = "claude-opus-4-8", role: str = "lineup",
                 max_uses: int = 3, max_tokens: int = 4000) -> tuple[float | None, str]:
    """One web-search-grounded forecast: the model searches for the confirmed lineup,
    recent form, and injury/pundit news for THIS match, then prices the question.
    Returns (probability|None, rationale_text). max_uses bounds search cost."""
    c = client()
    if c is None:
        return None, ""
    system = build_system(role) + (
        " You have a web_search tool. Search for the CONFIRMED starting lineup / team news, "
        "the player's recent form, and any injury, suspension, or rotation news for THIS specific "
        "match — then review the model's estimate. If a player is benched, doubtful, or out, weight "
        "that heavily. Stay strictly pre-match (never use the actual result). End your reply with "
        "EXACTLY these two final lines:\n"
        "PROBABILITY: <p>   (your reviewed YES probability, 0-1)\n"
        "EVIDENCE_STRENGTH: <hard|soft|none>   ('hard' ONLY for a verifiable fact you actually found "
        "in search — a confirmed XI, an official suspension or injury ruling; 'soft' for form/opinion; "
        "'none' if nothing concrete contradicts the model)"
    )
    user = build_user(question, model_pct, grounding)
    tool = {**WEB_SEARCH_TOOL, "max_uses": max_uses}
    msgs = [{"role": "user", "content": user}]
    r = None
    try:
        for _ in range(4):  # resume server-tool loop on pause_turn
            r = c.messages.create(model=model, max_tokens=max_tokens,
                                  thinking={"type": "adaptive"}, system=system,
                                  messages=msgs, tools=[tool])
            if r.stop_reason != "pause_turn":
                break
            msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": r.content}]
    except Exception as e:
        print(f"  ! {model} web failed on {question[:46]!r}: {str(e)[:70]}")
        return None, ""
    text = "\n".join(b.text for b in r.content if getattr(b, "type", None) == "text")
    searches = sum(1 for b in r.content if getattr(b, "type", None) == "server_tool_use")
    return _parse_prob(text), text + f"\n[searches={searches}]"


# Web-search panel: 1-wide (Opus) by default for budget; add Sonnet for a 2-wide
# ensemble when budget allows (each web call is ~$0.2-0.4).
WEB_PANEL = ["claude-opus-4-8"]


def web_forecast(question: str, model_pct: float, grounding: dict | None = None,
                 models: list[str] | None = None, role: str = "lineup",
                 max_uses: int = 3) -> dict:
    """Web-search ensemble for one question: each model searches live lineup/form/news
    then prices it; robust-combine. Sequential (web search is slow and few questions
    run per slate). Returns {'models': {m:p}, 'p_ensemble', 'n', 'texts'}."""
    models = models or WEB_PANEL
    out: dict[str, float | None] = {}
    texts: dict[str, str] = {}
    for m in models:
        p, t = forecast_web(question, model_pct, grounding, model=m, role=role, max_uses=max_uses)
        out[m], texts[m] = p, t
    probs = [p for p in out.values() if p is not None]
    # evidence_strength only counts from panelists that actually returned a probability
    strengths = [_parse_evidence_strength(texts[m]) for m, p in out.items() if p is not None]
    return {"models": out, "p_ensemble": combine_logodds(probs), "n": len(probs),
            "texts": texts, "evidence_strength": agg_evidence(strengths)}


def panel_forecast(question: str, model_pct: float, role: str = "tactical",
                   grounding: dict | None = None, models: list[str] | None = None) -> dict:
    """Fan out across the panel (one thread per model); robust-combine the results.
    Returns {'models': {model: prob|None}, 'p_ensemble', 'n', 'evidence_strength', 'factors'}.
    The ensemble math is unchanged — only `.probability` feeds combine_logodds; the
    evidence_strength + factors are captured for the Phase-2 deconfliction verifier."""
    models = models or (LINEUP_PANEL if role == "lineup" else PANEL)
    res: dict[str, "Forecast | None"] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(models))) as ex:
        futs = {ex.submit(forecast_one, question, model_pct, m, role, grounding): m for m in models}
        for fu, m in futs.items():
            try:
                res[m] = fu.result()
            except Exception:
                res[m] = None
    out = {m: (f.probability if f is not None else None) for m, f in res.items()}
    probs = [p for p in out.values() if p is not None]
    dets = [f for f in res.values() if f is not None]
    return {
        "models": out,
        "p_ensemble": combine_logodds(probs),
        "n": len(probs),
        "evidence_strength": agg_evidence([d.evidence_strength for d in dets]),
        "factors": [f.model_dump() for d in dets for f in d.factors],
    }
