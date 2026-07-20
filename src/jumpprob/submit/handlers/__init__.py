"""One handler per question archetype, dispatched by name.

The platform asks the same few dozen kinds of question in endlessly varying
prose. The parser turns prose into a ``ParsedQuestion`` carrying an archetype
name; this package turns an archetype into a probability.

Handlers are grouped by what they read:

* ``goals``   — anything derived from the scoreline grid or the goal intensities
* ``counts``  — corners, cards, shots, fouls, offsides, shooter concentration
* ``players`` — individual player props and player-vs-player races
* ``events``  — timing windows, substitutions, penalties, VAR, own goals

Every handler takes a single :class:`Answering` argument and returns a
probability, or ``None`` when the question cannot be answered honestly. Return
``None`` freely: an unanswered question is ledgered loudly by the template audit,
whereas a confidently wrong one costs real points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jumpprob.submit.parser import ParsedQuestion
    from jumpprob.submit.runner import MatchContext, Runner


@dataclass
class Answering:
    """Everything a handler needs to price one question.

    Bundles the runner (for fitted tables and player rates), the parsed
    question, the match context (intensities and scoreline grid), and the
    resolved side — plus the count-model helpers most handlers want.
    """

    runner: "Runner"
    q: "ParsedQuestion"
    ctx: "MatchContext"
    side: str | None

    # -- shorthands for the most-read context fields -----------------------
    @property
    def grid(self) -> np.ndarray:
        return self.ctx.grid

    @property
    def lam_a(self) -> float:
        return self.ctx.lam_a

    @property
    def lam_b(self) -> float:
        return self.ctx.lam_b

    @property
    def lam_tot(self) -> float:
        return self.ctx.lam_a + self.ctx.lam_b

    @property
    def rho(self) -> float:
        return self.ctx.rho_dc

    @property
    def cm(self):
        """The fitted count model, or None when counts are unavailable."""
        return self.runner.pipeline.count_model

    @property
    def threshold(self) -> int | None:
        return self.q.threshold

    @property
    def half(self) -> int | None:
        return self.q.half

    @property
    def other_side(self) -> str | None:
        return "b" if self.side == "a" else "a" if self.side == "b" else None

    @property
    def league_lam_total(self) -> float:
        """Total goal intensity of an average match — the yardstick for tempo."""
        return 2.0 * float(np.exp(self.runner.pipeline.goal_model.b0))

    @property
    def tempo_scale(self) -> float:
        """This match's expected goals relative to an average one.

        Player and event rates are multiplied by this, so the same player is
        priced higher in an open tie than in a cagey one.
        """
        return self.lam_tot / self.league_lam_total

    # -- count-model helpers ----------------------------------------------
    def lams(self, s: str) -> tuple[float, float]:
        """(team, opponent) goal intensities for side ``s``."""
        return (self.lam_a, self.lam_b) if s == "a" else (self.lam_b, self.lam_a)

    def cpmf(self, fam: str, half: int, s: str) -> np.ndarray:
        """Count pmf for one family, one half, one side."""
        lt, lo = self.lams(s)
        return self.cm.pmf(fam, self.cm.mu(fam, half, lt, lo))

    def match_pmf(self, fam: str, s: str) -> np.ndarray:
        """Full-match count pmf: the convolution of both halves."""
        return np.convolve(self.cpmf(fam, 1, s), self.cpmf(fam, 2, s))

    def side_pmf(self, fam: str, s: str) -> np.ndarray:
        """Count pmf scoped to the question's half, or the whole match."""
        return self.cpmf(fam, self.half, s) if self.half else self.match_pmf(fam, s)

    def total_goals_pmf(self) -> list[float]:
        """P(total goals == k) for each k, read off the scoreline grid."""
        g, n = self.grid, self.grid.shape[0]
        return [float(sum(g[i, k - i] for i in range(k + 1) if k - i < n and i < n))
                for k in range(2 * n - 1)]


Handler = Callable[[Answering], "float | None"]

# Archetypes answered only when a count model is available. Most sit downstream
# of the count pmf; the rest read the model's fitted parameters directly.
NEEDS_COUNT_MODEL: set[str] = set()

HANDLERS: dict[str, Handler] = {}


def register(*archetypes: str, needs_counts: bool = False):
    """Bind a handler function to one or more archetype names."""

    def deco(fn: Handler) -> Handler:
        for a in archetypes:
            if a in HANDLERS:
                raise ValueError(f"duplicate handler for archetype {a!r}")
            HANDLERS[a] = fn
            if needs_counts:
                NEEDS_COUNT_MODEL.add(a)
        return fn

    return deco


# Importing the modules populates HANDLERS via the decorator.
from jumpprob.submit.handlers import counts, events, goals, players  # noqa: E402,F401


def answer(a: Answering) -> float | None:
    """Price one parsed question, or return None if we cannot answer it."""
    arch = a.q.archetype
    fn = HANDLERS.get(arch)
    if fn is None:
        return None
    if arch in NEEDS_COUNT_MODEL and a.cm is None:
        return None
    return fn(a)


__all__ = ["Answering", "HANDLERS", "NEEDS_COUNT_MODEL", "answer", "register"]
