"""Canonical team names across sources.

Canonical form = the martj42 results.csv spelling (it is the largest corpus and
our Elo backbone). Every other source maps INTO that. The ALIASES table is the
single source of truth; tests assert that every StatsBomb tournament team
resolves to a canonical name that actually appears in results.csv — any new
source must pass the same test before its data is used.
"""

from __future__ import annotations

# alias (as it appears in a source) -> canonical (martj42 spelling)
ALIASES: dict[str, str] = {
    # StatsBomb spellings observed in open data
    "Korea Republic": "South Korea",
    "Czech Republic": "Czech Republic",  # martj42 uses "Czech Republic" (not "Czechia")
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea DPR": "North Korea",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Cape Verde Islands": "Cape Verde",
    "Guinea Bissau": "Guinea-Bissau",
    "USA": "United States",
    "Czechia": "Czech Republic",
    "Curacao": "Curaçao",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",  # The Odds API spelling
}


def canonical(name: str) -> str:
    """Map a source team name to its canonical form (identity if unmapped)."""
    s = name.strip()
    # The platform sometimes prefixes the definite article ("the United States",
    # "the Netherlands"); no canonical name carries it, so drop it for matching.
    if s[:4].lower() == "the " and s[4:].strip():
        s = s[4:].strip()
    return ALIASES.get(s, s)
