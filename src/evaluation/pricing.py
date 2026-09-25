"""Per-model token pricing, for estimating request cost from token counts.

Provenance matters here, so it is recorded per entry:

- `gpt-4o-mini` rates were read from OpenAI's own pricing page
  (developers.openai.com/api/docs/pricing) on 2026-09-23.
- The Jev rate came from the project owner's notes, NOT from a vendor page
  this code has verified. Treat Jev cost figures as approximate until
  confirmed against TypeSafe's published pricing.

A model with no entry below yields `estimated_cost_usd=None` rather than a
guessed number. Rates change -- re-check before quoting any cost publicly.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPricing:
    """USD per 1,000,000 tokens."""

    input_per_million_usd: float
    output_per_million_usd: float


PRICING: dict[str, TokenPricing] = {
    # Jev bills input only; output tokens are free because nothing is generated.
    # Source: project owner's notes (unverified against a vendor page).
    "jev-1.13.0": TokenPricing(input_per_million_usd=0.042, output_per_million_usd=0.0),
    # Source: OpenAI pricing page, read 2026-09-23.
    "gpt-4o-mini": TokenPricing(input_per_million_usd=0.15, output_per_million_usd=0.60),
}


def estimate_cost_usd(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """Estimate USD cost from token counts, or None if `model` has no pricing entry."""
    pricing = PRICING.get(model)
    if pricing is None or input_tokens is None or output_tokens is None:
        return None
    return (input_tokens * pricing.input_per_million_usd + output_tokens * pricing.output_per_million_usd) / 1_000_000
