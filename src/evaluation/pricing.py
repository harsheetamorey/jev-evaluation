"""Per-model token pricing, for estimating request cost from token counts.

Rates here are NOT verified against any provider's live pricing page -- they
are placeholders you must fill in and keep current yourself before trusting
any estimated_cost_usd figure this project produces. A model with no entry
below yields `estimated_cost_usd=None` rather than a guessed number.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPricing:
    """USD per 1,000,000 tokens."""

    input_per_million_usd: float
    output_per_million_usd: float


# Populate with the exact model name(s) you run (e.g. the string in
# PredictionResult.provider for an LLM, or the Jev model returned in
# SystemOneResponse.model) and verified current rates. Left empty by
# default so cost is honestly "unknown" rather than fabricated.
PRICING: dict[str, TokenPricing] = {}


def estimate_cost_usd(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """Estimate USD cost from token counts, or None if `model` has no pricing entry."""
    pricing = PRICING.get(model)
    if pricing is None or input_tokens is None or output_tokens is None:
        return None
    return (input_tokens * pricing.input_per_million_usd + output_tokens * pricing.output_per_million_usd) / 1_000_000
