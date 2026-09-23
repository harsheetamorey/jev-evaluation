"""Builds an (async client, explicit provider label) pair from a --providers spec string.

Specs:
    "jev"                       -> Jev, via AsyncJevClient
    "openai:gpt-4o-mini"        -> that OpenAI model, via AsyncLlmClient
    "gemini:gemini-2.0-flash"   -> that Gemini model, via AsyncLlmClient
    "anthropic:claude-..."      -> that Anthropic model, via AsyncLlmClient

The label returned is what should be used as SystemOneEvaluator's `provider`
(and therefore PredictionResult.provider) -- "jev" for Jev, or the exact
model name for an LLM, never a generic "llm".
"""

from typing import cast

from clients.jev_client import AsyncJevClient
from clients.llm_client import AsyncLlmClient, ProviderName

LLM_PROVIDER_NAMES: set[str] = {"openai", "anthropic", "gemini"}


def build_client(spec: str) -> tuple[AsyncJevClient | AsyncLlmClient, str]:
    """Parse one --providers entry into a ready-to-use client and its provider label."""
    if spec == "jev":
        return AsyncJevClient(), "jev"

    if ":" not in spec:
        raise ValueError(f"Unrecognized provider spec {spec!r}. Use 'jev' or '<provider>:<model>', e.g. 'openai:gpt-4o-mini'.")

    provider, model = spec.split(":", 1)
    if provider not in LLM_PROVIDER_NAMES:
        raise ValueError(f"Unknown LLM provider {provider!r} in {spec!r}. Expected one of {sorted(LLM_PROVIDER_NAMES)}.")
    if not model:
        raise ValueError(f"Missing model name in {spec!r}. Expected '<provider>:<model>', e.g. 'openai:gpt-4o-mini'.")

    return AsyncLlmClient(provider=cast(ProviderName, provider), model=model), model
