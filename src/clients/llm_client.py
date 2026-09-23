"""Thin wrapper around TypeSafe's System One Adapter (an LLM behind the Jev API).

Runs a chosen LLM provider (OpenAI, Anthropic, Gemini) behind the same
Choice/Score/Noul question shape as Jev, via `system-one-adapter`, so results
are directly comparable with `evaluation.runner.SystemOneEvaluator`.

Provider API keys are resolved by each provider's own SDK, not this
project's Settings: OPENAI_API_KEY for "openai"; GEMINI_API_KEY or
GOOGLE_API_KEY for "gemini"; ANTHROPIC_API_KEY for "anthropic".
"""

import os
from types import TracebackType
from typing import Literal, Self

from system_one_adapter import AsyncSystemOneAdapterClient
from typesafe_sdk import JSONContent, Questions, SystemOneResponse

ProviderName = Literal["openai", "anthropic", "gemini"]

# The adapter resolves its underlying provider lazily, on the first `system_one`
# call -- unlike AsyncTypeSafeClient, which validates its key eagerly at
# construction. Left unchecked, a missing key would surface as a raw
# provider-native exception (e.g. openai.OpenAIError) from *inside* the first
# example's try/except in SystemOneEvaluator.predict, which only catches
# TypeSafeError -- so it wouldn't be recorded as that example's error, it would
# crash the whole concurrent batch with a deep traceback. Checking eagerly here
# instead fails fast with one clear message, matching AsyncJevClient's behavior.
_PROVIDER_ENV_VARS: dict[ProviderName, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
}


class AsyncLlmClient:
    """Async System One Adapter client, fixed to one provider/model pair.

    Fixing provider+model per client (rather than per-call) keeps one
    `SystemOneEvaluator` instance mapped to exactly one explicit model name,
    which is what should show up as `PredictionResult.provider` -- never a
    generic "llm" label.
    """

    def __init__(self, provider: ProviderName, model: str) -> None:
        env_vars = _PROVIDER_ENV_VARS[provider]
        if not any(os.environ.get(var) for var in env_vars):
            raise RuntimeError(f"No API key found for provider {provider!r}. Set one of: {', '.join(env_vars)}.")

        self.provider = provider
        self.model = model
        self._client = AsyncSystemOneAdapterClient(
            structured_outputs=True,
            llm_answer_mode="probabilities",
            provider=provider,
            model=model,
        )

    async def system_one(self, state: JSONContent, questions: Questions) -> SystemOneResponse:
        return await self._client.system_one(state, questions)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
