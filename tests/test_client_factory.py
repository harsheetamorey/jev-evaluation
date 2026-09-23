"""Unit tests for clients.factory.build_client.

Checks spec parsing and the resulting provider label; doesn't exercise the
real network clients (those need real API keys). The "jev" spec is the one
exception: unlike the LLM adapter (which resolves its provider lazily on
first call), AsyncTypeSafeClient validates its API key eagerly at
construction, so that one case needs a real TYPESAFE_API_KEY to build.
"""

import pytest

from clients.factory import build_client
from clients.jev_client import AsyncJevClient
from clients.llm_client import AsyncLlmClient
from config import settings


def test_jev_spec_builds_jev_client() -> None:
    if not settings.typesafe_api_key:
        pytest.skip("TYPESAFE_API_KEY not set; AsyncJevClient validates its key at construction")
    client, label = build_client("jev")
    assert isinstance(client, AsyncJevClient)
    assert label == "jev"


def test_llm_spec_uses_exact_model_name_as_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client, label = build_client("openai:gpt-4o-mini")
    assert isinstance(client, AsyncLlmClient)
    assert label == "gpt-4o-mini"
    assert client.provider == "openai"
    assert client.model == "gpt-4o-mini"


def test_gemini_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client, label = build_client("gemini:gemini-2.0-flash")
    assert client.provider == "gemini"
    assert label == "gemini-2.0-flash"


def test_llm_spec_without_api_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No API key found for provider 'openai'"):
        build_client("openai:gpt-4o-mini")


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_client("cohere:command-r")


def test_missing_colon_raises() -> None:
    with pytest.raises(ValueError, match="Unrecognized provider spec"):
        build_client("gpt-4o-mini")


def test_missing_model_raises() -> None:
    with pytest.raises(ValueError, match="Missing model name"):
        build_client("openai:")
