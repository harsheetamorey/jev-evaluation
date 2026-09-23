"""Thin wrappers around TypeSafe's Jev clients.

See `evaluation.runner.SystemOneEvaluator` for the AsyncEvaluator adapter --
it's shared with `clients.llm_client` since both clients' responses conform
to the same `typesafe_sdk.SystemOneResponse` shape.
"""

from types import TracebackType
from typing import Self

from typesafe_sdk import AsyncTypeSafeClient, JSONContent, Questions, SystemOneResponse, TypeSafeClient

from config import settings


class JevClient:
    """Constructs a `TypeSafeClient` from project settings and exposes `system_one`."""

    def __init__(self) -> None:
        self._client = TypeSafeClient(api_key=settings.typesafe_api_key, model=settings.jev_model)

    def system_one(self, state: JSONContent, questions: Questions) -> SystemOneResponse:
        return self._client.system_one(state, questions)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class AsyncJevClient:
    """Async counterpart of `JevClient`, for concurrent Jev calls."""

    def __init__(self) -> None:
        self._client = AsyncTypeSafeClient(api_key=settings.typesafe_api_key, model=settings.jev_model)

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
