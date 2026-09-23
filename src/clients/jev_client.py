"""Thin wrapper around TypeSafe's `TypeSafeClient` for Jev System One calls."""

from types import TracebackType
from typing import Self

from typesafe_sdk import JSONContent, Questions, SystemOneResponse, TypeSafeClient

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
