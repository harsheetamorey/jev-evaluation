"""Project-wide settings, loaded from the environment and `.env`."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    typesafe_api_key: str | None = None
    jev_model: str = Field(default="jev-latest", validation_alias="TYPESAFE_DEFAULT_MODEL")


settings = Settings()
