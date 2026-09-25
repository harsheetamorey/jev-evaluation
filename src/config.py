"""Project-wide settings, loaded from the environment and `.env`."""

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Export .env into os.environ as well. pydantic-settings reads .env only to
# populate this Settings object -- it does NOT put those values in os.environ.
# Provider SDKs (OpenAI, Gemini, ...) resolve their own keys straight from
# os.environ, so without this an OPENAI_API_KEY correctly placed in .env is
# invisible to them and every LLM call fails with "no API key".
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    typesafe_api_key: str | None = None
    jev_model: str = Field(default="jev-latest", validation_alias="TYPESAFE_DEFAULT_MODEL")


settings = Settings()
