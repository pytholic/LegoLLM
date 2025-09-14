"""Configuration settings for the project."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main settings for the project."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Add your settings here
    # Example:
    # app_name: str = "My App"
    # debug: bool = False
    # database_url: str = "sqlite:///./test.db"


settings = Settings()
