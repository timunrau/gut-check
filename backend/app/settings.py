import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_password: str = "change-me"
    session_secret: str = "change-me-long-random-string"
    app_timezone: str = "America/Winnipeg"
    database_path: str = "/data/gutcheck.db"
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:1.5b"


def get_settings() -> Settings:
    return Settings(
        app_password=os.getenv("APP_PASSWORD", "change-me"),
        session_secret=os.getenv("SESSION_SECRET", "change-me-long-random-string"),
        app_timezone=os.getenv("APP_TIMEZONE", "America/Winnipeg"),
        database_path=os.getenv("DATABASE_PATH", "/data/gutcheck.db"),
        ollama_url=os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
    )

