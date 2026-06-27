import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _env_time(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    try:
        hour_text, minute_text = value.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError:
        return default
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return default
    return f"{hour:02d}:{minute:02d}"


@dataclass(frozen=True)
class Settings:
    app_password: str = "change-me"
    session_secret: str = "change-me-long-random-string"
    app_timezone: str = "America/Winnipeg"
    database_path: str = "/data/gutcheck.db"
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "qwen3:4b"
    ollama_num_ctx: int = 4096
    ollama_num_predict: int = 1024
    ollama_timeout_seconds: float = 60.0
    garmin_tokenstore: str = "/data/garmin_tokens"
    garmin_auto_sync_enabled: bool = True
    garmin_sync_time: str = "03:15"
    garmin_sync_days: int = 14


def get_settings() -> Settings:
    return Settings(
        app_password=os.getenv("APP_PASSWORD", "change-me"),
        session_secret=os.getenv("SESSION_SECRET", "change-me-long-random-string"),
        app_timezone=os.getenv("APP_TIMEZONE", "America/Winnipeg"),
        database_path=os.getenv("DATABASE_PATH", "/data/gutcheck.db"),
        ollama_url=os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
        ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")),
        ollama_num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "1024")),
        ollama_timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60")),
        garmin_tokenstore=os.getenv("GARMIN_TOKENSTORE", "/data/garmin_tokens"),
        garmin_auto_sync_enabled=_env_bool("GARMIN_AUTO_SYNC_ENABLED", True),
        garmin_sync_time=_env_time("GARMIN_SYNC_TIME", "03:15"),
        garmin_sync_days=_env_int("GARMIN_SYNC_DAYS", 14, 1, 60),
    )
