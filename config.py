from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")

settings = Settings()

# Ensure directories exist
settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
settings.log_dir.mkdir(parents=True, exist_ok=True)
