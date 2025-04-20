from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    num_node_features: int = 16  # Default value, adjust as needed
    num_classes: int = 2  # Default value, adjust as needed
    hidden_dim: int = 16  # Default hidden dimension


settings = Settings()
model_settings = ModelSettings()

# Ensure directories exist
settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
settings.log_dir.mkdir(parents=True, exist_ok=True)
