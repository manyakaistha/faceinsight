"""Config exports for backend."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_title: str = "FaceInsight AI API"
    api_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"

    class Config:
        env_file = ".env"
        env_prefix = "FACEINSIGHT_"


settings = Settings()
