from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    openai_api_key: str

    environment: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"

    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    chroma_persist_dir: str = "./data/chroma_db"
    knowledge_base_dir: str = "./knowledge_base"

    database_url: str = "sqlite:///./data/conversations.db"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True


settings = Settings()
