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
    llm_max_tokens: int = 800

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    chroma_persist_dir: str = "./data/chroma_db"
    knowledge_base_dir: str = "./knowledge_base"

    database_url: str = "sqlite:///./data/conversations.db"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_search_k: int = 3
    rag_score_threshold: float = 0.5
    rag_file_encoding: str = "utf-8"
    rag_file_glob: str = "**/*.txt"

    memory_max_history: int = 10
    memory_context_messages: int = 4

    agent_search_k: int = 2
    agent_comparison_k: int = 4

    premium_base_rate: float = 0.05
    premium_smoker_multiplier: float = 2.5

    eligibility_min_age: int = 18
    eligibility_max_term_age: int = 75
    eligibility_senior_age: int = 65
    eligibility_high_coverage: int = 1000000

    tool_default_age: int = 35
    tool_default_coverage: int = 500000
    tool_default_term: int = 20

    voice_enabled: bool = False
    voice_stt_model: str = "whisper-1"
    voice_tts_model: str = "tts-1"
    voice_tts_voice: str = "nova"
    voice_tts_speed: float = 1.0

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    intent_classification_temperature: float = 0.3
    tool_selection_temperature: float = 0.2

    tool_age_min: int = 18
    tool_age_max: int = 85
    tool_coverage_min: int = 10000
    tool_coverage_max: int = 10000000
    tool_term_min: int = 5
    tool_term_max: int = 40

    cache_enabled: bool = True
    cache_ttl: int = 3600
    cache_max_size: int = 1000

    rate_limit_enabled: bool = True
    rate_limit_calls: int = 100
    rate_limit_period: int = 60

    streaming_enabled: bool = True
    streaming_chunk_size: int = 50

    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 3600

settings = Settings()
