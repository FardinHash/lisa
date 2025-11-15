import pytest

from app.config import Settings


class TestSettings:
    def test_default_values(self):
        settings = Settings(openai_api_key="test-key")

        assert settings.environment == "local"
        assert settings.log_level == "INFO"
        assert settings.llm_model == "gpt-4o-mini"
        assert settings.llm_temperature == 0.7
        assert settings.llm_max_tokens == 800

    def test_embedding_config(self):
        settings = Settings(openai_api_key="test-key")

        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.embedding_dimensions == 1536

    def test_rag_config(self):
        settings = Settings(openai_api_key="test-key")

        assert settings.rag_chunk_size == 1000
        assert settings.rag_chunk_overlap == 200
        assert settings.rag_search_k == 3
        assert settings.rag_score_threshold == 0.5

    def test_memory_config(self):
        settings = Settings(openai_api_key="test-key")

        assert settings.memory_max_history == 10
        assert settings.memory_context_messages == 4

    def test_agent_config(self):
        settings = Settings(openai_api_key="test-key")

        assert settings.agent_search_k == 2
        assert settings.agent_comparison_k == 4

    def test_premium_config(self):
        settings = Settings(openai_api_key="test-key")

        assert settings.premium_base_rate == 0.05
        assert settings.premium_smoker_multiplier == 2.5

    def test_eligibility_config(self):
        settings = Settings(openai_api_key="test-key")

        assert settings.eligibility_min_age == 18
        assert settings.eligibility_max_term_age == 75
        assert settings.eligibility_senior_age == 65
        assert settings.eligibility_high_coverage == 1000000

    def test_tool_defaults(self):
        settings = Settings(openai_api_key="test-key")

        assert settings.tool_default_age == 35
        assert settings.tool_default_coverage == 500000
        assert settings.tool_default_term == 20

    def test_custom_environment(self):
        settings = Settings(openai_api_key="test-key", environment="production")

        assert settings.environment == "production"

    def test_api_config(self):
        settings = Settings(
            openai_api_key="test-key",
            api_host="0.0.0.0",
            api_port=8000,
            api_reload=True,
        )

        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.api_reload is True
