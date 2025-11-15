from unittest.mock import MagicMock, patch

import pytest

from app.services.llm import LLMService
from app.services.llm_provider import OpenAIProvider


class TestLLMService:
    def test_message_conversion(self):
        provider = OpenAIProvider()

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        langchain_messages = provider._convert_messages(messages)

        assert len(langchain_messages) == 3
        assert langchain_messages[0].__class__.__name__ == "SystemMessage"
        assert langchain_messages[1].__class__.__name__ == "HumanMessage"
        assert langchain_messages[2].__class__.__name__ == "AIMessage"

    @patch("app.services.llm_provider.ChatOpenAI")
    def test_invoke_with_custom_temperature(self, mock_openai):
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm.with_config.return_value = mock_llm
        mock_openai.return_value = mock_llm

        service = LLMService()
        result = service.invoke([{"role": "user", "content": "Test"}], temperature=0.5)

        assert result == "Test response"

    def test_get_embedding_model(self):
        service = LLMService()
        embedding_model = service.get_embedding_model()

        assert embedding_model is not None
        assert hasattr(embedding_model, "model")
