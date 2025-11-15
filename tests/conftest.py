import os
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ["ENVIRONMENT"] = "local"
os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"


@pytest.fixture
def mock_llm_response(monkeypatch):
    def mock_invoke(self, messages, temperature=None):
        return "Test response"

    from app.services import llm

    monkeypatch.setattr(llm.LLMService, "invoke", mock_invoke)


@pytest.fixture
def mock_rag_service(monkeypatch):
    def mock_search(query, k=4):
        return [
            {
                "content": "Test insurance content",
                "source": "test_source.txt",
                "score": 0.9,
            }
        ]

    def mock_get_relevant_context(query, k=4):
        return "[Source 1: test_source.txt]\nTest insurance content"

    from app.services import rag

    monkeypatch.setattr(
        rag.RAGService, "search", lambda self, query, k=4: mock_search(query, k)
    )
    monkeypatch.setattr(
        rag.RAGService,
        "get_relevant_context",
        lambda self, query, k=4: mock_get_relevant_context(query, k),
    )
