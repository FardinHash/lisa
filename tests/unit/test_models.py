import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models import (
    ChatRequest,
    ChatResponse,
    SessionCreate,
    SessionResponse,
    MessageRole,
    HealthResponse,
)


class TestModels:
    def test_chat_request_valid(self):
        request = ChatRequest(session_id="test-session", message="Hello")

        assert request.session_id == "test-session"
        assert request.message == "Hello"

    def test_chat_request_invalid(self):
        with pytest.raises(ValidationError):
            ChatRequest(session_id="", message="")

    def test_chat_response_creation(self):
        response = ChatResponse(
            session_id="test-session",
            message="Response message",
            sources=["source1.txt"],
            agent_reasoning="Intent: GENERAL",
            timestamp=datetime.utcnow(),
        )

        assert response.session_id == "test-session"
        assert response.message == "Response message"
        assert len(response.sources) == 1
        assert response.agent_reasoning is not None

    def test_session_create_with_user_id(self):
        session = SessionCreate(user_id="user123")

        assert session.user_id == "user123"

    def test_session_create_without_user_id(self):
        session = SessionCreate()

        assert session.user_id is None

    def test_session_response(self):
        response = SessionResponse(
            session_id="session123", created_at=datetime.utcnow(), message_count=0
        )

        assert response.session_id == "session123"
        assert response.message_count == 0
        assert response.created_at is not None

    def test_message_role_enum(self):
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"

    def test_health_response(self):
        response = HealthResponse(
            status="healthy", environment="local", timestamp=datetime.utcnow()
        )

        assert response.status == "healthy"
        assert response.environment == "local"
        assert response.timestamp is not None

