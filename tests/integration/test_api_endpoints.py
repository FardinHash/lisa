import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestChatEndpoints:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "environment" in data

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_create_session(self):
        response = client.post("/api/v1/chat/session", json={"user_id": "test_user"})
        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert data["message_count"] == 0

    def test_send_message(self, mock_llm_response, mock_rag_service):
        session_response = client.post(
            "/api/v1/chat/session", json={"user_id": "test_user"}
        )
        session_id = session_response.json()["session_id"]

        message_response = client.post(
            "/api/v1/chat/message",
            json={"session_id": session_id, "message": "What is term life insurance?"},
        )

        assert message_response.status_code == 200
        data = message_response.json()
        assert "message" in data
        assert data["session_id"] == session_id

    def test_get_session_history(self, mock_llm_response, mock_rag_service):
        session_response = client.post(
            "/api/v1/chat/session", json={"user_id": "test_user"}
        )
        session_id = session_response.json()["session_id"]

        client.post(
            "/api/v1/chat/message",
            json={"session_id": session_id, "message": "Hello"},
        )

        history_response = client.get(f"/api/v1/chat/session/{session_id}")
        assert history_response.status_code == 200
        data = history_response.json()
        assert "messages" in data
        assert len(data["messages"]) >= 1

    def test_delete_session(self):
        session_response = client.post(
            "/api/v1/chat/session", json={"user_id": "test_user"}
        )
        session_id = session_response.json()["session_id"]

        delete_response = client.delete(f"/api/v1/chat/session/{session_id}")
        assert delete_response.status_code == 204

        history_response = client.get(f"/api/v1/chat/session/{session_id}")
        assert history_response.status_code == 404

    def test_list_sessions(self):
        response = client.get("/api/v1/chat/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data
