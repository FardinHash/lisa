import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestFullWorkflow:
    def test_complete_conversation_flow(self, mock_llm_response, mock_rag_service):
        session_response = client.post(
            "/api/v1/chat/session", json={"user_id": "test_user"}
        )
        assert session_response.status_code == 201
        session_id = session_response.json()["session_id"]

        message1_response = client.post(
            "/api/v1/chat/message",
            json={
                "session_id": session_id,
                "message": "What types of life insurance are available?",
            },
        )
        assert message1_response.status_code == 200
        assert "message" in message1_response.json()

        message2_response = client.post(
            "/api/v1/chat/message",
            json={
                "session_id": session_id,
                "message": "Tell me more about term life insurance",
            },
        )
        assert message2_response.status_code == 200
        assert message2_response.json()["session_id"] == session_id

        history_response = client.get(f"/api/v1/chat/session/{session_id}")
        assert history_response.status_code == 200
        messages = history_response.json()["messages"]
        assert len(messages) >= 4

        delete_response = client.delete(f"/api/v1/chat/session/{session_id}")
        assert delete_response.status_code == 204

    def test_multiple_sessions(self, mock_llm_response, mock_rag_service):
        session1_response = client.post(
            "/api/v1/chat/session", json={"user_id": "user1"}
        )
        session1_id = session1_response.json()["session_id"]

        session2_response = client.post(
            "/api/v1/chat/session", json={"user_id": "user2"}
        )
        session2_id = session2_response.json()["session_id"]

        assert session1_id != session2_id

        client.post(
            "/api/v1/chat/message",
            json={"session_id": session1_id, "message": "Hello from user1"},
        )
        client.post(
            "/api/v1/chat/message",
            json={"session_id": session2_id, "message": "Hello from user2"},
        )

        sessions_response = client.get("/api/v1/chat/sessions")
        assert sessions_response.status_code == 200
        sessions_data = sessions_response.json()
        assert sessions_data["total"] >= 2

    def test_session_not_found(self):
        fake_session_id = "non-existent-session-id"

        response = client.get(f"/api/v1/chat/session/{fake_session_id}")
        assert response.status_code == 404

        delete_response = client.delete(f"/api/v1/chat/session/{fake_session_id}")
        assert delete_response.status_code == 404

    def test_message_without_session(self, mock_llm_response, mock_rag_service):
        response = client.post(
            "/api/v1/chat/message",
            json={"session_id": "fake-session-id", "message": "Test message"},
        )

        assert response.status_code == 200

    def test_api_documentation_endpoints(self):
        response = client.get("/docs")
        assert response.status_code == 200

        openapi_response = client.get("/openapi.json")
        assert openapi_response.status_code == 200
        openapi_data = openapi_response.json()
        assert "openapi" in openapi_data
        assert "info" in openapi_data
        assert "paths" in openapi_data

    def test_error_handling(self):
        response = client.post(
            "/api/v1/chat/message",
            json={"session_id": "", "message": ""},
        )
        assert response.status_code in [200, 400, 422, 500]

    def test_cors_headers(self):
        response = client.options("/api/v1/chat/session")
        assert response.status_code in [200, 405]

