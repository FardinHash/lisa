from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestOutboundVoice:
    @patch("app.services.voice.voice_service.twilio_client")
    def test_initiate_outbound_call_success(self, mock_twilio_client):
        mock_call = MagicMock()
        mock_call.sid = "CA1234567890abcdef"
        mock_call.status = "queued"

        mock_twilio_client.calls.create.return_value = mock_call

        response = client.post(
            "/api/v1/voice/outbound/initiate",
            json={
                "to_number": "+19876543210",
                "initial_message": "Hello, this is a test call",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["call_sid"] == "CA1234567890abcdef"
        assert data["to"] == "+19876543210"
        assert data["status"] == "queued"
        assert "initiated" in data["message"].lower()

    def test_initiate_outbound_call_voice_disabled(self):
        with patch("app.config.settings.voice_enabled", False):
            response = client.post(
                "/api/v1/voice/outbound/initiate",
                json={"to_number": "+19876543210"},
            )

            assert response.status_code == 503
            assert "disabled" in response.json()["detail"].lower()

    @patch("app.services.voice.voice_service.twilio_client", None)
    def test_initiate_outbound_call_no_twilio_client(self):
        response = client.post(
            "/api/v1/voice/outbound/initiate",
            json={"to_number": "+19876543210"},
        )

        assert response.status_code == 500
        assert "twilio" in response.json()["detail"].lower()

    @patch("app.services.voice.voice_service.twilio_client")
    def test_get_call_status_success(self, mock_twilio_client):
        mock_call = MagicMock()
        mock_call.sid = "CA1234567890abcdef"
        mock_call.status = "in-progress"
        mock_call.duration = "45"
        mock_call.to = "+19876543210"
        mock_call.from_ = "+18574038869"
        mock_call.direction = "outbound-api"

        mock_twilio_client.calls.return_value.fetch.return_value = mock_call

        response = client.get("/api/v1/voice/call/CA1234567890abcdef/status")

        assert response.status_code == 200
        data = response.json()
        assert data["call_sid"] == "CA1234567890abcdef"
        assert data["status"] == "in-progress"
        assert data["duration"] == "45"

    @patch("app.services.voice.voice_service.twilio_client")
    def test_hangup_call_success(self, mock_twilio_client):
        mock_call = MagicMock()
        mock_call.sid = "CA1234567890abcdef"
        mock_call.status = "completed"

        mock_twilio_client.calls.return_value.update.return_value = mock_call

        response = client.post("/api/v1/voice/call/CA1234567890abcdef/hangup")

        assert response.status_code == 200
        data = response.json()
        assert data["call_sid"] == "CA1234567890abcdef"
        assert data["status"] == "completed"
        assert "terminated" in data["message"].lower()

    def test_outbound_connect_webhook(self):
        response = client.post("/api/v1/voice/outbound/connect")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/xml"
        assert "<Response>" in response.text
        assert "<Say" in response.text
        assert "<Stream" in response.text

    def test_outbound_status_callback(self):
        response = client.post("/api/v1/voice/outbound/connect/status")

        assert response.status_code == 200
        assert response.text == "OK"

    def test_invalid_phone_number(self):
        response = client.post(
            "/api/v1/voice/outbound/initiate",
            json={"to_number": "123"},
        )

        assert response.status_code == 422
