from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestVoiceAPI:
    def test_voice_status(self):
        response = client.get("/api/v1/voice/status")
        assert response.status_code == 200

        data = response.json()
        assert "voice_enabled" in data
        assert "stt_model" in data
        assert "tts_model" in data
        assert "tts_voice" in data

    def test_twilio_answer_voice_disabled(self):
        response = client.post("/api/v1/voice/twilio/answer")

        assert response.status_code in [200, 503]

    def test_websocket_voice_disabled(self):
        from starlette.websockets import WebSocketDisconnect

        with patch("app.api.voice.settings.voice_enabled", False):
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/api/v1/voice/stream") as websocket:
                    pass
