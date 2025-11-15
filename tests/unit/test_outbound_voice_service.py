from unittest.mock import MagicMock, patch

import pytest

from app.services.voice import VoiceService


class TestOutboundVoiceService:
    @patch("app.services.voice.Client")
    def test_initiate_outbound_call_success(self, mock_twilio_client_class):
        mock_client_instance = MagicMock()
        mock_call = MagicMock()
        mock_call.sid = "CA1234567890abcdef"
        mock_call.status = "queued"

        mock_client_instance.calls.create.return_value = mock_call
        mock_twilio_client_class.return_value = mock_client_instance

        service = VoiceService()
        service.twilio_client = mock_client_instance

        result = service.initiate_outbound_call(
            to_number="+19876543210",
            callback_url="https://example.com/voice/connect",
            initial_message="Test message",
        )

        assert result["call_sid"] == "CA1234567890abcdef"
        assert result["to"] == "+19876543210"
        assert result["status"] == "queued"

        mock_client_instance.calls.create.assert_called_once()

    def test_initiate_outbound_call_no_client(self):
        service = VoiceService()
        service.twilio_client = None

        with pytest.raises(ValueError, match="Twilio client not initialized"):
            service.initiate_outbound_call(
                to_number="+19876543210",
                callback_url="https://example.com/voice/connect",
            )

    @patch("app.services.voice.Client")
    def test_get_call_status_success(self, mock_twilio_client_class):
        mock_client_instance = MagicMock()
        mock_call = MagicMock()
        mock_call.sid = "CA1234567890abcdef"
        mock_call.status = "in-progress"
        mock_call.duration = "45"
        mock_call.to = "+19876543210"
        mock_call.from_ = "+18574038869"
        mock_call.direction = "outbound-api"

        mock_client_instance.calls.return_value.fetch.return_value = mock_call
        mock_twilio_client_class.return_value = mock_client_instance

        service = VoiceService()
        service.twilio_client = mock_client_instance

        result = service.get_call_status("CA1234567890abcdef")

        assert result["call_sid"] == "CA1234567890abcdef"
        assert result["status"] == "in-progress"
        assert result["duration"] == "45"

    def test_get_call_status_no_client(self):
        service = VoiceService()
        service.twilio_client = None

        with pytest.raises(ValueError, match="Twilio client not initialized"):
            service.get_call_status("CA1234567890abcdef")

    @patch("app.services.voice.Client")
    def test_hangup_call_success(self, mock_twilio_client_class):
        mock_client_instance = MagicMock()
        mock_call = MagicMock()
        mock_call.sid = "CA1234567890abcdef"
        mock_call.status = "completed"

        mock_client_instance.calls.return_value.update.return_value = mock_call
        mock_twilio_client_class.return_value = mock_client_instance

        service = VoiceService()
        service.twilio_client = mock_client_instance

        result = service.hangup_call("CA1234567890abcdef")

        assert result["call_sid"] == "CA1234567890abcdef"
        assert result["status"] == "completed"

    def test_hangup_call_no_client(self):
        service = VoiceService()
        service.twilio_client = None

        with pytest.raises(ValueError, match="Twilio client not initialized"):
            service.hangup_call("CA1234567890abcdef")

    @patch("app.services.voice.Client")
    def test_initiate_call_with_error(self, mock_twilio_client_class):
        mock_client_instance = MagicMock()
        mock_client_instance.calls.create.side_effect = Exception("API Error")
        mock_twilio_client_class.return_value = mock_client_instance

        service = VoiceService()
        service.twilio_client = mock_client_instance

        with pytest.raises(Exception, match="API Error"):
            service.initiate_outbound_call(
                to_number="+19876543210",
                callback_url="https://example.com/voice/connect",
            )
