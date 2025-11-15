from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.voice import VoiceService


class TestVoiceService:
    @pytest.mark.asyncio
    @patch("app.services.voice.AsyncOpenAI")
    async def test_transcribe_audio(self, mock_openai):
        mock_response = MagicMock()
        mock_response.text = "Hello, how can I help you?"

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client

        service = VoiceService()
        service.client = mock_client

        audio_data = b"fake_audio_data"
        result = await service.transcribe_audio(audio_data, audio_format="wav")

        assert result == "Hello, how can I help you?"
        mock_client.audio.transcriptions.create.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.voice.AsyncOpenAI")
    async def test_synthesize_speech(self, mock_openai):
        class MockStreamingResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

            async def iter_bytes(self, chunk_size):
                yield b"audio_chunk_1"
                yield b"audio_chunk_2"

        mock_streaming_response = MockStreamingResponse()
        mock_create = MagicMock(return_value=mock_streaming_response)

        mock_client = AsyncMock()
        mock_client.audio.speech.with_streaming_response.create = mock_create
        mock_openai.return_value = mock_client

        service = VoiceService()
        service.client = mock_client

        chunks = []
        async for chunk in service.synthesize_speech("Hello world"):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0] == b"audio_chunk_1"
        assert chunks[1] == b"audio_chunk_2"

    @pytest.mark.asyncio
    @patch("app.services.voice.AsyncOpenAI")
    async def test_transcribe_audio_error(self, mock_openai):
        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        mock_openai.return_value = mock_client

        service = VoiceService()
        service.client = mock_client

        with pytest.raises(Exception) as exc_info:
            await service.transcribe_audio(b"fake_audio")

        assert "API Error" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("app.services.voice.AsyncOpenAI")
    async def test_synthesize_speech_error(self, mock_openai):
        class MockFailingResponse:
            async def __aenter__(self):
                raise Exception("TTS Error")

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        mock_failing_response = MockFailingResponse()
        mock_create = MagicMock(return_value=mock_failing_response)

        mock_client = AsyncMock()
        mock_client.audio.speech.with_streaming_response.create = mock_create
        mock_openai.return_value = mock_client

        service = VoiceService()
        service.client = mock_client

        with pytest.raises(Exception) as exc_info:
            async for _ in service.synthesize_speech("Test"):
                pass

        assert "TTS Error" in str(exc_info.value)
