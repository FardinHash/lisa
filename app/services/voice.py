import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class VoiceService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def transcribe_audio(
        self, audio_data: bytes, audio_format: str = "wav"
    ) -> str:
        try:
            audio_file = (f"audio.{audio_format}", audio_data, f"audio/{audio_format}")

            response = await self.client.audio.transcriptions.create(
                model=settings.voice_stt_model, file=audio_file, language="en"
            )

            return response.text

        except Exception as e:
            logger.error(f"Audio transcription error: {str(e)}")
            raise

    async def synthesize_speech(self, text: str) -> AsyncGenerator[bytes, None]:
        try:
            async with self.client.audio.speech.with_streaming_response.create(
                model=settings.voice_tts_model,
                voice=settings.voice_tts_voice,
                input=text,
                speed=settings.voice_tts_speed,
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=1024):
                    yield chunk

        except Exception as e:
            logger.error(f"Speech synthesis error: {str(e)}")
            raise


voice_service = VoiceService()
