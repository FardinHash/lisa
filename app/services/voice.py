import logging
from typing import AsyncGenerator, Optional

from openai import AsyncOpenAI
from twilio.rest import Client

from app.config import settings

logger = logging.getLogger(__name__)


class VoiceService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.twilio_client: Optional[Client] = None
        if settings.twilio_account_sid and settings.twilio_auth_token:
            self.twilio_client = Client(
                settings.twilio_account_sid, settings.twilio_auth_token
            )

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

    def initiate_outbound_call(
        self, to_number: str, callback_url: str, initial_message: Optional[str] = None
    ) -> dict:
        if not self.twilio_client:
            raise ValueError("Twilio client not initialized. Check credentials.")

        try:
            call = self.twilio_client.calls.create(
                to=to_number,
                from_=settings.twilio_phone_number,
                url=callback_url,
                method="POST",
                status_callback=f"{callback_url}/status",
                status_callback_event=["initiated", "ringing", "answered", "completed"],
                status_callback_method="POST",
            )

            logger.info(f"Initiated outbound call to {to_number}, SID: {call.sid}")
            return {
                "call_sid": call.sid,
                "to": to_number,
                "from": settings.twilio_phone_number,
                "status": call.status,
            }

        except Exception as e:
            logger.error(f"Failed to initiate outbound call: {str(e)}")
            raise

    def get_call_status(self, call_sid: str) -> dict:
        if not self.twilio_client:
            raise ValueError("Twilio client not initialized. Check credentials.")

        try:
            call = self.twilio_client.calls(call_sid).fetch()
            return {
                "call_sid": call.sid,
                "status": call.status,
                "duration": call.duration,
                "to": call.to,
                "from": call.from_formatted,
                "direction": call.direction,
            }

        except Exception as e:
            logger.error(f"Failed to fetch call status: {str(e)}")
            raise

    def hangup_call(self, call_sid: str) -> dict:
        if not self.twilio_client:
            raise ValueError("Twilio client not initialized. Check credentials.")

        try:
            call = self.twilio_client.calls(call_sid).update(status="completed")
            logger.info(f"Terminated call: {call_sid}")
            return {"call_sid": call.sid, "status": call.status}

        except Exception as e:
            logger.error(f"Failed to hangup call: {str(e)}")
            raise


voice_service = VoiceService()
