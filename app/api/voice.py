import base64
import json
import logging
import os
from typing import Dict

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.agents.graph import agent
from app.config import settings
from app.models import (
    CallHangupResponse,
    CallStatusResponse,
    OutboundCallRequest,
    OutboundCallResponse,
)
from app.services.memory import memory_service
from app.services.voice import voice_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])


class VoiceConnection:
    def __init__(self, websocket: WebSocket, session_id: str):
        self.websocket = websocket
        self.session_id = session_id
        self.stream_sid = ""
        self.audio_buffer = bytearray()

    async def receive_audio_chunk(self) -> Dict:
        data = await self.websocket.receive_json()
        return data

    async def get_complete_audio(self) -> bytes:
        audio_data = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        return audio_data

    async def send_audio(self, audio_data: bytes):
        encoded = base64.b64encode(audio_data).decode("utf-8")

        await self.websocket.send_json(
            {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": encoded},
            }
        )

    async def send_mark(self, mark_name: str):
        await self.websocket.send_json({"event": "mark", "mark": {"name": mark_name}})


@router.websocket("/stream")
async def voice_stream(websocket: WebSocket):
    if not settings.voice_enabled:
        await websocket.close(code=1008, reason="Voice not enabled")
        return

    await websocket.accept()

    connection = None
    session_id = None

    try:
        start_data = await websocket.receive_json()

        if start_data.get("event") == "start":
            stream_sid = start_data["start"]["streamSid"]
            call_sid = start_data["start"]["callSid"]

            session_id = memory_service.create_session(user_id=f"voice_{call_sid}")

            connection = VoiceConnection(websocket, session_id)
            connection.stream_sid = stream_sid

            logger.info(f"Voice call started: {call_sid}, session: {session_id}")

            greeting = (
                "Hello! I'm your life insurance assistant. How can I help you today?"
            )
            await send_voice_response(connection, greeting, session_id)

        while True:
            data = await connection.receive_audio_chunk()

            if data.get("event") == "media":
                payload = data["media"]["payload"]
                audio_chunk = base64.b64decode(payload)
                connection.audio_buffer.extend(audio_chunk)

            elif data.get("event") == "stop":
                audio_data = await connection.get_complete_audio()

                if len(audio_data) > 0:
                    user_text = await voice_service.transcribe_audio(
                        audio_data, audio_format="mulaw"
                    )

                    logger.info(f"User said: {user_text}")

                    memory_service.add_message(
                        session_id=session_id, role="user", content=user_text
                    )

                    response = agent.process_message(user_text, session_id)
                    answer = response["answer"]

                    await send_voice_response(connection, answer, session_id)

    except WebSocketDisconnect:
        logger.info(f"Voice call ended: {session_id}")
    except Exception as e:
        logger.error(f"Voice stream error: {str(e)}")
    finally:
        if websocket.client_state.value == 1:
            await websocket.close()


async def send_voice_response(connection: VoiceConnection, text: str, session_id: str):
    try:
        audio_chunks = []
        async for chunk in voice_service.synthesize_speech(text):
            audio_chunks.append(chunk)

        full_audio = b"".join(audio_chunks)
        await connection.send_audio(full_audio)

        await connection.send_mark("end_of_response")

        memory_service.add_message(
            session_id=session_id, role="assistant", content=text
        )

    except Exception as e:
        logger.error(f"Error sending voice response: {str(e)}")
        raise


@router.post("/twilio/answer")
async def twilio_answer():
    if not settings.voice_enabled:
        return Response(
            content="Voice not enabled", status_code=503, media_type="text/plain"
        )

    webhook_url = f"wss://{settings.api_host}/api/v1/voice/stream"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{webhook_url}"/>
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@router.get("/status")
async def voice_status():
    return {
        "voice_enabled": settings.voice_enabled,
        "stt_model": settings.voice_stt_model,
        "tts_model": settings.voice_tts_model,
        "tts_voice": settings.voice_tts_voice,
    }


@router.post("/outbound/initiate", response_model=OutboundCallResponse)
async def initiate_outbound_call(request: OutboundCallRequest):
    if not settings.voice_enabled:
        raise HTTPException(status_code=503, detail="Voice agent is disabled")

    if not voice_service.twilio_client:
        raise HTTPException(
            status_code=500, detail="Twilio client not configured properly"
        )

    try:
        api_host = os.getenv("API_HOST", settings.api_host)
        callback_url = f"https://{api_host}/api/v1/voice/outbound/connect"

        result = voice_service.initiate_outbound_call(
            to_number=request.to_number,
            callback_url=callback_url,
            initial_message=request.initial_message,
        )

        return OutboundCallResponse(
            call_sid=result["call_sid"],
            to=result["to"],
            from_number=result["from"],
            status=result["status"],
            message=f"Outbound call initiated to {request.to_number}",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to initiate outbound call: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initiate call")


@router.post("/outbound/connect")
async def outbound_call_connect():
    if not settings.voice_enabled:
        return Response(
            content="Voice not enabled", status_code=503, media_type="text/plain"
        )

    webhook_url = f"wss://{settings.api_host}/api/v1/voice/stream"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Hello! I'm your life insurance assistant calling from our agency. How can I help you today?</Say>
    <Connect>
        <Stream url="{webhook_url}"/>
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@router.post("/outbound/connect/status")
async def outbound_call_status_callback():
    logger.info("Outbound call status update received")
    return Response(content="OK", status_code=200)


@router.get("/call/{call_sid}/status", response_model=CallStatusResponse)
async def get_call_status(call_sid: str):
    if not voice_service.twilio_client:
        raise HTTPException(
            status_code=500, detail="Twilio client not configured properly"
        )

    try:
        result = voice_service.get_call_status(call_sid)
        return CallStatusResponse(**result)

    except Exception as e:
        logger.error(f"Failed to get call status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get call status")


@router.post("/call/{call_sid}/hangup", response_model=CallHangupResponse)
async def hangup_call(call_sid: str):
    if not voice_service.twilio_client:
        raise HTTPException(
            status_code=500, detail="Twilio client not configured properly"
        )

    try:
        result = voice_service.hangup_call(call_sid)
        return CallHangupResponse(
            call_sid=result["call_sid"],
            status=result["status"],
            message=f"Call {call_sid} terminated",
        )

    except Exception as e:
        logger.error(f"Failed to hangup call: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to terminate call")
