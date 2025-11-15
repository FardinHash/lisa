import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from app.agents.graph import agent
from app.models import (
    ChatRequest,
    ChatResponse,
    MessageRole,
    SessionCreate,
    SessionHistory,
    SessionResponse,
)
from app.services.memory import memory_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post(
    "/session", response_model=SessionResponse, status_code=status.HTTP_201_CREATED
)
async def create_session(session_data: SessionCreate):
    """
    Create a new chat session.
    """
    try:
        session_id = memory_service.create_session(user_id=session_data.user_id)

        return SessionResponse(
            session_id=session_id, created_at=datetime.utcnow(), message_count=0
        )
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session",
        )


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    Send a message and receive a response from the life insurance assistant.
    """
    try:
        if not memory_service.session_exists(request.session_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {request.session_id} not found. Please create a session first.",
            )

        memory_service.add_message(
            session_id=request.session_id,
            role=MessageRole.USER.value,
            content=request.message,
        )

        result = agent.process_message(
            message=request.message, session_id=request.session_id
        )

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process message",
            )

        answer = result["answer"]

        memory_service.add_message(
            session_id=request.session_id,
            role=MessageRole.ASSISTANT.value,
            content=answer,
            metadata={
                "sources": result.get("sources", []),
                "reasoning": result.get("agent_reasoning", ""),
            },
        )

        return ChatResponse(
            session_id=request.session_id,
            message=answer,
            sources=result.get("sources"),
            agent_reasoning=result.get("agent_reasoning"),
            timestamp=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}",
        )


@router.get("/session/{session_id}", response_model=SessionHistory)
async def get_session_history(session_id: str):
    """
    Retrieve the conversation history for a session.
    """
    try:
        if not memory_service.session_exists(session_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        messages = memory_service.get_conversation_history(session_id)
        metadata = memory_service.get_session_metadata(session_id)

        from app.models import ChatMessage

        chat_messages = [
            ChatMessage(
                role=MessageRole(msg["role"]),
                content=msg["content"],
                timestamp=datetime.fromisoformat(msg["timestamp"]),
                metadata=msg.get("metadata"),
            )
            for msg in messages
        ]

        return SessionHistory(
            session_id=session_id,
            messages=chat_messages,
            created_at=metadata["created_at"],
            updated_at=metadata["updated_at"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session history",
        )


@router.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    """
    Delete a chat session and its history.
    """
    try:
        success = memory_service.clear_session(session_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session",
        )


@router.get("/sessions")
async def list_sessions():
    """
    List all active sessions.
    """
    try:
        sessions = memory_service.get_all_sessions()

        session_list = []
        for session_id in sessions:
            metadata = memory_service.get_session_metadata(session_id)
            if metadata:
                session_list.append(
                    {
                        "session_id": session_id,
                        "created_at": metadata["created_at"],
                        "updated_at": metadata["updated_at"],
                        "message_count": metadata["message_count"],
                    }
                )

        return {"sessions": session_list, "total": len(session_list)}

    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list sessions",
        )
