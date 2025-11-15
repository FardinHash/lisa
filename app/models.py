from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    message: str
    sources: Optional[List[str]] = None
    agent_reasoning: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionCreate(BaseModel):
    user_id: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    message_count: int = 0


class SessionHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    status: str
    environment: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
