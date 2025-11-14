import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class ConversationMemory:
    def __init__(self, max_history: int = None):
        self.sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        self.max_history = max_history or settings.memory_max_history

    def create_session(self, user_id: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        self.session_metadata[session_id] = {
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "user_id": user_id,
            "message_count": 0,
        }
        logger.info(f"Created new session: {session_id}")
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found, creating new one")
            self.sessions[session_id] = []
            self.session_metadata[session_id] = {
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "user_id": None,
                "message_count": 0,
            }

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        self.sessions[session_id].append(message)

        if len(self.sessions[session_id]) > self.max_history * 2:
            self.sessions[session_id] = self.sessions[session_id][
                -self.max_history * 2 :
            ]

        self.session_metadata[session_id]["updated_at"] = datetime.utcnow()
        self.session_metadata[session_id]["message_count"] += 1

        logger.debug(f"Added {role} message to session {session_id}")

    def get_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found")
            return []

        messages = self.sessions[session_id]

        if limit:
            return messages[-limit:]

        return messages

    def get_conversation_history(
        self, session_id: str, format_for_llm: bool = False
    ) -> List[Dict[str, str]]:
        messages = self.get_messages(session_id)

        if format_for_llm:
            return [
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ]

        return messages

    def clear_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            if session_id in self.session_metadata:
                del self.session_metadata[session_id]
            logger.info(f"Cleared session {session_id}")
            return True

        logger.warning(f"Session {session_id} not found")
        return False

    def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.session_metadata.get(session_id)

    def session_exists(self, session_id: str) -> bool:
        return session_id in self.sessions

    def get_all_sessions(self) -> List[str]:
        return list(self.sessions.keys())

    def get_recent_context(self, session_id: str, num_messages: int = None) -> str:
        num_messages = num_messages or settings.memory_context_messages
        messages = self.get_messages(session_id, limit=num_messages)

        if not messages:
            return "No previous conversation history."

        context_parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)


memory_service = ConversationMemory()
