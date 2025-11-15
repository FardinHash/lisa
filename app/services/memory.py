import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class MemoryBackend(ABC):
    @abstractmethod
    def create_session(self, user_id: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    @abstractmethod
    def get_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def clear_session(self, session_id: str) -> bool:
        pass

    @abstractmethod
    def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def session_exists(self, session_id: str) -> bool:
        pass

    @abstractmethod
    def get_all_sessions(self) -> List[str]:
        pass


class InMemoryBackend(MemoryBackend):
    def __init__(self, max_history: int):
        self.sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        self.max_history = max_history

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
            return []

        messages = self.sessions[session_id]
        return messages[-limit:] if limit else messages

    def clear_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            if session_id in self.session_metadata:
                del self.session_metadata[session_id]
            logger.info(f"Cleared session {session_id}")
            return True
        return False

    def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.session_metadata.get(session_id)

    def session_exists(self, session_id: str) -> bool:
        return session_id in self.sessions

    def get_all_sessions(self) -> List[str]:
        return list(self.sessions.keys())


class DatabaseBackend(MemoryBackend):
    def __init__(self, max_history: int):
        from app.database import MessageModel, SessionLocal, SessionModel, init_db

        self.max_history = max_history
        self.SessionLocal = SessionLocal
        self.SessionModel = SessionModel
        self.MessageModel = MessageModel
        init_db()

    def create_session(self, user_id: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        db = self.SessionLocal()
        try:
            session = self.SessionModel(
                id=session_id,
                user_id=user_id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                message_count=0,
            )
            db.add(session)
            db.commit()
            logger.info(f"Created new session: {session_id}")
            return session_id
        finally:
            db.close()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        db = self.SessionLocal()
        try:
            session = db.query(self.SessionModel).filter_by(id=session_id).first()
            if not session:
                logger.warning(f"Session {session_id} not found, creating new one")
                session = self.SessionModel(
                    id=session_id,
                    user_id=None,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    message_count=0,
                )
                db.add(session)

            message = self.MessageModel(
                session_id=session_id,
                role=role,
                content=content,
                timestamp=datetime.utcnow(),
                metadata_json=json.dumps(metadata or {}),
            )
            db.add(message)

            session.updated_at = datetime.utcnow()
            session.message_count += 1

            db.commit()
            logger.debug(f"Added {role} message to session {session_id}")
        finally:
            db.close()

    def get_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        db = self.SessionLocal()
        try:
            query = (
                db.query(self.MessageModel)
                .filter_by(session_id=session_id)
                .order_by(self.MessageModel.id)
            )

            if limit:
                total = query.count()
                offset = max(0, total - limit)
                messages_db = query.offset(offset).limit(limit).all()
            else:
                messages_db = query.all()

            messages = []
            for msg in messages_db:
                messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "metadata": (
                            json.loads(msg.metadata_json) if msg.metadata_json else {}
                        ),
                    }
                )
            return messages
        finally:
            db.close()

    def clear_session(self, session_id: str) -> bool:
        db = self.SessionLocal()
        try:
            session = db.query(self.SessionModel).filter_by(id=session_id).first()
            if session:
                db.query(self.MessageModel).filter_by(session_id=session_id).delete()
                db.delete(session)
                db.commit()
                logger.info(f"Cleared session {session_id}")
                return True
            return False
        finally:
            db.close()

    def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        db = self.SessionLocal()
        try:
            session = db.query(self.SessionModel).filter_by(id=session_id).first()
            if session:
                return {
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "user_id": session.user_id,
                    "message_count": session.message_count,
                }
            return None
        finally:
            db.close()

    def session_exists(self, session_id: str) -> bool:
        db = self.SessionLocal()
        try:
            session = db.query(self.SessionModel).filter_by(id=session_id).first()
            return session is not None
        finally:
            db.close()

    def get_all_sessions(self) -> List[str]:
        db = self.SessionLocal()
        try:
            sessions = db.query(self.SessionModel.id).all()
            return [s.id for s in sessions]
        finally:
            db.close()


class ConversationMemory:
    def __init__(self, max_history: int = None):
        self.max_history = max_history or settings.memory_max_history

        if settings.environment == "local":
            self.backend = InMemoryBackend(self.max_history)
            logger.info("Using in-memory backend for session storage")
        else:
            self.backend = DatabaseBackend(self.max_history)
            logger.info("Using database backend for session storage")

    def create_session(self, user_id: Optional[str] = None) -> str:
        return self.backend.create_session(user_id)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.backend.add_message(session_id, role, content, metadata)

    def get_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return self.backend.get_messages(session_id, limit)

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
        return self.backend.clear_session(session_id)

    def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.backend.get_session_metadata(session_id)

    def session_exists(self, session_id: str) -> bool:
        return self.backend.session_exists(session_id)

    def get_all_sessions(self) -> List[str]:
        return self.backend.get_all_sessions()

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
