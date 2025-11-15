import pytest

from app.services.memory import ConversationMemory, InMemoryBackend


class TestInMemoryBackend:
    def test_create_session(self):
        backend = InMemoryBackend(max_history=10)
        session_id = backend.create_session(user_id="test_user")

        assert session_id is not None
        assert backend.session_exists(session_id)

        metadata = backend.get_session_metadata(session_id)
        assert metadata["user_id"] == "test_user"
        assert metadata["message_count"] == 0

    def test_add_message(self):
        backend = InMemoryBackend(max_history=10)
        session_id = backend.create_session()

        backend.add_message(session_id, "user", "Hello", {"test": "metadata"})

        messages = backend.get_messages(session_id)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[0]["metadata"]["test"] == "metadata"

    def test_message_limit(self):
        backend = InMemoryBackend(max_history=2)
        session_id = backend.create_session()

        for i in range(10):
            backend.add_message(session_id, "user", f"Message {i}")

        messages = backend.get_messages(session_id)
        assert len(messages) == 4

    def test_get_messages_with_limit(self):
        backend = InMemoryBackend(max_history=10)
        session_id = backend.create_session()

        for i in range(5):
            backend.add_message(session_id, "user", f"Message {i}")

        recent_messages = backend.get_messages(session_id, limit=2)
        assert len(recent_messages) == 2
        assert recent_messages[0]["content"] == "Message 3"
        assert recent_messages[1]["content"] == "Message 4"

    def test_clear_session(self):
        backend = InMemoryBackend(max_history=10)
        session_id = backend.create_session()
        backend.add_message(session_id, "user", "Test")

        assert backend.session_exists(session_id)
        result = backend.clear_session(session_id)

        assert result is True
        assert not backend.session_exists(session_id)

    def test_get_all_sessions(self):
        backend = InMemoryBackend(max_history=10)
        session1 = backend.create_session()
        session2 = backend.create_session()

        sessions = backend.get_all_sessions()
        assert len(sessions) == 2
        assert session1 in sessions
        assert session2 in sessions


class TestConversationMemory:
    def test_get_recent_context(self):
        memory = ConversationMemory()
        session_id = memory.create_session()

        memory.add_message(session_id, "user", "What is term insurance?")
        memory.add_message(session_id, "assistant", "Term insurance is...")

        context = memory.get_recent_context(session_id)

        assert "User: What is term insurance?" in context
        assert "Assistant: Term insurance is..." in context

    def test_conversation_history_format(self):
        memory = ConversationMemory()
        session_id = memory.create_session()

        memory.add_message(session_id, "user", "Hello")
        memory.add_message(session_id, "assistant", "Hi there")

        history = memory.get_conversation_history(session_id, format_for_llm=True)

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert "timestamp" not in history[0]
