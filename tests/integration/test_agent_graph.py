import pytest

from app.agents.graph import LifeInsuranceAgent
from app.agents.services import ToolExecutor
from app.services.memory import ConversationMemory


class TestAgentGraph:
    def test_intent_classification(self, mock_llm_response, mock_rag_service):
        agent = LifeInsuranceAgent()

        state = {
            "messages": [],
            "question": "What types of life insurance are available?",
            "intent": "",
            "context": "",
            "needs_clarification": False,
            "tool_results": {},
            "final_answer": "",
            "session_id": "test-session",
        }

        result = agent._analyze_intent(state)

        assert "intent" in result
        assert result["intent"] in [
            "POLICY_TYPES",
            "ELIGIBILITY",
            "CLAIMS",
            "PREMIUMS",
            "COVERAGE",
            "GENERAL",
        ]

    def test_should_use_tools_for_premium(self):
        agent = LifeInsuranceAgent()

        state = {
            "question": "Calculate premium for 35 year old",
            "intent": "PREMIUMS",
        }

        result = agent._should_use_tools(state)
        assert result == "use_tools"

    def test_should_use_tools_for_comparison(self):
        agent = LifeInsuranceAgent()

        state = {
            "question": "Compare term and whole life insurance",
            "intent": "POLICY_TYPES",
        }

        result = agent._should_use_tools(state)
        assert result == "use_tools"

    def test_should_skip_tools_for_general(self):
        agent = LifeInsuranceAgent()

        state = {
            "question": "What is life insurance?",
            "intent": "GENERAL",
        }

        result = agent._should_use_tools(state)
        assert result == "generate_answer"

    def test_extract_number_age(self):
        tool_executor = ToolExecutor()

        age = tool_executor._extract_number("I am 35 years old", "age", 30)
        assert age == 35

        age = tool_executor._extract_number("age 42", "age", 30)
        assert age == 42

    def test_extract_number_coverage(self):
        tool_executor = ToolExecutor()

        coverage = tool_executor._extract_number("$500k coverage", "coverage", 250000)
        assert coverage == 500000

        coverage = tool_executor._extract_number("$1000k coverage", "coverage", 250000)
        assert coverage == 1000000

        coverage = tool_executor._extract_number("$250000 coverage", "coverage", 100000)
        assert coverage == 250000

    def test_extract_number_default(self):
        tool_executor = ToolExecutor()

        age = tool_executor._extract_number("no age mentioned", "age", 30)
        assert age == 30

    def test_process_message_success(self, mock_llm_response, mock_rag_service):
        memory = ConversationMemory()
        session_id = memory.create_session()

        agent = LifeInsuranceAgent()
        result = agent.process_message("What is term life insurance?", session_id)

        assert result["success"] is True
        assert "answer" in result
        assert "agent_reasoning" in result
