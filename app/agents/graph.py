import logging
import re
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agents.services import (
    ContextRetriever,
    IntentAnalyzer,
    ResponseGenerator,
    ToolExecutor,
    ToolSelector,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    intent: str
    context: str
    needs_clarification: bool
    tool_results: dict
    final_answer: str
    session_id: str


class LifeInsuranceAgent:
    def __init__(self) -> None:
        self.intent_analyzer = IntentAnalyzer()
        self.context_retriever = ContextRetriever()
        self.tool_selector = ToolSelector()
        self.tool_executor = ToolExecutor()
        self.response_generator = ResponseGenerator()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("analyze_intent", self._analyze_intent)
        workflow.add_node("retrieve_information", self._retrieve_information)
        workflow.add_node("use_tools", self._use_tools)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.set_entry_point("analyze_intent")

        workflow.add_edge("analyze_intent", "retrieve_information")

        workflow.add_conditional_edges(
            "retrieve_information",
            self._should_use_tools,
            {"use_tools": "use_tools", "generate_answer": "generate_answer"},
        )

        workflow.add_edge("use_tools", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def _analyze_intent(self, state: AgentState) -> AgentState:
        question = state["question"]
        state["intent"] = self.intent_analyzer.analyze(question)
        return state

    def _retrieve_information(self, state: AgentState) -> AgentState:
        question = state["question"]
        intent = state.get("intent", "GENERAL")
        session_id = state.get("session_id", "")

        state["context"] = self.context_retriever.retrieve(question, intent, session_id)
        return state

    def _should_use_tools(self, state: AgentState) -> str:
        question = state["question"]
        intent = state.get("intent", "GENERAL")

        should_use = self.tool_selector.should_use_tools(question, intent)
        return "use_tools" if should_use else "generate_answer"

    def _use_tools(self, state: AgentState) -> AgentState:
        question = state["question"]
        state["tool_results"] = self.tool_executor.execute(question)
        return state

    def _generate_answer(self, state: AgentState) -> AgentState:
        question = state["question"]
        context = state.get("context", "")
        tool_results = state.get("tool_results", {})
        session_id = state.get("session_id", "")

        state["final_answer"] = self.response_generator.generate(
            question, context, tool_results, session_id
        )
        return state

    def process_message(self, message: str, session_id: str) -> dict:
        try:
            logger.info(f"Processing message for session {session_id}")

            initial_state = {
                "messages": [HumanMessage(content=message)],
                "question": message,
                "intent": "",
                "context": "",
                "needs_clarification": False,
                "tool_results": {},
                "final_answer": "",
                "session_id": session_id,
            }

            result = self.graph.invoke(initial_state)

            answer = result.get(
                "final_answer", "I'm sorry, I couldn't generate a response."
            )

            sources = []
            if result.get("context"):
                context_sources = re.findall(
                    r"\[Source \d+: ([^\]]+)\]", result.get("context", "")
                )
                sources = list(set(context_sources))

            agent_reasoning = f"Intent: {result.get('intent', 'Unknown')}"
            if result.get("tool_results"):
                agent_reasoning += (
                    f" | Tools Used: {', '.join(result['tool_results'].keys())}"
                )

            return {
                "answer": answer,
                "sources": sources,
                "agent_reasoning": agent_reasoning,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error. Please try again.",
                "sources": [],
                "agent_reasoning": f"Error: {str(e)}",
                "success": False,
            }


agent = LifeInsuranceAgent()
