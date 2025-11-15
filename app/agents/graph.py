import logging
import re
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agents.prompts import (ANSWER_GENERATION_PROMPT, CLARIFICATION_PROMPT,
                                INTENT_CLASSIFIER_PROMPT, SYSTEM_PROMPT)
from app.agents.tools import (calculate_premium_estimate, check_eligibility,
                              get_policy_comparison, search_knowledge_base)
from app.config import settings
from app.services.llm import llm_service
from app.services.memory import memory_service

logger = logging.getLogger(__name__)

VALID_INTENTS = [
    "POLICY_TYPES",
    "ELIGIBILITY",
    "CLAIMS",
    "PREMIUMS",
    "COVERAGE",
    "GENERAL",
]


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
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("analyze_intent", self._analyze_intent)
        workflow.add_node("retrieve_information", self._retrieve_information)
        workflow.add_node("check_clarification", self._check_clarification)
        workflow.add_node("use_tools", self._use_tools)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.set_entry_point("analyze_intent")

        workflow.add_edge("analyze_intent", "retrieve_information")
        workflow.add_edge("retrieve_information", "check_clarification")

        workflow.add_conditional_edges(
            "check_clarification",
            self._should_use_tools,
            {"use_tools": "use_tools", "generate_answer": "generate_answer"},
        )

        workflow.add_edge("use_tools", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def _analyze_intent(self, state: AgentState) -> AgentState:
        logger.info("Analyzing user intent...")

        question = state["question"]

        try:
            prompt = INTENT_CLASSIFIER_PROMPT.format(question=question)
            intent = llm_service.invoke([{"role": "user", "content": prompt}])
            intent = intent.strip().upper()

            if intent not in VALID_INTENTS:
                intent = "GENERAL"

            logger.info(f"Classified intent: {intent}")
            state["intent"] = intent

        except Exception as e:
            logger.error(f"Error analyzing intent: {str(e)}")
            state["intent"] = "GENERAL"

        return state

    def _retrieve_information(self, state: AgentState) -> AgentState:
        logger.info("Retrieving relevant information...")

        question = state["question"]
        intent = state.get("intent", "GENERAL")

        try:
            search_queries = [question]

            intent_queries = {
                "POLICY_TYPES": "types of life insurance policies",
                "ELIGIBILITY": "life insurance eligibility requirements",
                "CLAIMS": "life insurance claims process",
                "PREMIUMS": "life insurance premium costs",
                "COVERAGE": "life insurance coverage amounts",
            }

            if intent in intent_queries:
                search_queries.append(intent_queries[intent])

            all_context = []
            for query in search_queries[:2]:
                result = search_knowledge_base(query, k=settings.agent_search_k)
                if result["success"]:
                    all_context.append(result["context"])

            state["context"] = (
                "\n\n".join(all_context)
                if all_context
                else "No specific information found."
            )
            logger.info(f"Retrieved {len(all_context)} context sections")

        except Exception as e:
            logger.error(f"Error retrieving information: {str(e)}")
            state["context"] = "Error retrieving information."

        return state

    def _check_clarification(self, state: AgentState) -> AgentState:
        logger.info("Checking if clarification needed...")

        question = state["question"]
        context = state.get("context", "")

        try:
            prompt = CLARIFICATION_PROMPT.format(
                question=question, context=context[:500]
            )

            response = llm_service.invoke([{"role": "user", "content": prompt}])

            if "CLEAR" in response.upper():
                state["needs_clarification"] = False
                logger.info("Question is clear, proceeding...")
            else:
                state["needs_clarification"] = True
                logger.info("Question may need clarification")

        except Exception as e:
            logger.error(f"Error checking clarification: {str(e)}")
            state["needs_clarification"] = False

        return state

    def _should_use_tools(self, state: AgentState) -> str:
        question = state["question"].lower()
        intent = state.get("intent", "GENERAL")

        calculate_keywords = [
            "calculate",
            "estimate",
            "cost",
            "price",
            "premium",
            "how much",
        ]
        eligibility_keywords = [
            "eligible",
            "qualify",
            "can i get",
            "approved",
            "health",
        ]
        compare_keywords = ["compare", "difference", "versus", "vs", "better"]

        if intent == "PREMIUMS" or any(kw in question for kw in calculate_keywords):
            return "use_tools"

        if intent == "ELIGIBILITY" or any(
            kw in question for kw in eligibility_keywords
        ):
            return "use_tools"

        if any(kw in question for kw in compare_keywords):
            return "use_tools"

        return "generate_answer"

    def _use_tools(self, state: AgentState) -> AgentState:
        logger.info("Using specialized tools...")

        question = state["question"].lower()
        tool_results = {}

        try:
            if "calculate" in question or "estimate" in question or "cost" in question:
                age = self._extract_number(
                    question, context="age", default=settings.tool_default_age
                )
                coverage = self._extract_number(
                    question, context="coverage", default=settings.tool_default_coverage
                )
                term = self._extract_number(
                    question, context="term", default=settings.tool_default_term
                )
                is_smoker = "smok" in question

                result = calculate_premium_estimate(
                    age=age,
                    coverage_amount=coverage,
                    term_length=term,
                    is_smoker=is_smoker,
                )
                tool_results["premium_estimate"] = result
                logger.info("Premium calculation completed")

            if "eligible" in question or "qualify" in question:
                age = self._extract_number(
                    question, context="age", default=settings.tool_default_age
                )
                smoker = "smok" in question

                occupation_match = re.search(
                    r"(?:work in|occupation|job)(?:\s+(?:is|as))?\s+(\w+)", question
                )
                occupation = (
                    occupation_match.group(1) if occupation_match else "standard"
                )

                health_conditions = []
                common_terms = [
                    "diabetes",
                    "high blood pressure",
                    "cholesterol",
                    "cancer",
                    "heart disease",
                    "asthma",
                ]
                for term in common_terms:
                    if term in question:
                        health_conditions.append(term)

                result = check_eligibility(
                    age=age,
                    health_conditions=health_conditions,
                    smoker=smoker,
                    occupation=occupation,
                )
                tool_results["eligibility"] = result
                logger.info("Eligibility check completed")

            if "compare" in question:
                policy_types = []
                known_types = ["term", "whole", "universal", "variable"]
                for ptype in known_types:
                    if ptype in question:
                        policy_types.append(ptype)

                if len(policy_types) >= 2:
                    result = get_policy_comparison(policy_types)
                    tool_results["comparison"] = result
                    logger.info(f"Policy comparison completed for: {policy_types}")

            state["tool_results"] = tool_results

        except Exception as e:
            logger.error(f"Error using tools: {str(e)}")
            state["tool_results"] = {}

        return state

    def _extract_number(self, text: str, context: str, default: int) -> int:
        pattern_map = {
            "age": [r"(\d+)\s*year", r"age\s*(\d+)", r"i'm\s*(\d+)", r"i am\s*(\d+)"],
            "coverage": [r"\$?(\d+)k", r"\$?(\d+),?\d*,?\d*\s*coverage", r"\$(\d+)"],
            "term": [r"(\d+)\s*year.*term", r"(\d+)-year"],
        }

        patterns = pattern_map.get(context)
        if not patterns:
            return default

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                num = int(match.group(1))
                if context == "coverage" and "k" in text.lower():
                    num *= 1000
                return num

        return default

    def _generate_answer(self, state: AgentState) -> AgentState:
        logger.info("Generating final answer...")

        question = state["question"]
        context = state.get("context", "")
        tool_results = state.get("tool_results", {})
        session_id = state.get("session_id", "")

        try:
            conversation_history = ""
            if session_id:
                conversation_history = memory_service.get_recent_context(session_id)

            tool_context = ""
            if tool_results:
                tool_context = "\n\nTool Results:\n"
                for tool_name, result in tool_results.items():
                    tool_context += f"\n{tool_name.upper()}:\n{str(result)}\n"

            full_context = f"{context}\n{tool_context}"

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ANSWER_GENERATION_PROMPT.format(
                        conversation_history=conversation_history
                        or "No previous conversation",
                        context=full_context,
                        question=question,
                    ),
                },
            ]

            answer = llm_service.invoke(messages)
            state["final_answer"] = answer
            logger.info("Answer generated successfully")

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            state["final_answer"] = (
                "I apologize, but I encountered an error processing your question. Please try rephrasing or contact support."
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
