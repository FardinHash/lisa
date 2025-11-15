import logging
import re
from typing import Dict, List

from app.agents.prompts import (
    ANSWER_GENERATION_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    SYSTEM_PROMPT,
)
from app.agents.tools import (
    calculate_premium_estimate,
    check_eligibility,
    get_policy_comparison,
    search_knowledge_base,
)
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


class IntentAnalyzer:
    def analyze(self, question: str) -> str:
        logger.info("Analyzing user intent...")

        try:
            prompt = INTENT_CLASSIFIER_PROMPT.format(question=question)
            intent = llm_service.invoke(
                [{"role": "user", "content": prompt}],
                temperature=settings.intent_classification_temperature,
            )
            intent = intent.strip().upper()

            if intent not in VALID_INTENTS:
                intent = "GENERAL"

            logger.info(f"Classified intent: {intent}")
            return intent

        except Exception as e:
            logger.error(f"Error analyzing intent: {str(e)}")
            return "GENERAL"


class ContextRetriever:
    def retrieve(self, question: str, intent: str, session_id: str = None) -> str:
        logger.info("Retrieving relevant information...")

        try:
            search_query = question

            if session_id:
                conversation_history = memory_service.get_recent_context(
                    session_id, num_messages=2
                )
                if (
                    conversation_history
                    and conversation_history != "No previous conversation history."
                ):
                    search_query = (
                        f"{conversation_history}\n\nCurrent question: {question}"
                    )
                    logger.info("Enhanced search query with conversation context")

            intent_keywords = {
                "POLICY_TYPES": "policy types",
                "ELIGIBILITY": "eligibility",
                "CLAIMS": "claims",
                "PREMIUMS": "premiums",
                "COVERAGE": "coverage",
            }

            if intent in intent_keywords:
                search_query = f"{search_query} {intent_keywords[intent]}"

            result = search_knowledge_base(search_query, k=settings.agent_search_k)

            context = (
                result["context"]
                if result["success"]
                else "No specific information found."
            )
            logger.info("Retrieved context successfully")
            return context

        except Exception as e:
            logger.error(f"Error retrieving information: {str(e)}")
            return "Error retrieving information."


class ToolSelector:
    def should_use_tools(self, question: str, intent: str) -> bool:
        logger.info("Determining if tools are needed...")

        try:
            prompt = f"""Analyze this user question and determine if specialized tools are needed.

User Question: {question}
Detected Intent: {intent}

Available Tools:
1. Premium Calculator - For calculating insurance costs and premiums
2. Eligibility Checker - For checking if someone qualifies for insurance
3. Policy Comparator - For comparing different policy types

Respond with ONLY "YES" if tools are needed, or "NO" if the question can be answered from knowledge base alone.

Examples:
- "How much would insurance cost for a 35 year old?" -> YES (need calculator)
- "Can I get insurance if I have diabetes?" -> YES (need eligibility checker)
- "What is term life insurance?" -> NO (knowledge base sufficient)
- "Compare term and whole life" -> YES (need comparator)

Your answer (YES or NO):"""

            response = llm_service.invoke(
                [{"role": "user", "content": prompt}],
                temperature=settings.tool_selection_temperature,
            )

            should_use = "YES" in response.upper()
            logger.info(
                f"Tool selection decision: {'Use tools' if should_use else 'Skip tools'}"
            )
            return should_use

        except Exception as e:
            logger.error(f"Error in tool selection: {str(e)}")
            question_lower = question.lower()
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

            if intent == "PREMIUMS" or any(
                kw in question_lower for kw in calculate_keywords
            ):
                return True
            if intent == "ELIGIBILITY" or any(
                kw in question_lower for kw in eligibility_keywords
            ):
                return True
            if any(kw in question_lower for kw in compare_keywords):
                return True

            return False


class ToolExecutor:
    def execute(self, question: str) -> Dict:
        logger.info("Using specialized tools...")

        question_lower = question.lower()
        tool_results = {}

        try:
            if (
                "calculate" in question_lower
                or "estimate" in question_lower
                or "cost" in question_lower
            ):
                age = self._extract_number(
                    question_lower, context="age", default=settings.tool_default_age
                )
                coverage = self._extract_number(
                    question_lower,
                    context="coverage",
                    default=settings.tool_default_coverage,
                )
                term = self._extract_number(
                    question_lower, context="term", default=settings.tool_default_term
                )
                is_smoker = "smok" in question_lower

                result = calculate_premium_estimate(
                    age=age,
                    coverage_amount=coverage,
                    term_length=term,
                    is_smoker=is_smoker,
                )
                tool_results["premium_estimate"] = result
                logger.info("Premium calculation completed")

            if "eligible" in question_lower or "qualify" in question_lower:
                age = self._extract_number(
                    question_lower, context="age", default=settings.tool_default_age
                )
                smoker = "smok" in question_lower

                occupation_match = re.search(
                    r"(?:work in|occupation|job)(?:\s+(?:is|as))?\s+(\w+)",
                    question_lower,
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
                    if term in question_lower:
                        health_conditions.append(term)

                result = check_eligibility(
                    age=age,
                    health_conditions=health_conditions,
                    smoker=smoker,
                    occupation=occupation,
                )
                tool_results["eligibility"] = result
                logger.info("Eligibility check completed")

            if "compare" in question_lower:
                policy_types = []
                known_types = ["term", "whole", "universal", "variable"]
                for ptype in known_types:
                    if ptype in question_lower:
                        policy_types.append(ptype)

                if len(policy_types) >= 2:
                    result = get_policy_comparison(policy_types)
                    tool_results["comparison"] = result
                    logger.info(f"Policy comparison completed for: {policy_types}")

        except Exception as e:
            logger.error(f"Error using tools: {str(e)}")

        return tool_results

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
            match = re.search(pattern, text)
            if match:
                num = int(match.group(1))
                if context == "coverage" and "k" in text:
                    num *= 1000
                return num

        return default


class ResponseGenerator:
    def generate(
        self,
        question: str,
        context: str,
        tool_results: Dict,
        session_id: str = None,
    ) -> str:
        logger.info("Generating final answer...")

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
            logger.info("Answer generated successfully")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return (
                "I apologize, but I encountered an error processing your question. "
                "Please try rephrasing or contact support."
            )
