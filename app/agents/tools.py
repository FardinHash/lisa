import logging
from typing import Any, Dict, Optional

from app.config import settings
from app.services.llm import llm_service
from app.services.rag import rag_service

logger = logging.getLogger(__name__)


def search_knowledge_base(query: str, k: int = 4) -> Dict[str, Any]:
    """
    Search the life insurance knowledge base for relevant information.
    """
    try:
        results = rag_service.search(query, k=k)

        if not results:
            return {
                "success": False,
                "context": "No relevant information found.",
                "sources": [],
                "num_results": 0,
            }

        context = rag_service.get_relevant_context(query, k=k)
        sources = [result["source"] for result in results]

        return {
            "success": True,
            "context": context,
            "sources": sources,
            "num_results": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        return {
            "success": False,
            "context": "Error searching knowledge base.",
            "sources": [],
            "num_results": 0,
            "error": str(e),
        }


def calculate_premium_estimate(
    age: int,
    coverage_amount: int,
    term_length: int,
    is_smoker: bool = False,
    health_rating: str = "standard",
) -> Dict[str, Any]:
    """
    Calculate estimated life insurance premium using knowledge base criteria.
    """
    try:
        criteria_query = f"premium rating factors for age {age}, term {term_length} years, health rating {health_rating}, smoker status {is_smoker}"
        criteria_result = search_knowledge_base(criteria_query, k=3)

        prompt = f"""Based on the following life insurance rating criteria, calculate an estimated premium.

Premium Rating Criteria:
{criteria_result.get('context', '')}

Calculate for:
- Age: {age} years
- Coverage Amount: ${coverage_amount:,}
- Term Length: {term_length} years
- Smoker: {'Yes' if is_smoker else 'No'}
- Health Rating: {health_rating}

Use a base rate of ${settings.premium_base_rate} per $1,000 of coverage per month.
Apply the rating factors from the criteria above.

Provide the calculation in this JSON format:
{{
    "monthly_premium": <calculated amount>,
    "annual_premium": <monthly * 12>,
    "total_term_cost": <annual * term_length>,
    "explanation": "<brief explanation of factors applied>"
}}"""

        response = llm_service.invoke([{"role": "user", "content": prompt}])

        import json
        import re

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            calculation = json.loads(json_match.group())

            return {
                "success": True,
                "monthly_premium": round(calculation.get("monthly_premium", 0), 2),
                "annual_premium": round(calculation.get("annual_premium", 0), 2),
                "total_term_cost": round(calculation.get("total_term_cost", 0), 2),
                "factors": {
                    "age": age,
                    "coverage_amount": f"${coverage_amount:,}",
                    "term_length": f"{term_length} years",
                    "smoker_status": "Smoker" if is_smoker else "Non-smoker",
                    "health_rating": health_rating,
                },
                "explanation": calculation.get("explanation", ""),
                "note": "This is an estimate based on industry standards. Actual premiums may vary.",
            }

        base_rate_per_1000 = settings.premium_base_rate
        monthly_premium = (
            (coverage_amount / 1000) * base_rate_per_1000 * (2.5 if is_smoker else 1.0)
        )

        return {
            "success": True,
            "monthly_premium": round(monthly_premium, 2),
            "annual_premium": round(monthly_premium * 12, 2),
            "total_term_cost": round(monthly_premium * 12 * term_length, 2),
            "factors": {
                "age": age,
                "coverage_amount": f"${coverage_amount:,}",
                "term_length": f"{term_length} years",
                "smoker_status": "Smoker" if is_smoker else "Non-smoker",
                "health_rating": health_rating,
            },
            "note": "This is a basic estimate. Actual premiums may vary based on detailed underwriting.",
        }

    except Exception as e:
        logger.error(f"Error calculating premium: {str(e)}")
        return {"success": False, "error": str(e)}


def check_eligibility(
    age: int,
    health_conditions: list = None,
    smoker: bool = False,
    occupation: str = "standard",
    coverage_amount: int = None,
) -> Dict[str, Any]:
    """
    Check life insurance eligibility using LLM reasoning on knowledge base criteria.
    """
    try:
        health_conditions = health_conditions or []
        coverage_amount = coverage_amount or settings.tool_default_coverage

        criteria_query = "life insurance eligibility criteria risk assessment health conditions age requirements"
        criteria_result = search_knowledge_base(criteria_query, k=4)

        conditions_str = (
            ", ".join(health_conditions) if health_conditions else "none reported"
        )

        prompt = f"""As a life insurance underwriting expert, assess the eligibility for this applicant.

Eligibility Criteria from Knowledge Base:
{criteria_result.get('context', '')}

Applicant Profile:
- Age: {age} years
- Health Conditions: {conditions_str}
- Smoker: {'Yes' if smoker else 'No'}
- Occupation: {occupation}
- Desired Coverage: ${coverage_amount:,}

Provide your assessment in JSON format:
{{
    "eligibility": "<Good/Moderate/Challenging>",
    "likely_approved": <true/false>,
    "issues": ["<list any concerns>"],
    "recommendations": ["<list recommendations>"],
    "reasoning": "<explain your assessment>"
}}"""

        response = llm_service.invoke([{"role": "user", "content": prompt}])

        import json
        import re

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            assessment = json.loads(json_match.group())

            return {
                "success": True,
                "eligibility": assessment.get("eligibility", "Moderate"),
                "likely_approved": assessment.get("likely_approved", True),
                "issues": assessment.get("issues", []),
                "recommendations": assessment.get("recommendations", []),
                "reasoning": assessment.get("reasoning", ""),
                "suggested_actions": [
                    "Get quotes from multiple insurers",
                    "Prepare medical records and documentation",
                    "Consider working with an independent insurance broker",
                    "Review different policy types for your situation",
                ],
            }

        return {
            "success": True,
            "eligibility": "Moderate",
            "likely_approved": True,
            "issues": [],
            "recommendations": [
                "Consult with an insurance agent for detailed assessment"
            ],
            "suggested_actions": [
                "Get quotes from multiple insurers",
                "Prepare medical records and documentation",
            ],
        }

    except Exception as e:
        logger.error(f"Error checking eligibility: {str(e)}")
        return {"success": False, "error": str(e)}


def get_policy_comparison(policy_types: list) -> Dict[str, Any]:
    """
    Get a comparison of different policy types from the knowledge base.
    """
    try:
        query = f"Compare {', '.join(policy_types)} life insurance policies including features, benefits, and costs"
        results = search_knowledge_base(query, k=settings.agent_comparison_k)

        return {
            "success": True,
            "comparison": results.get("context", ""),
            "sources": results.get("sources", []),
        }

    except Exception as e:
        logger.error(f"Error getting policy comparison: {str(e)}")
        return {"success": False, "error": str(e)}
