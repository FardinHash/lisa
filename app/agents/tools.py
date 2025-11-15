import logging
import re
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
    Calculate estimated life insurance premium using AI-powered analysis of rating criteria.
    """
    try:
        criteria_query = f"life insurance premium rating factors age {age} term {term_length} years smoker {is_smoker}"
        criteria_result = search_knowledge_base(criteria_query, k=2)

        prompt = f"""You are a life insurance actuarial expert. Based on the rating criteria below, provide a premium estimate.

Rating Criteria from Knowledge Base:
{criteria_result.get('context', 'Use industry standard rating factors')}

Applicant Profile:
- Age: {age} years
- Coverage Amount: ${coverage_amount:,}
- Term Length: {term_length} years
- Smoker: {'Yes' if is_smoker else 'No'}
- Health Rating: {health_rating}

Base Rate: ${settings.premium_base_rate} per $1,000 of coverage per month

Provide your analysis and calculation. Include:
1. Monthly premium estimate
2. Annual premium (monthly × 12)
3. Total term cost (annual × {term_length})
4. Brief explanation of factors applied

IMPORTANT: Use plain text for calculations, NOT LaTeX or mathematical notation.
For example: "Monthly Premium = (Coverage / 1,000) × Base Rate = (500,000 / 1,000) × 0.05 = $25"

Format your response clearly and professionally."""

        response = llm_service.invoke([{"role": "user", "content": prompt}])

        import re

        monthly_match = re.search(
            r"\$?(\d+[.,]?\d*)\s*(?:per month|monthly)", response, re.IGNORECASE
        )
        annual_match = re.search(
            r"\$?(\d+[.,]?\d*)\s*(?:per year|annual|yearly)", response, re.IGNORECASE
        )

        if monthly_match:
            monthly_premium = float(monthly_match.group(1).replace(",", ""))
            annual_premium = monthly_premium * 12
            total_term_cost = annual_premium * term_length
        else:
            base_rate = settings.premium_base_rate
            age_factor = 1.0 + ((age - 30) * 0.015) if age >= 30 else 0.85
            smoker_factor = settings.premium_smoker_multiplier if is_smoker else 1.0
            monthly_premium = (
                (coverage_amount / 1000) * base_rate * age_factor * smoker_factor
            )
            annual_premium = monthly_premium * 12
            total_term_cost = annual_premium * term_length

        return {
            "success": True,
            "monthly_premium": round(monthly_premium, 2),
            "annual_premium": round(annual_premium, 2),
            "total_term_cost": round(total_term_cost, 2),
            "factors": {
                "age": age,
                "coverage_amount": f"${coverage_amount:,}",
                "term_length": f"{term_length} years",
                "smoker_status": "Smoker" if is_smoker else "Non-smoker",
                "health_rating": health_rating,
            },
            "explanation": response,
            "note": "This is an AI-generated estimate based on industry standards. Actual premiums may vary.",
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
    Check life insurance eligibility using AI-powered underwriting analysis.
    """
    try:
        health_conditions = health_conditions or []
        coverage_amount = coverage_amount or settings.tool_default_coverage

        criteria_query = "life insurance eligibility underwriting criteria risk assessment health conditions"
        criteria_result = search_knowledge_base(criteria_query, k=2)

        conditions_str = (
            ", ".join(health_conditions) if health_conditions else "none reported"
        )

        prompt = f"""You are an experienced life insurance underwriting expert. Assess the eligibility for this applicant based on the criteria below.

Underwriting Criteria from Knowledge Base:
{criteria_result.get('context', 'Use standard underwriting guidelines')}

Applicant Profile:
- Age: {age} years
- Health Conditions: {conditions_str}
- Smoker: {'Yes' if smoker else 'No'}
- Occupation: {occupation}
- Desired Coverage: ${coverage_amount:,}

Provide a comprehensive assessment including:
1. Overall eligibility status (Good/Moderate/Challenging)
2. Likelihood of approval (likely/possible/challenging)
3. Key issues or concerns
4. Specific recommendations for this applicant
5. Your professional reasoning

IMPORTANT: Use plain text formatting only. NO LaTeX or mathematical notation.

Be specific and actionable in your response."""

        response = llm_service.invoke([{"role": "user", "content": prompt}])

        response_lower = response.lower()

        if "good" in response_lower or "excellent" in response_lower:
            eligibility_status = "Good"
            likely_approved = True
        elif "challenging" in response_lower or "difficult" in response_lower:
            eligibility_status = "Challenging"
            likely_approved = False
        else:
            eligibility_status = "Moderate"
            likely_approved = True

        issues_section = re.search(
            r"(?:issues|concerns)[:\s]+(.*?)(?:\n\n|recommendations|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        recommendations_section = re.search(
            r"recommendations[:\s]+(.*?)(?:\n\n|reasoning|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )

        issues = []
        if issues_section:
            issues_text = issues_section.group(1).strip()
            issues = [
                line.strip("- ").strip()
                for line in issues_text.split("\n")
                if line.strip() and not line.strip().startswith("Recommendations")
            ]

        recommendations = []
        if recommendations_section:
            rec_text = recommendations_section.group(1).strip()
            recommendations = [
                line.strip("- ").strip()
                for line in rec_text.split("\n")
                if line.strip()
            ]

        return {
            "success": True,
            "eligibility": eligibility_status,
            "likely_approved": likely_approved,
            "issues": issues if issues else ["See assessment details below"],
            "recommendations": (
                recommendations
                if recommendations
                else ["Consult with insurance broker for personalized guidance"]
            ),
            "reasoning": response,
            "suggested_actions": [
                "Get quotes from multiple insurers",
                "Prepare medical records and documentation",
                "Consider working with an independent insurance broker",
                "Review different policy types for your situation",
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
