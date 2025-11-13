import logging
from typing import Any, Dict, Optional

from app.services.rag import rag_service

logger = logging.getLogger(__name__)


def search_knowledge_base(query: str, k: int = 4) -> Dict[str, Any]:
    """
    Search the life insurance knowledge base for relevant information.

    Args:
        query: The search query
        k: Number of results to return

    Returns:
        Dictionary containing search results and formatted context
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
    Calculate estimated life insurance premium.

    Args:
        age: Applicant's age
        coverage_amount: Desired coverage amount in dollars
        term_length: Term length in years (10, 15, 20, 25, 30)
        is_smoker: Whether applicant is a smoker
        health_rating: Health rating (preferred_plus, preferred, standard, substandard)

    Returns:
        Dictionary with premium estimate and breakdown
    """
    try:
        base_rate_per_1000 = 0.05

        age_multiplier = 1.0
        if age < 30:
            age_multiplier = 0.8
        elif age < 40:
            age_multiplier = 1.0
        elif age < 50:
            age_multiplier = 1.5
        elif age < 60:
            age_multiplier = 2.5
        else:
            age_multiplier = 4.0

        term_multiplier = {10: 0.9, 15: 1.0, 20: 1.1, 25: 1.2, 30: 1.3}.get(
            term_length, 1.0
        )

        health_multiplier = {
            "preferred_plus": 0.8,
            "preferred": 0.9,
            "standard": 1.0,
            "substandard": 1.5,
        }.get(health_rating.lower(), 1.0)

        smoker_multiplier = 2.5 if is_smoker else 1.0

        coverage_units = coverage_amount / 1000

        monthly_premium = (
            base_rate_per_1000
            * coverage_units
            * age_multiplier
            * term_multiplier
            * health_multiplier
            * smoker_multiplier
        )

        annual_premium = monthly_premium * 12
        total_cost = annual_premium * term_length

        return {
            "success": True,
            "monthly_premium": round(monthly_premium, 2),
            "annual_premium": round(annual_premium, 2),
            "total_term_cost": round(total_cost, 2),
            "factors": {
                "age": age,
                "coverage_amount": f"${coverage_amount:,}",
                "term_length": f"{term_length} years",
                "smoker_status": "Smoker" if is_smoker else "Non-smoker",
                "health_rating": health_rating,
            },
            "note": "This is an estimate. Actual premiums may vary based on detailed underwriting.",
        }

    except Exception as e:
        logger.error(f"Error calculating premium: {str(e)}")
        return {"success": False, "error": str(e)}


def check_eligibility(
    age: int,
    health_conditions: list = None,
    smoker: bool = False,
    occupation: str = "standard",
    coverage_amount: int = 500000,
) -> Dict[str, Any]:
    """
    Check basic eligibility for life insurance.

    Args:
        age: Applicant's age
        health_conditions: List of health conditions
        smoker: Whether applicant smokes
        occupation: Type of occupation
        coverage_amount: Desired coverage amount

    Returns:
        Dictionary with eligibility assessment
    """
    try:
        health_conditions = health_conditions or []

        issues = []
        recommendations = []

        if age < 18:
            issues.append("Must be at least 18 years old for individual coverage")
        elif age > 75:
            issues.append(
                "Age may limit term life options; consider guaranteed issue or final expense"
            )

        if age > 65 and coverage_amount > 1000000:
            issues.append(
                "High coverage amounts may require extensive underwriting at this age"
            )

        high_risk_conditions = [
            "cancer",
            "heart disease",
            "kidney disease",
            "liver disease",
            "hiv",
            "aids",
            "terminal illness",
        ]

        moderate_risk_conditions = [
            "diabetes",
            "high blood pressure",
            "high cholesterol",
            "asthma",
            "depression",
            "anxiety",
        ]

        has_high_risk = any(
            condition in " ".join(health_conditions).lower()
            for condition in high_risk_conditions
        )

        has_moderate_risk = any(
            condition in " ".join(health_conditions).lower()
            for condition in moderate_risk_conditions
        )

        if has_high_risk:
            issues.append(
                "Serious health conditions may require specialized underwriting or guaranteed issue"
            )
            recommendations.append(
                "Consider working with an agent who specializes in high-risk cases"
            )
        elif has_moderate_risk:
            recommendations.append(
                "Condition management documentation will be important for underwriting"
            )

        if smoker:
            recommendations.append(
                "Consider quitting smoking for 12+ months to qualify for non-smoker rates (save 50-70%)"
            )

        high_risk_occupations = ["construction", "mining", "pilot", "firefighter"]
        if any(occ in occupation.lower() for occ in high_risk_occupations):
            recommendations.append(
                "High-risk occupation may require specialized policy or occupational exclusions"
            )

        if 18 <= age <= 75 and not has_high_risk:
            eligibility = "Good" if not has_moderate_risk else "Moderate"
            likely_approved = True
        else:
            eligibility = "Challenging"
            likely_approved = False

        return {
            "success": True,
            "eligibility": eligibility,
            "likely_approved": likely_approved,
            "issues": issues,
            "recommendations": recommendations,
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

    Args:
        policy_types: List of policy types to compare

    Returns:
        Dictionary with comparison information
    """
    try:
        query = f"Compare {', '.join(policy_types)} life insurance policies including features, benefits, and costs"
        results = search_knowledge_base(query, k=6)

        return {
            "success": True,
            "comparison": results.get("context", ""),
            "sources": results.get("sources", []),
        }

    except Exception as e:
        logger.error(f"Error getting policy comparison: {str(e)}")
        return {"success": False, "error": str(e)}
