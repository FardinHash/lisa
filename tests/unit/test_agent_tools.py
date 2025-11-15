import pytest

from app.agents.tools import (calculate_premium_estimate, check_eligibility,
                              get_policy_comparison)


class TestPremiumCalculator:
    def test_basic_calculation(self, mock_llm_response, mock_rag_service):
        result = calculate_premium_estimate(
            age=35, coverage_amount=500000, term_length=20, is_smoker=False
        )

        assert result["success"] is True
        assert "monthly_premium" in result
        assert "annual_premium" in result
        assert "total_term_cost" in result
        assert result["factors"]["age"] == 35
        assert result["factors"]["coverage_amount"] == "$500,000"

    def test_smoker_vs_nonsmoker(self, mock_llm_response, mock_rag_service):
        non_smoker = calculate_premium_estimate(35, 500000, 20, False)
        smoker = calculate_premium_estimate(35, 500000, 20, True)

        assert non_smoker["success"] is True
        assert smoker["success"] is True
        assert smoker["factors"]["smoker_status"] == "Smoker"
        assert non_smoker["factors"]["smoker_status"] == "Non-smoker"

    def test_premium_factors(self, mock_llm_response, mock_rag_service):
        result = calculate_premium_estimate(40, 1000000, 30, True, "preferred")

        assert result["success"] is True
        assert result["factors"]["age"] == 40
        assert result["factors"]["term_length"] == "30 years"
        assert result["factors"]["health_rating"] == "preferred"


class TestEligibilityChecker:
    def test_basic_eligibility(self, mock_llm_response, mock_rag_service):
        result = check_eligibility(age=30, health_conditions=[], smoker=False)

        assert result["success"] is True
        assert "eligibility" in result
        assert "suggested_actions" in result

    def test_with_health_conditions(self, mock_llm_response, mock_rag_service):
        result = check_eligibility(
            age=45, health_conditions=["diabetes", "high blood pressure"], smoker=True
        )

        assert result["success"] is True
        assert "eligibility" in result
        assert "recommendations" in result

    def test_with_occupation(self, mock_llm_response, mock_rag_service):
        result = check_eligibility(age=35, occupation="pilot", coverage_amount=1000000)

        assert result["success"] is True
        assert "eligibility" in result


class TestPolicyComparison:
    def test_policy_comparison(self, mock_llm_response, mock_rag_service):
        result = get_policy_comparison(["term", "whole"])

        assert result["success"] is True
        assert "comparison" in result
        assert "sources" in result

    def test_multiple_policies(self, mock_llm_response, mock_rag_service):
        result = get_policy_comparison(["term", "whole", "universal", "variable"])

        assert result["success"] is True
        assert result["comparison"] is not None
