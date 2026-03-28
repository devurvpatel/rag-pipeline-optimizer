"""
Tests for LangGraph Evaluator Agent — runs without real API keys.
All GPT-4o calls are mocked to avoid API costs in CI.
"""
import pytest
from unittest.mock import MagicMock, patch
from backend.evaluation.langgraph_agent import (
    build_evaluator_agent,
    run_evaluation,
    calculate_cost,
    analyze_scores,
    generate_recommendation,
    format_output,
    EvaluatorState,
)


# ── Sample RAGAS scores for testing ───────────────────────────────────────────
SAMPLE_SCORES = {
    "Pipeline 1 — Fixed 512 + OpenAI": {
        "faithfulness": 0.625,
        "answer_relevancy": 0.979,
        "context_precision": 0.500,
        "context_recall": 0.833,
    },
    "Pipeline 2 — Recursive + Cohere Rerank": {
        "faithfulness": 0.500,
        "answer_relevancy": 0.979,
        "context_precision": 0.917,
        "context_recall": 1.000,
    },
    "Pipeline 3 — Semantic + BGE + Cross-Encoder": {
        "faithfulness": 0.500,
        "answer_relevancy": 0.979,
        "context_precision": 0.500,
        "context_recall": 0.583,
    },
    "Pipeline 4 — Fixed 1024 + MMR": {
        "faithfulness": 0.500,
        "answer_relevancy": 0.979,
        "context_precision": 1.000,
        "context_recall": 1.000,
    },
}

SAMPLE_STATE: EvaluatorState = {
    "pipeline_scores": SAMPLE_SCORES,
    "cost_data": {},
    "analysis": "",
    "recommendation": "",
    "final_report": {},
}


# ── calculate_cost Node Tests ──────────────────────────────────────────────────
class TestCalculateCost:

    def test_calculate_cost_returns_cost_data(self):
        """calculate_cost should return a dict with cost_data key."""
        result = calculate_cost(SAMPLE_STATE)
        assert "cost_data" in result

    def test_calculate_cost_has_all_pipelines(self):
        """cost_data should contain entries for all 4 pipelines."""
        result = calculate_cost(SAMPLE_STATE)
        cost_data = result["cost_data"]
        assert len(cost_data) == 4

    def test_calculate_cost_uses_live_costs_when_available(self):
        """calculate_cost should use existing cost_data if already populated."""
        state_with_costs: EvaluatorState = {
            **SAMPLE_STATE,
            "cost_data": {
                "Pipeline 1": {"cost_usd_this_query": 0.0001}
            }
        }
        result = calculate_cost(state_with_costs)
        assert result["cost_data"] == {"Pipeline 1": {"cost_usd_this_query": 0.0001}}

    def test_calculate_cost_fallback_has_required_fields(self):
        """Fallback cost data should have embedding_model, reranking, llm fields."""
        result = calculate_cost(SAMPLE_STATE)
        cost_data = result["cost_data"]
        for pipeline_name, costs in cost_data.items():
            assert "embedding_model" in costs
            assert "reranking" in costs
            assert "llm" in costs


# ── analyze_scores Node Tests ──────────────────────────────────────────────────
class TestAnalyzeScores:

    @patch("backend.evaluation.langgraph_agent.ChatOpenAI")
    def test_analyze_scores_returns_analysis(self, mock_llm_class):
        """analyze_scores should return a dict with analysis key."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Pipeline 4 leads in context precision with 1.0."
        )
        mock_llm_class.return_value = mock_llm

        result = analyze_scores(SAMPLE_STATE)

        assert "analysis" in result
        assert isinstance(result["analysis"], str)
        assert len(result["analysis"]) > 0

    @patch("backend.evaluation.langgraph_agent.ChatOpenAI")
    def test_analyze_scores_calls_llm(self, mock_llm_class):
        """analyze_scores should call the LLM exactly once."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Analysis text")
        mock_llm_class.return_value = mock_llm

        analyze_scores(SAMPLE_STATE)

        mock_llm.invoke.assert_called_once()


# ── generate_recommendation Node Tests ────────────────────────────────────────
class TestGenerateRecommendation:

    @patch("backend.evaluation.langgraph_agent.ChatOpenAI")
    def test_generate_recommendation_returns_recommendation(self, mock_llm_class):
        """generate_recommendation should return a dict with recommendation key."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="""
WINNER: Pipeline 4 — Fixed 1024 + MMR
REASON: Achieves perfect context precision and recall.
TRADEOFF: Moderate faithfulness score.
BEST_FOR_QUALITY: Pipeline 1 — Fixed 512 + OpenAI
BEST_FOR_COST: Pipeline 3 — Semantic + BGE + Cross-Encoder
PRODUCTION_ADVICE: Monitor faithfulness in production.
""")
        mock_llm_class.return_value = mock_llm

        state_with_analysis: EvaluatorState = {
            **SAMPLE_STATE,
            "analysis": "Pipeline 4 leads in context metrics.",
            "cost_data": {"Pipeline 1": {"embedding_model": "OpenAI"}},
        }

        result = generate_recommendation(state_with_analysis)

        assert "recommendation" in result
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0


# ── format_output Node Tests ───────────────────────────────────────────────────
class TestFormatOutput:

    def test_format_output_returns_final_report(self):
        """format_output should return a dict with final_report key."""
        state_with_recommendation: EvaluatorState = {
            **SAMPLE_STATE,
            "analysis": "Pipeline 4 leads in context metrics.",
            "cost_data": {},
            "recommendation": """
WINNER: Pipeline 4 — Fixed 1024 + MMR
REASON: Achieves perfect scores in context precision and recall.
TRADEOFF: Moderate faithfulness score may introduce hallucinations.
BEST_FOR_QUALITY: Pipeline 1 — Fixed 512 + OpenAI
BEST_FOR_COST: Pipeline 3 — Semantic + BGE + Cross-Encoder
PRODUCTION_ADVICE: Monitor faithfulness metrics closely in production.
""",
        }

        result = format_output(state_with_recommendation)

        assert "final_report" in result
        report = result["final_report"]
        assert "winner" in report
        assert "reason" in report
        assert "tradeoff" in report
        assert "best_for_quality" in report
        assert "best_for_cost" in report
        assert "production_advice" in report

    def test_format_output_parses_winner_correctly(self):
        """format_output should extract the WINNER field correctly."""
        state: EvaluatorState = {
            **SAMPLE_STATE,
            "analysis": "test",
            "cost_data": {},
            "recommendation": """
WINNER: Pipeline 4 — Fixed 1024 + MMR
REASON: Best overall performance.
TRADEOFF: Higher latency.
BEST_FOR_QUALITY: Pipeline 2 — Recursive + Cohere Rerank
BEST_FOR_COST: Pipeline 3 — Semantic + BGE + Cross-Encoder
PRODUCTION_ADVICE: Use in production with monitoring.
""",
        }

        result = format_output(state)
        assert "Pipeline 4" in result["final_report"]["winner"]


# ── build_evaluator_agent Tests ────────────────────────────────────────────────
class TestBuildEvaluatorAgent:

    def test_build_evaluator_agent_returns_compiled_graph(self):
        """build_evaluator_agent should return a compiled LangGraph."""
        agent = build_evaluator_agent()
        assert agent is not None

    def test_agent_has_invoke_method(self):
        """Compiled agent should have an invoke method."""
        agent = build_evaluator_agent()
        assert hasattr(agent, "invoke")


# ── run_evaluation Integration Tests ──────────────────────────────────────────
class TestRunEvaluation:

    @patch("backend.evaluation.langgraph_agent.ChatOpenAI")
    def test_run_evaluation_returns_final_report(self, mock_llm_class):
        """run_evaluation should return a complete final report dict."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content="Pipeline 4 leads in context metrics."),
            MagicMock(content="""
WINNER: Pipeline 4 — Fixed 1024 + MMR
REASON: Perfect context precision and recall scores.
TRADEOFF: Moderate faithfulness score.
BEST_FOR_QUALITY: Pipeline 1 — Fixed 512 + OpenAI
BEST_FOR_COST: Pipeline 3 — Semantic + BGE + Cross-Encoder
PRODUCTION_ADVICE: Monitor faithfulness in production.
"""),
        ]
        mock_llm_class.return_value = mock_llm

        report = run_evaluation(SAMPLE_SCORES)

        assert isinstance(report, dict)
        assert "winner" in report
        assert "reason" in report
        assert "tradeoff" in report
        assert "best_for_quality" in report
        assert "best_for_cost" in report
        assert "production_advice" in report
        assert "pipeline_scores" in report

    @patch("backend.evaluation.langgraph_agent.ChatOpenAI")
    def test_run_evaluation_includes_pipeline_scores(self, mock_llm_class):
        """Final report should include the original pipeline scores."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content="Analysis text"),
            MagicMock(content="""
WINNER: Pipeline 4 — Fixed 1024 + MMR
REASON: Best scores.
TRADEOFF: Some tradeoff.
BEST_FOR_QUALITY: Pipeline 1 — Fixed 512 + OpenAI
BEST_FOR_COST: Pipeline 3 — Semantic + BGE + Cross-Encoder
PRODUCTION_ADVICE: Monitor in production.
"""),
        ]
        mock_llm_class.return_value = mock_llm

        report = run_evaluation(SAMPLE_SCORES)
        assert report["pipeline_scores"] == SAMPLE_SCORES