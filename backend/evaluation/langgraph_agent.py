import os
import json
import re
from typing import TypedDict, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()


# ── State Definition ───────────────────────────────────────────────────────────
class EvaluatorState(TypedDict):
    """
    State object that flows through every node in the LangGraph.
    Each node reads from state and returns updated keys.
    Think of this as a shared memory that all nodes can read and write.
    """
    pipeline_scores: Dict    # RAGAS scores from evaluate_pipeline()
    cost_data: Dict          # Hardcoded cost estimates per pipeline
    analysis: str            # GPT-4o analysis of scores (set by analyze_scores)
    recommendation: str      # GPT-4o recommendation text (set by generate_recommendation)
    final_report: Dict       # Parsed structured output (set by format_output)


# ── Node 1: Analyze Scores ─────────────────────────────────────────────────────
def analyze_scores(state: EvaluatorState) -> Dict:
    """
    Node 1 — Uses GPT-4o to compare RAGAS metrics across all pipelines.
    Identifies which pipeline excels at each metric and notes trade-offs.

    Reads:  state["pipeline_scores"]
    Writes: state["analysis"]
    """
    print("Node 1: Analyzing RAGAS scores...")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    scores_summary = json.dumps(state["pipeline_scores"], indent=2)

    response = llm.invoke(f"""
You are an expert in RAG (Retrieval Augmented Generation) systems evaluation.
Analyze these RAGAS evaluation scores from 4 different RAG pipeline configurations:

{scores_summary}

The 4 metrics mean:
- faithfulness: Does the answer contain ONLY info from retrieved context? (catches hallucination)
- answer_relevancy: Does the answer actually address the question asked?
- context_precision: Are retrieved chunks actually relevant to the question?
- context_recall: Did the retriever find ALL relevant information?

Provide a concise technical analysis that:
1. Identifies which pipeline excels at each metric
2. Notes any significant trade-offs between pipelines
3. Highlights any surprising results worth noting
4. Keeps each point to 1-2 sentences maximum

Be direct and technical — this analysis is for an engineering team.
""")

    return {"analysis": response.content}


# ── Node 2: Calculate Cost ─────────────────────────────────────────────────────
def calculate_cost(state: EvaluatorState) -> Dict:
    """
    Node 2 — Hardcodes cost estimates per 1000 queries for each pipeline.
    Based on public API pricing as of 2024.

    Cost assumptions per query:
    - Average question: ~50 tokens
    - Average context passed to LLM: ~800 tokens
    - Average answer: ~200 tokens
    - OpenAI ada-002 embeddings: $0.10 per 1M tokens
    - Cohere embed-english-v3: $0.10 per 1M tokens
    - Cohere Rerank v3: $2.00 per 1000 searches
    - BGE-large + Cross-encoder: $0 (runs locally)
    - GPT-4o-mini: $0.15 input / $0.60 output per 1M tokens

    Reads:  nothing (hardcoded)
    Writes: state["cost_data"]
    """
    print("Node 2: Calculating cost estimates...")

    cost_data = {
        "Pipeline 1 — Fixed 512 + OpenAI": {
            "embedding_model": "OpenAI text-embedding-ada-002",
            "embedding_cost": "$0.10 per 1M tokens",
            "reranking": "None",
            "reranking_cost": "$0",
            "llm": "GPT-4o-mini",
            "llm_cost": "$0.60 per 1M output tokens",
            "estimated_per_1k_queries": "$0.45",
            "monthly_cost_10k_queries": "$4.50",
        },
        "Pipeline 2 — Recursive + Cohere Rerank": {
            "embedding_model": "Cohere embed-english-v3.0",
            "embedding_cost": "$0.10 per 1M tokens",
            "reranking": "Cohere Rerank v3.0",
            "reranking_cost": "$2.00 per 1000 searches",
            "llm": "GPT-4o-mini",
            "llm_cost": "$0.60 per 1M output tokens",
            "estimated_per_1k_queries": "$2.55",
            "monthly_cost_10k_queries": "$25.50",
        },
        "Pipeline 3 — Semantic + BGE + Cross-Encoder": {
            "embedding_model": "BGE-large-en-v1.5 (local)",
            "embedding_cost": "$0 (open source)",
            "reranking": "ms-marco-MiniLM cross-encoder (local)",
            "reranking_cost": "$0 (open source)",
            "llm": "GPT-4o-mini",
            "llm_cost": "$0.60 per 1M output tokens",
            "estimated_per_1k_queries": "$0.15",
            "monthly_cost_10k_queries": "$1.50",
        },
        "Pipeline 4 — Fixed 1024 + MMR": {
            "embedding_model": "OpenAI text-embedding-ada-002",
            "embedding_cost": "$0.10 per 1M tokens",
            "reranking": "MMR (built-in, no extra cost)",
            "reranking_cost": "$0",
            "llm": "GPT-4o-mini",
            "llm_cost": "$0.60 per 1M output tokens",
            "estimated_per_1k_queries": "$0.45",
            "monthly_cost_10k_queries": "$4.50",
        },
    }

    return {"cost_data": cost_data}


# ── Node 3: Generate Recommendation ───────────────────────────────────────────
def generate_recommendation(state: EvaluatorState) -> Dict:
    """
    Node 3 — Synthesizes the analysis and cost data into a structured
    recommendation using GPT-4o.

    Reads:  state["analysis"], state["cost_data"], state["pipeline_scores"]
    Writes: state["recommendation"]
    """
    print("Node 3: Generating recommendation...")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    response = llm.invoke(f"""
You are an expert RAG systems architect making a recommendation to an engineering team.

RAGAS Evaluation Analysis:
{state["analysis"]}

Cost Data:
{json.dumps(state["cost_data"], indent=2)}

Raw Scores:
{json.dumps(state["pipeline_scores"], indent=2)}

Based on both performance AND cost, provide a structured recommendation.
You MUST respond in EXACTLY this format with these exact labels:

WINNER: [Pipeline name]
REASON: [2-3 sentences explaining why this pipeline wins considering both performance and cost]
TRADEOFF: [1-2 sentences on what you sacrifice by choosing this pipeline]
BEST_FOR_QUALITY: [Pipeline name if cost is no concern — pure performance winner]
BEST_FOR_COST: [Pipeline name if you need the lowest cost with acceptable performance]
PRODUCTION_ADVICE: [1-2 sentences of practical advice for deploying this in production]
""")

    return {"recommendation": response.content}


# ── Node 4: Format Output ──────────────────────────────────────────────────────
def format_output(state: EvaluatorState) -> Dict:
    """
    Node 4 — Parses the recommendation text into a clean structured dict
    for the API and dashboard to consume.

    Reads:  state["recommendation"], state["pipeline_scores"], state["cost_data"]
    Writes: state["final_report"]
    """
    print("Node 4: Formatting final report...")

    recommendation_text = state["recommendation"]

    # Parse each labeled field from the recommendation
    def extract_field(label: str, text: str) -> str:
        pattern = rf"{label}:\s*(.+?)(?=\n[A-Z_]+:|$)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else "Not available"

    final_report = {
        "winner": extract_field("WINNER", recommendation_text),
        "reason": extract_field("REASON", recommendation_text),
        "tradeoff": extract_field("TRADEOFF", recommendation_text),
        "best_for_quality": extract_field("BEST_FOR_QUALITY", recommendation_text),
        "best_for_cost": extract_field("BEST_FOR_COST", recommendation_text),
        "production_advice": extract_field("PRODUCTION_ADVICE", recommendation_text),
        "pipeline_scores": state["pipeline_scores"],
        "cost_data": state["cost_data"],
        "full_analysis": state["analysis"],
        "raw_recommendation": recommendation_text,
    }

    return {"final_report": final_report}


# ── Graph Builder ──────────────────────────────────────────────────────────────
def build_evaluator_agent() -> StateGraph:
    """
    Build and compile the LangGraph evaluator agent.

    Graph structure:
    analyze_scores → calculate_cost → generate_recommendation → format_output → END

    Each node is a pure function that reads state and returns updated keys.
    LangGraph merges the returned dict back into the shared state automatically.

    Returns:
        Compiled LangGraph StateGraph ready to invoke
    """
    workflow = StateGraph(EvaluatorState)

    # Register nodes
    workflow.add_node("analyze_scores", analyze_scores)
    workflow.add_node("calculate_cost", calculate_cost)
    workflow.add_node("generate_recommendation", generate_recommendation)
    workflow.add_node("format_output", format_output)

    # Define edges — linear flow for this evaluator
    workflow.set_entry_point("analyze_scores")
    workflow.add_edge("analyze_scores", "calculate_cost")
    workflow.add_edge("calculate_cost", "generate_recommendation")
    workflow.add_edge("generate_recommendation", "format_output")
    workflow.add_edge("format_output", END)

    return workflow.compile()


# ── Convenience Function ───────────────────────────────────────────────────────
def run_evaluation(pipeline_scores: Dict) -> Dict:
    """
    Convenience function to run the full evaluator agent in one call.

    Args:
        pipeline_scores: Dict of {pipeline_name: {metric: score}}
                         Output from ragas_eval.evaluate_pipeline()

    Returns:
        final_report dict with winner, reason, tradeoff, costs, and scores
    """
    agent = build_evaluator_agent()

    # Initialize state — all fields must be present even if empty
    initial_state = {
        "pipeline_scores": pipeline_scores,
        "cost_data": {},
        "analysis": "",
        "recommendation": "",
        "final_report": {},
    }

    result = agent.invoke(initial_state)
    return result["final_report"]