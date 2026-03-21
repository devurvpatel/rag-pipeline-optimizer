import json
from backend.evaluation.langgraph_agent import run_evaluation

# Realistic fake RAGAS scores — no API cost to test the agent
fake_scores = {
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

print("Running LangGraph evaluator agent...")
print("4 nodes will run in sequence: analyze → cost → recommend → format\n")

report = run_evaluation(fake_scores)

print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)
print(f"\nWINNER:            {report['winner']}")
print(f"\nREASON:            {report['reason']}")
print(f"\nTRADEOFF:          {report['tradeoff']}")
print(f"\nBEST FOR QUALITY:  {report['best_for_quality']}")
print(f"\nBEST FOR COST:     {report['best_for_cost']}")
print(f"\nPRODUCTION ADVICE: {report['production_advice']}")
print("\n" + "=" * 60)
print("FULL ANALYSIS:")
print("=" * 60)
print(report['full_analysis'])
