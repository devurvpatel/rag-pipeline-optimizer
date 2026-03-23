import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings

from backend.utils.document_processor import DocumentProcessor
from backend.utils.vector_store_manager import (
    VectorStoreManager, PIPELINE_1, PIPELINE_2, PIPELINE_3, PIPELINE_4
)
from backend.pipelines.pipeline_1 import run_pipeline_1
from backend.pipelines.pipeline_2 import run_pipeline_2
from backend.pipelines.pipeline_3 import run_pipeline_3, get_bge_embeddings
from backend.pipelines.pipeline_4 import run_pipeline_4
from backend.evaluation.ragas_eval import evaluate_pipeline, compare_pipelines
from data.eval_dataset import EVAL_QUESTIONS, EVAL_GROUND_TRUTHS_CLEAN

load_dotenv()

# Use first 5 questions only for testing — full 25 costs more API calls
TEST_QUESTIONS = EVAL_QUESTIONS[:2]
TEST_GROUND_TRUTHS = EVAL_GROUND_TRUTHS_CLEAN[:2]

async def run_all_pipelines(vectorstores, questions):
    """Run all 4 pipelines on all questions and collect results."""
    all_answers = {1: [], 2: [], 3: [], 4: []}
    all_contexts = {1: [], 2: [], 3: [], 4: []}

    for question in questions:
        print(f"\nRunning question: {question[:60]}...")

        r1 = await run_pipeline_1(vectorstores[1], question)
        r2 = await run_pipeline_2(vectorstores[2], question)
        r3 = await run_pipeline_3(vectorstores[3], question)
        r4 = await run_pipeline_4(vectorstores[4], question)

        for pipeline_num, result in [(1,r1),(2,r2),(3,r3),(4,r4)]:
            all_answers[pipeline_num].append(result["answer"])
            all_contexts[pipeline_num].append(result["retrieved_chunks"])

    return all_answers, all_contexts


async def main():
    processor = DocumentProcessor()
    docs = processor.load_pdf("data/sample_docs/n8n guide.pdf")
    manager = VectorStoreManager()

    # Load all 4 existing collections
    print("Loading vector store collections...")
    vectorstores = {
        1: manager.load_chroma_collection(
            OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")), PIPELINE_1),
        2: manager.load_chroma_collection(
            CohereEmbeddings(model="embed-english-v3.0",
            cohere_api_key=os.getenv("COHERE_API_KEY")), PIPELINE_2),
        3: manager.load_chroma_collection(
            get_bge_embeddings(), PIPELINE_3),
        4: manager.load_chroma_collection(
            OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")), PIPELINE_4),
    }

    # Run all pipelines on test questions
    print(f"\nRunning {len(TEST_QUESTIONS)} questions through all 4 pipelines...")
    all_answers, all_contexts = await run_all_pipelines(vectorstores, TEST_QUESTIONS)

    # Evaluate each pipeline with RAGAS
    all_results = {}

    pipeline_configs = [
        (1, "Pipeline 1 — Fixed 512 + OpenAI"),
        (2, "Pipeline 2 — Recursive + Cohere Rerank"),
        (3, "Pipeline 3 — Semantic + BGE + Cross-Encoder"),
        (4, "Pipeline 4 — Fixed 1024 + MMR"),
    ]

    for num, name in pipeline_configs:
        result = evaluate_pipeline(
            pipeline_name=name,
            questions=TEST_QUESTIONS,
            answers=all_answers[num],
            contexts=all_contexts[num],
            ground_truths=TEST_GROUND_TRUTHS,
        )
        all_results[name] = result

    # Print comparison
    summary = compare_pipelines(all_results)
    print(summary)

asyncio.run(main())
