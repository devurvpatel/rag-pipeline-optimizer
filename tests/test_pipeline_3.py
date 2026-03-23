import asyncio
import os
from dotenv import load_dotenv

from backend.utils.vector_store_manager import VectorStoreManager, PIPELINE_3
from backend.utils.document_processor import DocumentProcessor
from backend.pipelines.pipeline_3 import run_pipeline_3, get_bge_embeddings

load_dotenv()

async def main():
    # Load and chunk with semantic strategy for pipeline 3
    processor = DocumentProcessor()
    docs = processor.load_pdf("data/sample_docs/n8n guide.pdf")

    # Semantic chunking makes OpenAI API calls — takes 1-2 minutes
    print("Running semantic chunking (this takes 1-2 minutes)...")
    chunks = processor.chunk_semantic(docs)

    # BGE embeddings — runs locally, no API cost
    print("Loading BGE embeddings (downloads model on first run ~1.3GB)...")
    embeddings = get_bge_embeddings()

    # Create ChromaDB collection for pipeline 3
    manager = VectorStoreManager()
    vectorstore = manager.get_or_create_collection(
        chunks=chunks,
        embeddings=embeddings,
        pipeline_name=PIPELINE_3,
        use_qdrant=False,
    )

    # Run pipeline
    question = "How do I set up n8n for the first time?"
    print(f"\nQuestion: {question}\n")

    result = await run_pipeline_3(vectorstore, question)

    print(f"Pipeline: {result['pipeline_name']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nRetrieved {len(result['retrieved_chunks'])} chunks after reranking")
    print(f"\nFirst chunk preview:\n{result['retrieved_chunks'][0][:200]}...")

asyncio.run(main())