import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from backend.utils.vector_store_manager import VectorStoreManager, PIPELINE_4
from backend.utils.document_processor import DocumentProcessor
from backend.pipelines.pipeline_4 import run_pipeline_4

load_dotenv()

async def main():
    # Pipeline 4 uses fixed chunking at 1024 size
    processor = DocumentProcessor()
    docs = processor.load_pdf("data/sample_docs/n8n guide.pdf")
    chunks = processor.chunk_fixed(docs, chunk_size=1024)

    # OpenAI embeddings same as Pipeline 1
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    # Create ChromaDB collection for pipeline 4
    manager = VectorStoreManager()
    vectorstore = manager.get_or_create_collection(
        chunks=chunks,
        embeddings=embeddings,
        pipeline_name=PIPELINE_4,
        use_qdrant=False,
    )

    # Run pipeline with same question for fair comparison
    question = "How do I set up n8n for the first time?"
    print(f"\nQuestion: {question}\n")

    result = await run_pipeline_4(vectorstore, question)

    print(f"Pipeline: {result['pipeline_name']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nRetrieved {len(result['retrieved_chunks'])} chunks via MMR")
    print(f"\nFirst chunk preview:\n{result['retrieved_chunks'][0][:200]}...")

    # Show all 4 chunk previews to demonstrate diversity
    print("\n--- All retrieved chunks (notice diversity vs Pipeline 1) ---")
    for i, chunk in enumerate(result['retrieved_chunks']):
        print(f"\nChunk {i+1} ({len(chunk)} chars):\n{chunk[:150]}...")

asyncio.run(main())