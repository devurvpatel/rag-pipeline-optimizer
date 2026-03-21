import asyncio
import os
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings

from backend.utils.vector_store_manager import VectorStoreManager, PIPELINE_2
from backend.utils.document_processor import DocumentProcessor
from backend.pipelines.pipeline_2 import run_pipeline_2

load_dotenv()

async def main():
    # Load and chunk with recursive strategy for pipeline 2
    processor = DocumentProcessor()
    docs = processor.load_pdf("data/sample_docs/n8n guide.pdf")
    chunks = processor.chunk_recursive(docs)

    # Cohere embeddings for pipeline 2
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )

    # Create ChromaDB collection for pipeline 2
    manager = VectorStoreManager()
    vectorstore = manager.get_or_create_collection(
        chunks=chunks,
        embeddings=embeddings,
        pipeline_name=PIPELINE_2,
        use_qdrant=False,
    )

    # Run pipeline
    question = "How do I set up n8n for the first time?"
    print(f"\nQuestion: {question}\n")

    result = await run_pipeline_2(vectorstore, question)

    print(f"Pipeline: {result['pipeline_name']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nRetrieved {len(result['retrieved_chunks'])} chunks after reranking")
    print(f"\nFirst chunk preview:\n{result['retrieved_chunks'][0][:200]}...")

asyncio.run(main())
