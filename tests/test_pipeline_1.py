import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from backend.utils.vector_store_manager import VectorStoreManager, PIPELINE_1
from backend.utils.document_processor import DocumentProcessor
from backend.pipelines.pipeline_1 import run_pipeline_1

load_dotenv()

async def main():
    # Load and chunk document
    processor = DocumentProcessor()
    docs = processor.load_pdf("data/sample_docs/n8n guide.pdf")
    chunks = processor.chunk_fixed(docs)

    # Load embeddings and vectorstore
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    manager = VectorStoreManager()

    # Load existing ChromaDB collection (already created in Phase 2)
    vectorstore = manager.load_chroma_collection(
        embeddings=embeddings,
        collection_name=PIPELINE_1,
    )

    # Run pipeline
    question = "How do I set up n8n for the first time?"
    print(f"\nQuestion: {question}\n")

    result = await run_pipeline_1(vectorstore, question)

    print(f"Pipeline: {result['pipeline_name']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nRetrieved {len(result['retrieved_chunks'])} chunks")
    print(f"\nFirst chunk preview:\n{result['retrieved_chunks'][0][:200]}...")

asyncio.run(main())
