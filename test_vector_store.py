import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from backend.utils.document_processor import DocumentProcessor
from backend.utils.vector_store_manager import (
    VectorStoreManager,
    PIPELINE_1,
    PIPELINE_3,
)

load_dotenv()

PDF_PATH = "data/sample_docs/n8n guide.pdf"

# Load and chunk
processor = DocumentProcessor()
docs = processor.load_pdf(PDF_PATH)
fixed_chunks = processor.chunk_fixed(docs)

# Create embeddings
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# Create vector store manager
manager = VectorStoreManager()

# Test ChromaDB with Pipeline 1 collection
print("\n--- Testing ChromaDB ---")
vectorstore = manager.get_or_create_collection(
    chunks=fixed_chunks,
    embeddings=embeddings,
    pipeline_name=PIPELINE_1,
    use_qdrant=False,
)

# Run a test query
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("What is n8n used for?")

print(f"\nTest query returned {len(results)} chunks:")
for i, doc in enumerate(results):
    print(f"\nChunk {i+1} ({len(doc.page_content)} chars):")
    print(doc.page_content[:200] + "...")
