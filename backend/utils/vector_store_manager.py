import os
from typing import List
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

load_dotenv()

# ── Collection name constants ──────────────────────────────────────────────────
PIPELINE_1 = "pipeline_1_fixed_openai"
PIPELINE_2 = "pipeline_2_recursive_cohere"
PIPELINE_3 = "pipeline_3_semantic_bge"
PIPELINE_4 = "pipeline_4_fixed_mmr"

ALL_PIPELINES = [PIPELINE_1, PIPELINE_2, PIPELINE_3, PIPELINE_4]


class VectorStoreManager:
    """
    Manages vector store collections for the RAG pipeline optimizer.
    Supports ChromaDB for local development and Qdrant for production.
    Each pipeline gets its own isolated collection to prevent
    cross-contamination between different embedding models.
    """

    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

    def create_chroma_collection(
        self,
        chunks: List[Document],
        embeddings: Embeddings,
        collection_name: str,
        persist_dir: str = "./chroma_db",
    ) -> Chroma:
        """
        Create or overwrite a ChromaDB collection from a list of chunks.
        ChromaDB runs locally with zero setup — ideal for development.

        Args:
            chunks: List of Document objects to embed and store
            embeddings: Embedding model to use for vectorization
            collection_name: Unique name for this collection
            persist_dir: Local directory to persist ChromaDB data

        Returns:
            Chroma vectorstore instance ready for retrieval
        """
        try:
            print(f"Creating ChromaDB collection: {collection_name}")

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=persist_dir,
            )

            print(f"ChromaDB collection '{collection_name}' created with {len(chunks)} chunks")
            return vectorstore

        except Exception as e:
            raise RuntimeError(
                f"Failed to create ChromaDB collection '{collection_name}': {e}"
            )

    def create_qdrant_collection(
        self,
        chunks: List[Document],
        embeddings: Embeddings,
        collection_name: str,
    ) -> QdrantVectorStore:
        """
        Create or overwrite a Qdrant cloud collection from a list of chunks.
        Qdrant is production-grade with better performance at scale.
        Reads QDRANT_URL and QDRANT_API_KEY from environment variables.

        Args:
            chunks: List of Document objects to embed and store
            embeddings: Embedding model to use for vectorization
            collection_name: Unique name for this collection

        Returns:
            QdrantVectorStore instance ready for retrieval
        """
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY must be set in your .env file"
            )

        try:
            print(f"Creating Qdrant collection: {collection_name}")

            vectorstore = QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                collection_name=collection_name,
                force_recreate=True,
            )

            print(f"Qdrant collection '{collection_name}' created with {len(chunks)} chunks")
            return vectorstore

        except Exception as e:
            raise RuntimeError(
                f"Failed to create Qdrant collection '{collection_name}': {e}"
            )

    def get_or_create_collection(
        self,
        chunks: List[Document],
        embeddings: Embeddings,
        pipeline_name: str,
        use_qdrant: bool = False,
    ) -> object:
        """
        Unified method to create a collection for a specific pipeline.
        Routes to ChromaDB or Qdrant based on the use_qdrant flag.

        Args:
            chunks: List of Document objects to embed and store
            embeddings: Embedding model instance
            pipeline_name: One of the PIPELINE_1 through PIPELINE_4 constants
            use_qdrant: If True uses Qdrant cloud, otherwise uses local ChromaDB

        Returns:
            Vectorstore instance (Chroma or QdrantVectorStore)
        """
        if pipeline_name not in ALL_PIPELINES:
            raise ValueError(
                f"pipeline_name must be one of {ALL_PIPELINES}, got '{pipeline_name}'"
            )

        if use_qdrant:
            return self.create_qdrant_collection(
                chunks=chunks,
                embeddings=embeddings,
                collection_name=pipeline_name,
            )
        else:
            return self.create_chroma_collection(
                chunks=chunks,
                embeddings=embeddings,
                collection_name=pipeline_name,
            )

    def load_chroma_collection(
        self,
        embeddings: Embeddings,
        collection_name: str,
        persist_dir: str = "./chroma_db",
    ) -> Chroma:
        """
        Load an existing ChromaDB collection without re-embedding.
        Use this after initial setup to avoid re-processing documents.

        Args:
            embeddings: Same embedding model used during creation
            collection_name: Name of the existing collection
            persist_dir: Directory where ChromaDB data was persisted

        Returns:
            Chroma vectorstore instance ready for retrieval
        """
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_dir,
            )
            print(f"Loaded existing ChromaDB collection: {collection_name}")
            return vectorstore

        except Exception as e:
            raise RuntimeError(
                f"Failed to load ChromaDB collection '{collection_name}': {e}"
            )
