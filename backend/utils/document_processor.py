import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()


class DocumentProcessor:
    """
    Handles PDF loading and chunking for the RAG pipeline optimizer.
    Supports three chunking strategies: fixed, recursive, and semantic.
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and return a list of Document objects.
        Each page becomes one Document.

        Args:
            file_path: Full path to the PDF file

        Returns:
            List of Document objects with page content and metadata
        """
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {file_path}")
        return documents

    def chunk_fixed(
        self, docs: List[Document], chunk_size: int = 512
    ) -> List[Document]:
        """
        Split documents into fixed-size chunks regardless of content boundaries.
        Fast and simple but may cut sentences mid-thought.

        Args:
            docs: List of Document objects to split
            chunk_size: Number of characters per chunk (default 512)

        Returns:
            List of chunked Document objects
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = splitter.split_documents(docs)
        print(f"Fixed chunking: {len(chunks)} chunks at size {chunk_size}")
        return chunks

    def chunk_recursive(self, docs: List[Document]) -> List[Document]:
        """
        Split documents using a hierarchy of separators, trying to keep
        semantically related content together. Most widely used in production.

        Separator priority: paragraph breaks > line breaks > sentences > words

        Args:
            docs: List of Document objects to split

        Returns:
            List of chunked Document objects
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_documents(docs)
        print(f"Recursive chunking: {len(chunks)} chunks at size 1024")
        return chunks

    def chunk_semantic(self, docs: List[Document]) -> List[Document]:
        """
        Split documents based on semantic similarity between sentences.
        Uses embeddings to detect when meaning changes significantly and
        splits at those boundaries. Higher quality but more expensive.

        Args:
            docs: List of Document objects to split

        Returns:
            List of chunked Document objects
        """
        splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )
        chunks = splitter.split_documents(docs)
        print(f"Semantic chunking: {len(chunks)} chunks")
        return chunks

    def get_chunk_stats(self, chunks: List[Document]) -> Dict:
        """
        Calculate statistics about a list of chunks.
        Useful for comparing chunking strategies side by side.

        Args:
            chunks: List of chunked Document objects

        Returns:
            Dict with count, avg_length, min_length, max_length
        """
        if not chunks:
            return {
                "count": 0,
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0,
            }

        lengths = [len(chunk.page_content) for chunk in chunks]

        return {
            "count": len(chunks),
            "avg_length": round(sum(lengths) / len(lengths), 1),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }
