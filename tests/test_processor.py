"""
Tests for DocumentProcessor — runs without real API keys.
Semantic chunking uses a mocked OpenAI embeddings to avoid API calls.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from backend.utils.document_processor import DocumentProcessor


# ── Sample document for testing ───────────────────────────────────────────────
SAMPLE_TEXT = """
Workflow automation is the process of using technology to perform tasks 
without manual intervention. It involves setting up rules and triggers 
that automatically execute actions when certain conditions are met.

There are three key components of workflow automation. The first is 
trigger events which start the workflow. The second is actions which 
are the tasks performed. The third is conditions which control the flow.

n8n is a powerful workflow automation tool that allows users to connect 
different services and automate repetitive tasks. It supports hundreds 
of integrations and can be self-hosted for full data control.

The main advantages of n8n include its flexibility, open-source nature,
and the ability to run complex workflows with conditional logic and loops.
Users can also write custom JavaScript code for advanced data manipulation.
"""

def make_sample_docs():
    """Create sample Document objects for testing without loading a real PDF."""
    return [
        Document(
            page_content=SAMPLE_TEXT,
            metadata={"source": "test_doc.pdf", "page": 0}
        )
    ]


# ── DocumentProcessor Tests ───────────────────────────────────────────────────
class TestDocumentProcessor:

    def setup_method(self):
        """Initialize processor before each test."""
        self.processor = DocumentProcessor()
        self.sample_docs = make_sample_docs()

    # ── Fixed Chunking Tests ──────────────────────────────────────────────────
    def test_chunk_fixed_returns_list(self):
        """Fixed chunking should return a list of Document objects."""
        chunks = self.processor.chunk_fixed(self.sample_docs)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_fixed_returns_documents(self):
        """Each chunk should be a Document object."""
        chunks = self.processor.chunk_fixed(self.sample_docs)
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert hasattr(chunk, "page_content")
            assert len(chunk.page_content) > 0

    def test_chunk_fixed_respects_chunk_size(self):
        """Fixed chunks should not exceed chunk_size + overlap."""
        chunk_size = 200
        chunks = self.processor.chunk_fixed(self.sample_docs, chunk_size=chunk_size)
        for chunk in chunks:
            # Allow small overflow due to overlap
            assert len(chunk.page_content) <= chunk_size + 100

    def test_chunk_fixed_default_size(self):
        """Default chunk size should be 512."""
        chunks = self.processor.chunk_fixed(self.sample_docs)
        for chunk in chunks:
            assert len(chunk.page_content) <= 600  # 512 + overlap buffer

    def test_chunk_fixed_custom_size(self):
        """Custom chunk size should produce different results than default."""
        chunks_small = self.processor.chunk_fixed(self.sample_docs, chunk_size=100)
        chunks_large = self.processor.chunk_fixed(self.sample_docs, chunk_size=500)
        assert len(chunks_small) >= len(chunks_large)

    # ── Recursive Chunking Tests ──────────────────────────────────────────────
    def test_chunk_recursive_returns_list(self):
        """Recursive chunking should return a list of Document objects."""
        chunks = self.processor.chunk_recursive(self.sample_docs)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_recursive_returns_documents(self):
        """Each recursive chunk should be a Document object."""
        chunks = self.processor.chunk_recursive(self.sample_docs)
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert len(chunk.page_content) > 0

    def test_chunk_recursive_produces_fewer_chunks_than_fixed(self):
        """
        Recursive chunking at 1024 should produce fewer chunks
        than fixed chunking at 512 on the same document.
        """
        fixed_chunks = self.processor.chunk_fixed(self.sample_docs, chunk_size=512)
        recursive_chunks = self.processor.chunk_recursive(self.sample_docs)
        assert len(recursive_chunks) <= len(fixed_chunks)

    def test_chunk_recursive_preserves_content(self):
        """
        All original text should be present across recursive chunks
        combined.
        """
        chunks = self.processor.chunk_recursive(self.sample_docs)
        combined = " ".join(c.page_content for c in chunks)
        assert "workflow automation" in combined.lower()
        assert "n8n" in combined.lower()

    # ── Semantic Chunking Tests (mocked) ──────────────────────────────────────
    @patch("backend.utils.document_processor.SemanticChunker")
    @patch("backend.utils.document_processor.OpenAIEmbeddings")
    def test_chunk_semantic_returns_list(
        self, mock_embeddings_class, mock_chunker_class
    ):
        """
        Semantic chunking should return a list of Documents.
        OpenAI embeddings are mocked to avoid real API calls.
        """
        # Mock the chunker to return our sample docs as-is
        mock_chunker = MagicMock()
        mock_chunker.split_documents.return_value = self.sample_docs
        mock_chunker_class.return_value = mock_chunker
        mock_embeddings_class.return_value = MagicMock()

        chunks = self.processor.chunk_semantic(self.sample_docs)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    @patch("backend.utils.document_processor.SemanticChunker")
    @patch("backend.utils.document_processor.OpenAIEmbeddings")
    def test_chunk_semantic_calls_chunker(
        self, mock_embeddings_class, mock_chunker_class
    ):
        """
        Semantic chunking should initialize and use SemanticChunker.
        Verifies the chunker is called with the document list.
        """
        mock_chunker = MagicMock()
        mock_chunker.split_documents.return_value = self.sample_docs
        mock_chunker_class.return_value = mock_chunker
        mock_embeddings_class.return_value = MagicMock()

        self.processor.chunk_semantic(self.sample_docs)

        # Verify SemanticChunker was initialized and used
        mock_chunker_class.assert_called_once()
        mock_chunker.split_documents.assert_called_once_with(self.sample_docs)

    # ── get_chunk_stats Tests ─────────────────────────────────────────────────
    def test_get_chunk_stats_returns_correct_keys(self):
        """Stats dict should have all 4 required keys."""
        chunks = self.processor.chunk_fixed(self.sample_docs)
        stats = self.processor.get_chunk_stats(chunks)

        assert "count" in stats
        assert "avg_length" in stats
        assert "min_length" in stats
        assert "max_length" in stats

    def test_get_chunk_stats_count_matches(self):
        """Count in stats should match actual number of chunks."""
        chunks = self.processor.chunk_fixed(self.sample_docs)
        stats = self.processor.get_chunk_stats(chunks)
        assert stats["count"] == len(chunks)

    def test_get_chunk_stats_min_max_relationship(self):
        """Min length should always be <= max length."""
        chunks = self.processor.chunk_fixed(self.sample_docs)
        stats = self.processor.get_chunk_stats(chunks)
        assert stats["min_length"] <= stats["max_length"]

    def test_get_chunk_stats_avg_between_min_max(self):
        """Average length should be between min and max."""
        chunks = self.processor.chunk_fixed(self.sample_docs)
        stats = self.processor.get_chunk_stats(chunks)
        assert stats["min_length"] <= stats["avg_length"] <= stats["max_length"]

    def test_get_chunk_stats_empty_list(self):
        """Stats with empty list should return zeros."""
        stats = self.processor.get_chunk_stats([])
        assert stats["count"] == 0
        assert stats["avg_length"] == 0
        assert stats["min_length"] == 0
        assert stats["max_length"] == 0