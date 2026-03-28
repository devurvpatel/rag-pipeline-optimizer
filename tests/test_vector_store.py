"""
Tests for VectorStoreManager — runs without real API keys.
All embedding and vector store operations are mocked.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from backend.utils.vector_store_manager import (
    VectorStoreManager,
    PIPELINE_1,
    PIPELINE_2,
    PIPELINE_3,
    PIPELINE_4,
    ALL_PIPELINES,
)


# ── Sample chunks for testing ──────────────────────────────────────────────────
def make_sample_chunks():
    return [
        Document(page_content="n8n is a workflow automation tool.", metadata={"source": "test.pdf"}),
        Document(page_content="You can self-host n8n using Docker.", metadata={"source": "test.pdf"}),
        Document(page_content="n8n supports hundreds of integrations.", metadata={"source": "test.pdf"}),
    ]


# ── Constants Tests ────────────────────────────────────────────────────────────
class TestPipelineConstants:

    def test_pipeline_constants_exist(self):
        """All 4 pipeline collection name constants should be defined."""
        assert PIPELINE_1 is not None
        assert PIPELINE_2 is not None
        assert PIPELINE_3 is not None
        assert PIPELINE_4 is not None

    def test_pipeline_constants_are_strings(self):
        """Pipeline constants should be strings."""
        assert isinstance(PIPELINE_1, str)
        assert isinstance(PIPELINE_2, str)
        assert isinstance(PIPELINE_3, str)
        assert isinstance(PIPELINE_4, str)

    def test_pipeline_constants_are_unique(self):
        """Each pipeline should have a unique collection name."""
        names = [PIPELINE_1, PIPELINE_2, PIPELINE_3, PIPELINE_4]
        assert len(names) == len(set(names))

    def test_all_pipelines_list_has_four_items(self):
        """ALL_PIPELINES should contain exactly 4 pipeline names."""
        assert len(ALL_PIPELINES) == 4

    def test_all_pipelines_contains_all_constants(self):
        """ALL_PIPELINES should contain all 4 pipeline constants."""
        assert PIPELINE_1 in ALL_PIPELINES
        assert PIPELINE_2 in ALL_PIPELINES
        assert PIPELINE_3 in ALL_PIPELINES
        assert PIPELINE_4 in ALL_PIPELINES


# ── VectorStoreManager Initialization Tests ────────────────────────────────────
class TestVectorStoreManagerInit:

    def test_manager_initializes(self):
        """VectorStoreManager should initialize without errors."""
        manager = VectorStoreManager()
        assert manager is not None

    def test_manager_reads_qdrant_env_vars(self):
        """Manager should read QDRANT_URL and QDRANT_API_KEY from environment."""
        with patch.dict("os.environ", {
            "QDRANT_URL": "https://test.qdrant.io",
            "QDRANT_API_KEY": "test-key"
        }):
            manager = VectorStoreManager()
            assert manager.qdrant_url == "https://test.qdrant.io"
            assert manager.qdrant_api_key == "test-key"


# ── ChromaDB Collection Tests ──────────────────────────────────────────────────
class TestChromaDBCollections:

    @patch("backend.utils.vector_store_manager.Chroma")
    def test_create_chroma_collection_returns_vectorstore(self, mock_chroma):
        """create_chroma_collection should return a Chroma vectorstore."""
        mock_vs = MagicMock()
        mock_chroma.from_documents.return_value = mock_vs

        manager = VectorStoreManager()
        chunks = make_sample_chunks()
        embeddings = MagicMock()

        result = manager.create_chroma_collection(
            chunks=chunks,
            embeddings=embeddings,
            collection_name=PIPELINE_1,
        )

        assert result == mock_vs
        mock_chroma.from_documents.assert_called_once()

    @patch("backend.utils.vector_store_manager.Chroma")
    def test_create_chroma_collection_uses_correct_name(self, mock_chroma):
        """ChromaDB collection should be created with the given collection name."""
        mock_chroma.from_documents.return_value = MagicMock()

        manager = VectorStoreManager()
        chunks = make_sample_chunks()
        embeddings = MagicMock()

        manager.create_chroma_collection(
            chunks=chunks,
            embeddings=embeddings,
            collection_name=PIPELINE_1,
        )

        call_kwargs = mock_chroma.from_documents.call_args[1]
        assert call_kwargs["collection_name"] == PIPELINE_1

    @patch("backend.utils.vector_store_manager.Chroma")
    def test_create_chroma_uses_default_persist_dir(self, mock_chroma):
        """ChromaDB should use ./chroma_db as default persist directory."""
        mock_chroma.from_documents.return_value = MagicMock()

        manager = VectorStoreManager()
        manager.create_chroma_collection(
            chunks=make_sample_chunks(),
            embeddings=MagicMock(),
            collection_name=PIPELINE_1,
        )

        call_kwargs = mock_chroma.from_documents.call_args[1]
        assert call_kwargs["persist_directory"] == "./chroma_db"


# ── get_or_create_collection Tests ────────────────────────────────────────────
class TestGetOrCreateCollection:

    @patch("backend.utils.vector_store_manager.Chroma")
    def test_get_or_create_uses_chroma_by_default(self, mock_chroma):
        """get_or_create_collection should use ChromaDB when use_qdrant=False."""
        mock_chroma.from_documents.return_value = MagicMock()

        manager = VectorStoreManager()
        manager.get_or_create_collection(
            chunks=make_sample_chunks(),
            embeddings=MagicMock(),
            pipeline_name=PIPELINE_1,
            use_qdrant=False,
        )

        mock_chroma.from_documents.assert_called_once()

    def test_get_or_create_raises_for_invalid_pipeline(self):
        """get_or_create_collection should raise ValueError for unknown pipeline."""
        manager = VectorStoreManager()

        with pytest.raises(ValueError):
            manager.get_or_create_collection(
                chunks=make_sample_chunks(),
                embeddings=MagicMock(),
                pipeline_name="invalid_pipeline_name",
                use_qdrant=False,
            )

    @patch("backend.utils.vector_store_manager.Chroma")
    def test_get_or_create_accepts_all_pipeline_names(self, mock_chroma):
        """get_or_create_collection should accept all 4 valid pipeline names."""
        mock_chroma.from_documents.return_value = MagicMock()

        manager = VectorStoreManager()

        for pipeline_name in ALL_PIPELINES:
            result = manager.get_or_create_collection(
                chunks=make_sample_chunks(),
                embeddings=MagicMock(),
                pipeline_name=pipeline_name,
                use_qdrant=False,
            )
            assert result is not None