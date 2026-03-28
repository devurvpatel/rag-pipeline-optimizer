"""
Tests for Pipeline 1 — Fixed 512 + OpenAI + No Rerank.
All external API calls are mocked to run without real API keys.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.documents import Document
from backend.pipelines.pipeline_1 import build_pipeline_1, run_pipeline_1


# ── Fixtures ───────────────────────────────────────────────────────────────────
def make_mock_vectorstore():
    """Create a mock vectorstore that returns fake chunks."""
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="n8n can be set up via cloud or self-hosting.", metadata={}),
        Document(page_content="Self-hosting n8n requires Docker or npm.", metadata={}),
        Document(page_content="n8n Cloud offers a managed experience.", metadata={}),
        Document(page_content="Configuration requires setting environment variables.", metadata={}),
    ]
    mock_vs.as_retriever.return_value = mock_retriever
    return mock_vs


# ── Pipeline 1 Build Tests ─────────────────────────────────────────────────────
class TestBuildPipeline1:

    def test_build_pipeline_returns_chain(self):
        """build_pipeline_1 should return a runnable chain."""
        mock_vs = make_mock_vectorstore()
        with patch("backend.pipelines.pipeline_1.ChatOpenAI"):
            chain = build_pipeline_1(mock_vs)
            assert chain is not None

    def test_build_pipeline_uses_similarity_search(self):
        """Pipeline 1 should use similarity search retriever with k=4."""
        mock_vs = make_mock_vectorstore()
        with patch("backend.pipelines.pipeline_1.ChatOpenAI"):
            build_pipeline_1(mock_vs)
            mock_vs.as_retriever.assert_called_once_with(
                search_type="similarity",
                search_kwargs={"k": 4},
            )


# ── Pipeline 1 Run Tests ───────────────────────────────────────────────────────
class TestRunPipeline1:

    @pytest.mark.asyncio
    async def test_run_pipeline_returns_dict(self):
        """run_pipeline_1 should return a dict with required keys."""
        mock_vs = make_mock_vectorstore()

        with patch("backend.pipelines.pipeline_1.build_pipeline_1") as mock_build:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value="You can set up n8n via cloud or self-hosting.")
            mock_build.return_value = mock_chain

            with patch("langchain_community.callbacks.get_openai_callback") as mock_cb:
                mock_context = MagicMock()
                mock_context.total_cost = 0.0001
                mock_context.total_tokens = 500
                mock_cb.return_value.__enter__ = MagicMock(return_value=mock_context)
                mock_cb.return_value.__exit__ = MagicMock(return_value=False)

                result = await run_pipeline_1(mock_vs, "How do I set up n8n?")

                assert isinstance(result, dict)
                assert "pipeline_name" in result
                assert "answer" in result
                assert "retrieved_chunks" in result
                assert "cost_usd" in result
                assert "tokens_used" in result

    @pytest.mark.asyncio
    async def test_run_pipeline_returns_correct_pipeline_name(self):
        """Pipeline name should identify Pipeline 1."""
        mock_vs = make_mock_vectorstore()

        with patch("backend.pipelines.pipeline_1.build_pipeline_1") as mock_build:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value="Answer")
            mock_build.return_value = mock_chain

            with patch("langchain_community.callbacks.get_openai_callback") as mock_cb:
                mock_context = MagicMock()
                mock_context.total_cost = 0.0001
                mock_context.total_tokens = 500
                mock_cb.return_value.__enter__ = MagicMock(return_value=mock_context)
                mock_cb.return_value.__exit__ = MagicMock(return_value=False)

                result = await run_pipeline_1(mock_vs, "test question")
                assert "Pipeline 1" in result["pipeline_name"]

    @pytest.mark.asyncio
    async def test_run_pipeline_retrieves_chunks(self):
        """run_pipeline_1 should return retrieved chunks as list of strings."""
        mock_vs = make_mock_vectorstore()

        with patch("backend.pipelines.pipeline_1.build_pipeline_1") as mock_build:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value="Answer")
            mock_build.return_value = mock_chain

            with patch("langchain_community.callbacks.get_openai_callback") as mock_cb:
                mock_context = MagicMock()
                mock_context.total_cost = 0.0001
                mock_context.total_tokens = 500
                mock_cb.return_value.__enter__ = MagicMock(return_value=mock_context)
                mock_cb.return_value.__exit__ = MagicMock(return_value=False)

                result = await run_pipeline_1(mock_vs, "test question")
                assert isinstance(result["retrieved_chunks"], list)
                assert len(result["retrieved_chunks"]) == 4
                assert all(isinstance(c, str) for c in result["retrieved_chunks"])