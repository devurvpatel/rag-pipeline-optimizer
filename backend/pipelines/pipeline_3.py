import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.callbacks import get_openai_callback

load_dotenv()

# ── Prompt Template ────────────────────────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the following context.
If the answer is not in the context, say "I could not find this information in the provided document."

Context:
{context}

Question: {question}

Answer:""")


def format_docs(docs: List[Document]) -> str:
    """
    Combine retrieved Document objects into a single string for the prompt.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def get_bge_embeddings() -> HuggingFaceBgeEmbeddings:
    """
    Initialize BGE-large embeddings.

    BGE-large-en-v1.5 (Beijing Academy of AI) is a free, open-source
    embedding model that outperforms OpenAI text-embedding-ada-002 on the
    MTEB (Massive Text Embedding Benchmark) leaderboard. It runs locally
    with zero API cost, making this pipeline the cheapest to operate at
    scale (~$0.15 per 1000 queries vs $0.45 for OpenAI embeddings).

    normalize_embeddings=True is required for correct cosine similarity
    calculation with BGE models.
    """
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_pipeline_3(vectorstore: VectorStore):
    """
    Pipeline 3 — BGE embeddings + Cross-Encoder reranking (fully free).
    Semantic chunking + BGE-large embeddings + ms-marco cross-encoder rerank.

    How the cross-encoder reranker differs from Cohere reranker (Pipeline 2):
    - Both re-score retrieved chunks using a more powerful model
    - Cohere reranker is a paid API call per search
    - Cross-encoder runs locally on your machine — zero cost per query
    - ms-marco-MiniLM was trained on Microsoft's MARCO dataset (100M queries)
      making it highly effective for question-answering retrieval tasks

    We retrieve 10 then rerank to 4 — same strategy as Pipeline 2 but
    at a fraction of the operating cost.

    Args:
        vectorstore: A pre-populated Chroma or Qdrant vectorstore

    Returns:
        LCEL chain that accepts a question string and returns an answer string
    """
    # Retrieve more candidates than needed — cross-encoder filters down
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

    # Cross-encoder reads query + chunk together for deep relevance scoring
    # This is more accurate than cosine similarity but slower — acceptable
    # because it only scores 10 candidates, not the entire index
    cross_encoder = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    reranker = CrossEncoderReranker(
        model=cross_encoder,
        top_n=4,
    )

    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


async def run_pipeline_3(
    vectorstore: VectorStore,
    question: str,
) -> Dict:
    """
    Run Pipeline 3 and return a structured result dict.

    Args:
        vectorstore: Pre-populated vectorstore for this pipeline
        question: The user's question string

    Returns:
        Dict with keys: answer, retrieved_chunks, pipeline_name
    """
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

    cross_encoder = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    reranker = CrossEncoderReranker(
        model=cross_encoder,
        top_n=4,
    )

    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )

    retrieved_docs = retriever.invoke(question)
    retrieved_chunks = [doc.page_content for doc in retrieved_docs]

    chain = build_pipeline_3(vectorstore)

    # Note: BGE embeddings and cross-encoder are free (local)
    # callback only captures GPT-4o-mini LLM cost
    with get_openai_callback() as cb:
        answer = await chain.ainvoke(
            question,
            config={
                "run_name": "Pipeline 3 — Semantic + BGE + Cross-Encoder Rerank",
                "metadata": {
                    "pipeline_id": "pipeline_3",
                    "chunking": "semantic",
                    "embeddings": "bge_large_en_v1.5",
                    "reranking": "cross_encoder_ms_marco",
                    "question": question,
                }
            }
        )
        actual_cost = round(cb.total_cost, 6)
        tokens_used = cb.total_tokens

    return {
        "pipeline_name": "Pipeline 3 — Semantic + BGE + Cross-Encoder Rerank",
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "cost_usd": actual_cost,
        "tokens_used": tokens_used,
    }
