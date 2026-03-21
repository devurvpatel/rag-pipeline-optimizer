import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
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


def build_pipeline_4(vectorstore: VectorStore):
    """
    Pipeline 4 — MMR (Maximal Marginal Relevance) retrieval.
    Fixed 1024-token chunking + OpenAI embeddings + MMR retrieval strategy.

    How MMR differs from standard similarity search (Pipeline 1):
    - Standard similarity: returns the 4 most similar chunks — often returns
      4 chunks that all say nearly the same thing (redundant context)
    - MMR: fetches 20 candidates then iteratively selects chunks that are
      BOTH relevant to the query AND different from already-selected chunks
    - Result: broader coverage of the document, less repetition in context

    lambda_mult controls the relevance/diversity tradeoff:
    - lambda_mult=0.0 → maximum diversity (ignores relevance completely)
    - lambda_mult=1.0 → maximum relevance (identical to similarity search)
    - lambda_mult=0.5 → balanced (equal weight to relevance and diversity)

    Best used for: long documents where the answer spans multiple sections,
    or when similarity search returns repetitive chunks.

    Args:
        vectorstore: A pre-populated Chroma or Qdrant vectorstore

    Returns:
        LCEL chain that accepts a question string and returns an answer string
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,          # Final number of chunks to return
            "fetch_k": 20,   # Number of candidates to fetch before MMR filtering
            "lambda_mult": 0.5,  # 0=max diversity, 1=max relevance, 0.5=balanced
        },
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


async def run_pipeline_4(
    vectorstore: VectorStore,
    question: str,
) -> Dict:
    """
    Run Pipeline 4 and return a structured result dict.

    Args:
        vectorstore: Pre-populated vectorstore for this pipeline
        question: The user's question string

    Returns:
        Dict with keys: answer, retrieved_chunks, pipeline_name
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        },
    )

    retrieved_docs = retriever.invoke(question)
    retrieved_chunks = [doc.page_content for doc in retrieved_docs]

    chain = build_pipeline_4(vectorstore)

    with get_openai_callback() as cb:
        answer = await chain.ainvoke(
            question,
            config={
                "run_name": "Pipeline 4 — Fixed 1024 + OpenAI + MMR Retrieval",
                "metadata": {
                    "pipeline_id": "pipeline_4",
                    "chunking": "fixed_1024",
                    "embeddings": "openai_ada002",
                    "reranking": "mmr",
                    "question": question,
                }
            }
        )
        actual_cost = round(cb.total_cost, 6)
        tokens_used = cb.total_tokens

    return {
        "pipeline_name": "Pipeline 4 — Fixed 1024 + OpenAI + MMR Retrieval",
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "cost_usd": actual_cost,
        "tokens_used": tokens_used,
    }
