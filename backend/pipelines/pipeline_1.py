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
    Each chunk is separated by a double newline for readability.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def build_pipeline_1(vectorstore: VectorStore):
    """
    Pipeline 1 — Baseline configuration.
    Fixed 512-token chunking + OpenAI embeddings + similarity search, no reranking.
    This is the control group — the simplest possible RAG implementation.
    Every other pipeline is measured against this one.

    Args:
        vectorstore: A pre-populated Chroma or Qdrant vectorstore

    Returns:
        LCEL chain that accepts a question string and returns an answer string
    """
    # Retrieve top 4 most similar chunks by cosine similarity
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # LCEL chain — data flows left to right through the pipe operator
    # Step 1: retriever fetches context, RunnablePassthrough passes question unchanged
    # Step 2: prompt formats both into a message
    # Step 3: llm generates a response
    # Step 4: parser extracts the string content from the response object
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


async def run_pipeline_1(
    vectorstore: VectorStore,
    question: str,
) -> Dict:
    """
    Run Pipeline 1 and return a structured result dict.

    Args:
        vectorstore: Pre-populated vectorstore for this pipeline
        question: The user's question string

    Returns:
        Dict with keys: answer, retrieved_chunks, pipeline_name, cost_usd, tokens_used
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    retrieved_docs = retriever.invoke(question)
    retrieved_chunks = [doc.page_content for doc in retrieved_docs]

    chain = build_pipeline_1(vectorstore)

    try:
        from langchain_community.callbacks import get_openai_callback
        with get_openai_callback() as cb:
            answer = await chain.ainvoke(
                question,
                config={
                    "run_name": "Pipeline 1 — Fixed 512 + OpenAI + No Rerank",
                    "metadata": {
                        "pipeline_id": "pipeline_1",
                        "chunking": "fixed_512",
                        "embeddings": "openai_ada002",
                        "reranking": "none",
                        "question": question,
                    }
                }
            )
            actual_cost = round(cb.total_cost, 6)
            tokens_used = cb.total_tokens
    except ImportError:
        # Fallback if callback import fails
        answer = await chain.ainvoke(question)
        actual_cost = 0.0
        tokens_used = 0

    return {
        "pipeline_name": "Pipeline 1 — Fixed 512 + OpenAI + No Rerank",
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "cost_usd": actual_cost,
        "tokens_used": tokens_used,
    }