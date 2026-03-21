import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever

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


def build_pipeline_2(vectorstore: VectorStore):
    """
    Pipeline 2 — Cohere embeddings + Cohere neural reranking.
    Recursive 1024-token chunking + Cohere embeddings + Cohere Rerank v3.

    How reranking works here:
    - Step 1: Base retriever fetches top 10 chunks by cosine similarity
    - Step 2: CohereRerank re-scores all 10 using a cross-encoder model
              that reads the query AND each chunk together (much more accurate
              than cosine similarity alone)
    - Step 3: Only the top 4 reranked chunks are passed to the LLM
    We retrieve 10 then rerank to 4 because cosine similarity sometimes
    misses relevant chunks — the reranker catches what similarity search missed.

    Args:
        vectorstore: A pre-populated Chroma or Qdrant vectorstore

    Returns:
        LCEL chain that accepts a question string and returns an answer string
    """
    # Retrieve more candidates than needed — reranker will filter down
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

    # Cohere reranker re-scores the 10 candidates and keeps only top 4
    reranker = CohereRerank(
        model="rerank-english-v3.0",
        top_n=4,
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )

    # ContextualCompressionRetriever wraps base retriever + reranker together
    # It first calls base_retriever, then passes results through the reranker
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


async def run_pipeline_2(
    vectorstore: VectorStore,
    question: str,
) -> Dict:
    """
    Run Pipeline 2 and return a structured result dict.

    Args:
        vectorstore: Pre-populated vectorstore for this pipeline
        question: The user's question string

    Returns:
        Dict with keys: answer, retrieved_chunks, pipeline_name
    """
    # Build reranking retriever separately to fetch chunks for RAGAS
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

    reranker = CohereRerank(
        model="rerank-english-v3.0",
        top_n=4,
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )

    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )

    # Fetch reranked chunks for RAGAS evaluation
    retrieved_docs = retriever.invoke(question)
    retrieved_chunks = [doc.page_content for doc in retrieved_docs]

    # Run the full chain for the answer
    chain = build_pipeline_2(vectorstore)
    answer = await chain.ainvoke(question)

    return {
        "pipeline_name": "Pipeline 2 — Recursive 1024 + Cohere + Cohere Rerank",
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
    }
