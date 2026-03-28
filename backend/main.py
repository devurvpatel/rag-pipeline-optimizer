import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv

# Load environment variables first — must happen before LangChain imports
# so LangSmith tracing is configured before any LangChain module initializes
load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag-pipeline-optimizer")

# LangChain imports after environment setup  # noqa: E402
from fastapi import FastAPI, UploadFile, File, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from langchain_openai import OpenAIEmbeddings  # noqa: E402
from langchain_cohere import CohereEmbeddings  # noqa: E402
from backend.utils.document_processor import DocumentProcessor  # noqa: E402
from backend.utils.vector_store_manager import (  # noqa: E402
    VectorStoreManager,
    PIPELINE_1, PIPELINE_2, PIPELINE_3, PIPELINE_4,
)
from backend.pipelines.pipeline_1 import build_pipeline_1, run_pipeline_1  # noqa: E402
from backend.pipelines.pipeline_2 import build_pipeline_2, run_pipeline_2  # noqa: E402
from backend.pipelines.pipeline_3 import build_pipeline_3, run_pipeline_3, get_bge_embeddings  # noqa: E402
from backend.pipelines.pipeline_4 import build_pipeline_4, run_pipeline_4  # noqa: E402
from backend.evaluation.ragas_eval import evaluate_pipeline, compare_pipelines  # noqa: E402
from backend.evaluation.langgraph_agent import run_evaluation  # noqa: E402

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Request/Response Models ────────────────────────────────────────────────────
class EvaluateRequest(BaseModel):
    question: str
    ground_truth: Optional[str] = ""


class EvaluateResponse(BaseModel):
    pipeline_results: dict
    ragas_scores: dict
    comparison_summary: str
    agent_report: dict


# ── Lifespan — Initialize all pipelines on startup ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup — initializes all embeddings and vector store
    collections, then builds all 4 pipeline chains.
    Stores everything in app.state so endpoints can access them.
    """
    logger.info("Starting RAG Pipeline Optimizer...")

    try:
        manager = VectorStoreManager()

        # Initialize embeddings for each pipeline
        logger.info("Loading embedding models...")
        openai_embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        cohere_embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=os.getenv("COHERE_API_KEY"),
        )
        bge_embeddings = get_bge_embeddings()

        # Load existing ChromaDB collections
        # Collections are created on first /upload call
        # On subsequent startups they load from chroma_db/
        logger.info("Loading vector store collections...")
        try:
            app.state.vectorstores = {
                PIPELINE_1: manager.load_chroma_collection(
                    openai_embeddings, PIPELINE_1
                ),
                PIPELINE_2: manager.load_chroma_collection(
                    cohere_embeddings, PIPELINE_2
                ),
                PIPELINE_3: manager.load_chroma_collection(
                    bge_embeddings, PIPELINE_3
                ),
                PIPELINE_4: manager.load_chroma_collection(
                    openai_embeddings, PIPELINE_4
                ),
            }
            logger.info("Vector store collections loaded successfully")

            # Build pipeline chains using loaded vectorstores
            logger.info("Building pipeline chains...")
            app.state.chains = {
                PIPELINE_1: build_pipeline_1(app.state.vectorstores[PIPELINE_1]),
                PIPELINE_2: build_pipeline_2(app.state.vectorstores[PIPELINE_2]),
                PIPELINE_3: build_pipeline_3(app.state.vectorstores[PIPELINE_3]),
                PIPELINE_4: build_pipeline_4(app.state.vectorstores[PIPELINE_4]),
            }
            logger.info("All 4 pipeline chains ready")

        except Exception as e:
            # Collections don't exist yet — first run before any upload
            logger.warning(
                f"Collections not found — upload a document first: {e}"
            )
            app.state.vectorstores = {}
            app.state.chains = {}

        # Store embedding models for reuse in /upload endpoint
        app.state.embeddings = {
            "openai": openai_embeddings,
            "cohere": cohere_embeddings,
            "bge": bge_embeddings,
        }
        app.state.manager = manager
        app.state.processor = DocumentProcessor()

        logger.info("Startup complete — RAG Pipeline Optimizer is ready")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield  # App runs here

    logger.info("Shutting down RAG Pipeline Optimizer...")


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Pipeline Optimizer",
    description="Benchmarks 4 RAG pipeline configurations and recommends the optimal one",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Quick check that the API is running."""
    chains_ready = len(app.state.chains) == 4
    return {
        "status": "ok",
        "chains_ready": chains_ready,
        "pipelines_loaded": list(app.state.chains.keys()),
    }


# ── Upload Endpoint ────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accept a PDF upload, process it through all 3 chunking strategies,
    store each chunk set in its respective ChromaDB collection,
    and return stats comparing the 3 strategies.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    logger.info(f"Uploading document: {file.filename}")

    # Save uploaded file
    save_path = f"data/sample_docs/{file.filename}"
    os.makedirs("data/sample_docs", exist_ok=True)

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    logger.info(f"File saved to {save_path}")

    try:
        processor = app.state.processor
        manager = app.state.manager
        embeddings = app.state.embeddings

        # Load PDF
        docs = processor.load_pdf(save_path)
        logger.info(f"Loaded {len(docs)} pages")

        # Chunk with all 3 strategies
        logger.info("Running 3 chunking strategies...")
        fixed_chunks = processor.chunk_fixed(docs, chunk_size=512)
        recursive_chunks = processor.chunk_recursive(docs)

        logger.info("Running semantic chunking (may take 1-2 minutes)...")
        semantic_chunks = processor.chunk_semantic(docs)

        # Store in ChromaDB — one collection per pipeline
        logger.info("Creating vector store collections...")

        vs1 = manager.get_or_create_collection(
            fixed_chunks, embeddings["openai"], PIPELINE_1
        )
        vs2 = manager.get_or_create_collection(
            recursive_chunks, embeddings["cohere"], PIPELINE_2
        )
        vs3 = manager.get_or_create_collection(
            semantic_chunks, embeddings["bge"], PIPELINE_3
        )
        vs4 = manager.get_or_create_collection(
            fixed_chunks, embeddings["openai"], PIPELINE_4
        )

        # Update app state with new vectorstores and rebuild chains
        app.state.vectorstores = {
            PIPELINE_1: vs1,
            PIPELINE_2: vs2,
            PIPELINE_3: vs3,
            PIPELINE_4: vs4,
        }

        from backend.pipelines.pipeline_1 import build_pipeline_1
        from backend.pipelines.pipeline_2 import build_pipeline_2
        from backend.pipelines.pipeline_3 import build_pipeline_3
        from backend.pipelines.pipeline_4 import build_pipeline_4

        app.state.chains = {
            PIPELINE_1: build_pipeline_1(vs1),
            PIPELINE_2: build_pipeline_2(vs2),
            PIPELINE_3: build_pipeline_3(vs3),
            PIPELINE_4: build_pipeline_4(vs4),
        }

        logger.info("All collections created and chains rebuilt")

        # Return chunking stats for comparison
        return {
            "status": "success",
            "filename": file.filename,
            "pages": len(docs),
            "chunking_stats": {
                "fixed_512": processor.get_chunk_stats(fixed_chunks),
                "recursive_1024": processor.get_chunk_stats(recursive_chunks),
                "semantic": processor.get_chunk_stats(semantic_chunks),
            },
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ── Evaluate Endpoint ──────────────────────────────────────────────────────────
@app.post("/evaluate")
async def evaluate_all_pipelines(request: EvaluateRequest):
    """
    Run all 4 pipelines on a question in parallel, evaluate with RAGAS,
    run the LangGraph agent, and return the full comparison.
    """
    if not app.state.chains:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Call /upload first."
        )

    logger.info(f"Evaluating question: {request.question[:60]}...")

    try:
        vectorstores = app.state.vectorstores

        # Run all 4 pipelines in parallel
        logger.info("Running all 4 pipelines in parallel...")
        results = await asyncio.gather(
            run_pipeline_1(vectorstores[PIPELINE_1], request.question),
            run_pipeline_2(vectorstores[PIPELINE_2], request.question),
            run_pipeline_3(vectorstores[PIPELINE_3], request.question),
            run_pipeline_4(vectorstores[PIPELINE_4], request.question),
            return_exceptions=True,
        )

        # Separate successful results from errors
        pipeline_results = {}
        valid_results = []

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Pipeline error: {result}")
            else:
                pipeline_results[result["pipeline_name"]] = {
                    "answer": result["answer"],
                    "chunks_retrieved": len(result["retrieved_chunks"]),
                    "cost_usd": result.get("cost_usd", 0),
                    "tokens_used": result.get("tokens_used", 0),
                }
                valid_results.append(result)

        if not valid_results:
            raise HTTPException(
                status_code=500,
                detail="All pipelines failed to produce results"
            )

        # Run RAGAS evaluation if ground truth is provided
        ragas_scores = {}
        comparison_summary = ""
        agent_report = {}

        if request.ground_truth:
            logger.info("Running RAGAS evaluation...")

            all_ragas_results = {}

            for result in valid_results:
                scores = await evaluate_pipeline(
                    pipeline_name=result["pipeline_name"],
                    questions=[request.question],
                    answers=[result["answer"]],
                    contexts=[result["retrieved_chunks"]],
                    ground_truths=[request.ground_truth],
                )
                ragas_scores[result["pipeline_name"]] = scores
                all_ragas_results[result["pipeline_name"]] = scores

            # Compare pipelines
            comparison_summary = compare_pipelines(all_ragas_results)

            # Run LangGraph evaluator agent
            logger.info("Running LangGraph evaluator agent...")

            # Collect live costs from pipeline results
            live_costs = {
                result["pipeline_name"]: {
                    "cost_usd_this_query": result.get("cost_usd", 0),
                    "tokens_used": result.get("tokens_used", 0),
                    "embedding_model": ["OpenAI ada-002", "Cohere embed-v3", "BGE-large (free)", "OpenAI ada-002"][i],
                    "reranking": ["None", "Cohere Rerank ($2/1k)", "Cross-Encoder (free)", "MMR (free)"][i],
                }
                for i, result in enumerate(valid_results)
            }

            # Run LangGraph evaluator agent with live costs
            agent_report = run_evaluation(all_ragas_results, live_costs)

        else:
            logger.info("No ground truth provided — skipping RAGAS evaluation")
            comparison_summary = "Provide a ground truth answer to enable RAGAS evaluation"

        return {
            "question": request.question,
            "pipeline_results": pipeline_results,
            "ragas_scores": ragas_scores,
            "comparison_summary": comparison_summary,
            "agent_report": agent_report,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# ── LangServe Routes ───────────────────────────────────────────────────────────
# These automatically add /invoke, /stream, /batch, /playground endpoints
# for each pipeline — gives you a free built-in UI at /pipeline1/playground
try:
    from langserve import add_routes

    @app.on_event("startup")
    async def add_langserve_routes():
        if app.state.chains:
            add_routes(app, app.state.chains[PIPELINE_1], path="/pipeline1")
            add_routes(app, app.state.chains[PIPELINE_2], path="/pipeline2")
            add_routes(app, app.state.chains[PIPELINE_3], path="/pipeline3")
            add_routes(app, app.state.chains[PIPELINE_4], path="/pipeline4")
            logger.info("LangServe routes added")

except ImportError:
    logger.warning("LangServe not available — skipping playground routes")

