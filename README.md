# 🔍 RAG Pipeline Optimizer

A production-grade benchmarking system that runs 4 RAG pipeline configurations simultaneously, evaluates them scientifically using the RAGAS framework, and uses a LangGraph agent to recommend the optimal configuration for your specific documents. This is an MLOps project — not a RAG chatbot demo.

---

## 🎯 Why This Project Is Different

Most RAG portfolio projects show you a chatbot that answers questions. This project asks a harder question: **which RAG configuration actually works best for your data?**

The analogy for interviewers: this is A/B testing for AI retrieval systems. Companies spend months guessing whether their RAG is good. This system tells them in minutes by running 4 pipelines in parallel, scoring each on faithfulness, answer relevancy, context precision, and context recall using the industry-standard RAGAS evaluation framework — then having a LangGraph agent synthesize the scores and cost data into a structured recommendation.

---

## 🎬 Demo

![RAG Pipeline Optimizer Demo](assets/demo.mp4)

---

## 🏗️ Architecture

```
User (Streamlit UI)
        │
        ▼
  Upload Document + Ask Question
        │
        ▼
FastAPI + LangServe Backend
        │
        ├─────────────────────────────────────────┐
        │                                         │
        ▼                                         ▼
4 RAG Pipelines (parallel)              LangSmith Tracing
        │
        ├── Pipeline 1: Fixed 512 + OpenAI embeddings + similarity search
        ├── Pipeline 2: Recursive 1024 + Cohere embeddings + Cohere Rerank
        ├── Pipeline 3: Semantic chunking + BGE-large + Cross-Encoder Rerank
        └── Pipeline 4: Fixed 1024 + OpenAI embeddings + MMR retrieval
        │
        ▼
RAGAS Evaluation Framework
(faithfulness, answer relevancy,
context precision, context recall)
        │
        ▼
LangGraph Evaluator Agent
(GPT-4o judge — analyzes scores + live API costs,
generates structured recommendation)
        │
        ▼
Streamlit Dashboard
(side-by-side answers, radar chart,
live cost tracking, winner recommendation)
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Pipeline Orchestration | LangChain LCEL | Declarative chain composition with pipe operator |
| Agent Framework | LangGraph | Stateful multi-node evaluator agent |
| Evaluation | RAGAS | Industry-standard RAG scoring framework |
| Observability | LangSmith | Production tracing, token usage, latency |
| API Backend | FastAPI + LangServe | REST endpoints + automatic streaming/playground |
| Vector Store (dev) | ChromaDB | Local in-memory vector storage |
| Vector Store (prod) | Qdrant Cloud | Production-grade vector database |
| Embeddings | OpenAI ada-002 | Pipelines 1 and 4 |
| Embeddings | Cohere embed-v3 | Pipeline 2 |
| Embeddings | BGE-large-en-v1.5 | Pipeline 3 — free, open-source, MTEB competitive |
| Reranking | Cohere Rerank v3 | Pipeline 2 — neural reranking |
| Reranking | ms-marco Cross-Encoder | Pipeline 3 — free local reranking |
| Frontend | Streamlit + Plotly | Dashboard with radar charts and cost analysis |
| Containerization | Docker + Compose | Multi-service containerization |
| Deployment | HuggingFace Spaces | Cloud deployment |
| LLM | OpenAI GPT-4o-mini | Generation across all pipelines |
| LLM (Agent) | OpenAI GPT-4o | LangGraph evaluator agent |

---

## ⚙️ Pipeline Configurations

| Pipeline | Chunking | Embeddings | Reranking | Est. Cost/1k queries |
|---|---|---|---|---|
| Pipeline 1 — Baseline | Fixed 512 tokens | OpenAI ada-002 | None | ~$0.45 |
| Pipeline 2 — Neural Rerank | Recursive 1024 tokens | Cohere embed-v3 | Cohere Rerank v3 | ~$2.55 |
| Pipeline 3 — Open Source | Semantic chunking | BGE-large (free) | Cross-Encoder (free) | ~$0.15 |
| Pipeline 4 — Diversity | Fixed 1024 tokens | OpenAI ada-002 | MMR retrieval | ~$0.45 |

**Pipeline 1** is the control group — the simplest possible RAG implementation that every other pipeline is measured against.

**Pipeline 2** introduces neural reranking — retrieves 10 chunks then re-scores them with Cohere's cross-encoder, keeping only the top 4. This dramatically improves context precision at higher cost.

**Pipeline 3** is the most cost-efficient. BGE-large-en-v1.5 outperforms OpenAI ada-002 on MTEB benchmarks at zero cost. The ms-marco cross-encoder reranker also runs locally. Total embedding + reranking cost: $0.

**Pipeline 4** uses MMR (Maximal Marginal Relevance) retrieval — fetches 20 candidates then selects 4 that maximize both relevance AND diversity. Prevents the common failure mode of returning 4 chunks that all say the same thing.

---

## 📊 RAGAS Evaluation Metrics

| Metric | What It Measures |
|---|---|
| Faithfulness | Does the answer contain ONLY information from retrieved context? Catches hallucination. |
| Answer Relevancy | Does the answer actually address the question asked? |
| Context Precision | Are the retrieved chunks actually relevant to the question? |
| Context Recall | Did the retriever find ALL relevant information? Requires ground truth. |

---

## 🔬 Key Findings



---

## 💡 What I Learned



---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Docker Desktop
- API keys: OpenAI, Cohere, Qdrant (free tier), LangSmith (free tier)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/devurvpatel/rag-pipeline-optimizer.git
cd rag-pipeline-optimizer

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Fill in your API keys in .env
```

### Environment Variables

```
OPENAI_API_KEY=
COHERE_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=
QDRANT_URL=
QDRANT_API_KEY=
```

### Run Locally

**Terminal 1 — Start backend:**
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Start frontend:**
```bash
python -m streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

### Run with Docker

```bash
docker-compose up --build
```

- Frontend: `http://localhost:8501`
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

---

## 📸 Screenshots

### 📝 4 Pipeline Answers Side by Side
![4 Pipeline Answers](assets/4%20pipeline%20answers.png)

### 📊 Performance vs Cost Analysis
![Performance vs Cost Charts](assets/performance%20vs%20cost%20charts.png)

### 📈 RAGAS Evaluation Scores
![RAGAS Evaluation](assets/RAGAS%20evaluation.png)

### 🤖 LangGraph Agent Recommendation
![LangGraph Agent Recommendation](assets/Langgraph%20Agent%20Recommendation.png)

### 📋 Full Analysis Report
![Full Analysis Report](assets/Full%20Analysis%20Report.png)

### 🔍 LangSmith Traces

#### Comparing 4 Pipelines
![Langsmith comparing 4 pipelines](assets/Langsmith%20comparing%204%20pipelines.png)

#### Deeper Analysis of Trace
![LangSmith Trace 1](assets/Langsmith%20Trace1.png)
![LangSmith Trace 2](assets/Langsmith%20Trace2.png)

---

## 📁 Project Structure

```
rag-pipeline-optimizer/
├── backend/
│   ├── main.py                    # FastAPI + LangServe app
│   ├── pipelines/
│   │   ├── pipeline_1.py          # Fixed 512 + OpenAI + similarity
│   │   ├── pipeline_2.py          # Recursive 1024 + Cohere + rerank
│   │   ├── pipeline_3.py          # Semantic + BGE + cross-encoder
│   │   └── pipeline_4.py          # Fixed 1024 + OpenAI + MMR
│   ├── evaluation/
│   │   ├── ragas_eval.py          # RAGAS evaluation framework
│   │   └── langgraph_agent.py     # LangGraph evaluator agent
│   └── utils/
│       ├── document_processor.py  # PDF loading + 3 chunking strategies
│       └── vector_store_manager.py # ChromaDB + Qdrant management
├── frontend/
│   └── app.py                     # Streamlit dashboard
├── data/
│   └── eval_dataset.py            # 25 ground truth Q&A pairs
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
└── requirements.txt
```

---

## 🔗 Live Demo

[HuggingFace Spaces — Live Demo](https://huggingface.co/spaces/urvpatel/rag-pipeline-optimizer)

---

## 👤 Author

**Urv Patel** — Data Analyst transitioning to AI Engineering

- GitHub: [@devurvpatel](https://github.com/devurvpatel)
- Built as part of an AI Engineering portfolio

---

## 📄 License

MIT License — feel free to use this project as a reference for your own RAG evaluation systems.