import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline Optimizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F4E79;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .winner-box {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    .winner-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .answer-box {
        background: #FAFAFA;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 1rem;
        height: 280px;
        overflow-y: auto;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .pipeline-label {
        font-weight: 600;
        color: #1F4E79;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1F4E79;
        color: white;
        border: none;
        padding: 0.6rem;
        border-radius: 6px;
        font-size: 1rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #2E75B6;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 3rem !important;
        max-width: 100% !important;
    }
    [data-testid="column"]:first-child {
        border-right: 2px solid #E0E0E0;
        padding-right: 2.5rem;
    }
    [data-testid="column"]:last-child {
        padding-left: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Helper Functions ───────────────────────────────────────────────────────────
def upload_file(file):
    file_bytes = file.read()
    files = {"file": (file.name, file_bytes, "application/pdf")}
    response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=300)
    response.raise_for_status()
    return response.json()

def evaluate(question, ground_truth):
    payload = {"question": question, "ground_truth": ground_truth}
    response = requests.post(f"{BACKEND_URL}/evaluate", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()

def make_individual_radar_chart(pipeline_name, scores, color):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_labels = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]
    values = [scores.get(m, 0) for m in metrics]
    values_closed = values + [values[0]]
    labels_closed = metric_labels + [metric_labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor=color,
        opacity=0.3,
        line=dict(color=color, width=2.5),
        name=pipeline_name,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                tickfont=dict(size=8),
                gridcolor="#E0E0E0",
            ),
            angularaxis=dict(
                tickfont=dict(size=9, color="#1F4E79"),
                gridcolor="#E0E0E0",
            ),
            bgcolor="white",
        ),
        showlegend=False,
        title=dict(
            text=pipeline_name.split("—")[-1].strip() if "—" in pipeline_name else pipeline_name,
            font=dict(size=11, color="#1F4E79"),
        ),
        height=280,
        margin=dict(l=40, r=40, t=50, b=20),
    )
    return fig

def make_live_cost_chart(live_costs):
    colors = ["#2E75B6", "#C55A11", "#1E6B3C", "#6B2C91"]
    names = []
    costs = []
    chart_colors = []

    for i, (name, cost) in enumerate(live_costs.items()):
        short = f"Pipeline {i+1}"
        for num in ["1", "2", "3", "4"]:
            if f"Pipeline {num}" in name:
                short = f"Pipeline {num}"
                chart_colors.append(colors[int(num) - 1])
                break
        names.append(short)
        costs.append(round(cost, 6))

    fig = go.Figure(go.Bar(
        x=names,
        y=costs,
        marker_color=chart_colors,
        text=[f"${c:.6f}" for c in costs],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(
            text="Actual Cost This Query (USD) — Live from OpenAI API",
            font=dict(size=16, color="#1F4E79"),
        ),
        yaxis=dict(
            title="Cost (USD)",
            range=[0, max(costs) * 1.5] if max(costs) > 0 else [0, 0.01],
        ),
        xaxis=dict(title="Pipeline"),
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="white",
        yaxis_gridcolor="#F0F0F0",
    )
    return fig

# ── Main Layout — Fixed 2-column, config left, content right ──────────────────
col_config, col_main = st.columns([1.5, 3.5])

# ── LEFT COLUMN — Configuration (always visible, never collapses) ─────────────
with col_config:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📄 Upload your document",
        type=["pdf"],
        help="Upload a PDF to benchmark all 4 RAG pipelines against it",
    )

    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")
        if st.button("Process Document"):
            with st.spinner("Processing document through 3 chunking strategies across 4 pipelines..."):
                try:
                    uploaded_file.seek(0)
                    result = upload_file(uploaded_file)
                    st.session_state["upload_result"] = result
                    st.session_state["document_ready"] = True
                    st.success("Document processed successfully")
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")

    st.markdown("---")
    st.markdown("## 💬 Ask a Question")

    question = st.text_area(
        "Your question",
        placeholder="What is n8n used for?",
        height=100,
    )

    ground_truth = st.text_area(
        "Ground truth answer (optional)",
        placeholder="Provide the correct answer to enable RAGAS scoring...",
        height=100,
        help="If provided, RAGAS will score each pipeline's answer against this",
    )

    run_button = st.button(
        "🚀 Run All Pipelines",
        disabled=not question,
        type="primary",
    )

# ── RIGHT COLUMN — All main content ───────────────────────────────────────────
with col_main:
    st.markdown(
        '<div class="main-header">🔍 RAG Pipeline Optimizer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Benchmark 4 RAG configurations simultaneously '
        'and find the optimal setup for your documents</div>',
        unsafe_allow_html=True,
    )

    # ── Before Run — Instructions + Pipeline Overview ─────────────────────────
    if "eval_result" not in st.session_state:

        if "upload_result" in st.session_state:
            r = st.session_state["upload_result"]
            st.success(f"✅ Document ready — {r['pages']} pages processed")
            stats = r["chunking_stats"]
            st.markdown("#### Chunking Strategy Comparison")
            chunk_col1, chunk_col2, chunk_col3 = st.columns(3)

            with chunk_col1:
                st.markdown(f"""
                <div style="background:#E8F4FD; padding:1rem; border-radius:8px; text-align:center;">
                    <div style="font-size:0.85rem; color:#666;">Fixed 512</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#1F4E79;">{stats['fixed_512']['count']} chunks</div>
                    <div style="font-size:0.85rem; color:#444;">avg {stats['fixed_512']['avg_length']} chars</div>
                </div>
                """, unsafe_allow_html=True)

            with chunk_col2:
                st.markdown(f"""
                <div style="background:#E8F4FD; padding:1rem; border-radius:8px; text-align:center;">
                    <div style="font-size:0.85rem; color:#666;">Recursive 1024</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#1F4E79;">{stats['recursive_1024']['count']} chunks</div>
                    <div style="font-size:0.85rem; color:#444;">avg {stats['recursive_1024']['avg_length']} chars</div>
                </div>
                """, unsafe_allow_html=True)

            with chunk_col3:
                st.markdown(f"""
                <div style="background:#E8F4FD; padding:1rem; border-radius:8px; text-align:center;">
                    <div style="font-size:0.85rem; color:#666;">Semantic</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#1F4E79;">{stats['semantic']['count']} chunks</div>
                    <div style="font-size:0.85rem; color:#444;">avg {stats['semantic']['avg_length']} chars</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

        st.markdown("### How It Works")
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("""
            **1. Upload your document** using the configuration panel on the left

            **2. Ask a question** that can be answered from your document

            **3. Optionally provide a ground truth** answer to enable RAGAS scoring

            **4. Click Run All Pipelines** — all 4 run simultaneously

            **5. Review the comparison** — scores, costs, and AI recommendation
            """)

        with col_b:
            st.markdown("### 🏗️ Pipeline Configurations")
            pipeline_table = pd.DataFrame([
                {"Pipeline": "Pipeline 1", "Chunking": "Fixed 512", "Embeddings": "OpenAI", "Reranking": "None"},
                {"Pipeline": "Pipeline 2", "Chunking": "Recursive 1024", "Embeddings": "Cohere", "Reranking": "Cohere Rerank"},
                {"Pipeline": "Pipeline 3", "Chunking": "Semantic", "Embeddings": "BGE-large", "Reranking": "Cross-Encoder"},
                {"Pipeline": "Pipeline 4", "Chunking": "Fixed 1024", "Embeddings": "OpenAI", "Reranking": "MMR"},
            ])
            st.dataframe(pipeline_table, hide_index=True, use_container_width=True)

    # ── Run Pipelines ──────────────────────────────────────────────────────────
    if run_button and question:
        with st.spinner("⚡ Running all 4 pipelines in parallel... (this takes 1-2 minutes)"):
            try:
                result = evaluate(question, ground_truth)
                st.session_state["eval_result"] = result
                st.session_state["last_question"] = question
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")
                st.info("Make sure the backend is running: python -m uvicorn backend.main:app --reload --port 8000")

    # ── Results ────────────────────────────────────────────────────────────────
    if "eval_result" in st.session_state:
        result = st.session_state["eval_result"]
        question_asked = st.session_state.get("last_question", "")

        st.markdown("---")
        st.markdown(f"### 💬 Question: *{question_asked}*")
        st.markdown("---")

        # ── 4 Pipeline Answers ────────────────────────────────────────────────
        st.markdown("### 📝 Pipeline Answers")
        pipeline_results = result.get("pipeline_results", {})

        cols = st.columns(4)
        colors = ["#2E75B6", "#C55A11", "#1E6B3C", "#6B2C91"]

        for i, (col, (name, data)) in enumerate(zip(cols, pipeline_results.items())):
            with col:
                st.markdown(
                    f'<div class="pipeline-label" style="color:{colors[i]}">Pipeline {i+1}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(name.split("—")[-1].strip() if "—" in name else name)
                st.markdown(
                    f'<div class="answer-box">{data["answer"]}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"📦 {data['chunks_retrieved']} chunks retrieved")
                st.caption(f"💰 ${data.get('cost_usd', 0):.6f} this query")
                st.caption(f"🔤 {data.get('tokens_used', 0)} tokens used")

        st.markdown("---")

        # ── RAGAS Scores + Cost Charts ─────────────────────────────────────────
        ragas_scores = result.get("ragas_scores", {})

        if ragas_scores:
            st.markdown("### 📊 Performance vs Cost Analysis")
            st.markdown("#### RAGAS Metrics — Individual Pipeline Performance")

            colors = ["#2E75B6", "#C55A11", "#1E6B3C", "#6B2C91"]
            radar_cols = st.columns(2)

            for i, (pipeline_name, scores) in enumerate(ragas_scores.items()):
                with radar_cols[i % 2]:
                    fig = make_individual_radar_chart(
                        pipeline_name,
                        scores,
                        colors[i % len(colors)]
                    )
                    st.plotly_chart(fig, use_container_width=True)

            live_costs = {
                name: data.get("cost_usd", 0)
                for name, data in pipeline_results.items()
            }
            cost_fig = make_live_cost_chart(live_costs)
            st.plotly_chart(cost_fig, use_container_width=True)

            # RAGAS scores table
            st.markdown("#### RAGAS Scores Detail")
            scores_data = []
            for name, scores in ragas_scores.items():
                scores_data.append({
                    "Pipeline": name.split("—")[0].strip(),
                    "Faithfulness": scores.get("faithfulness", 0),
                    "Answer Relevancy": scores.get("answer_relevancy", 0),
                    "Context Precision": scores.get("context_precision", 0),
                    "Context Recall": scores.get("context_recall", 0),
                    "Average": round(sum([
                        scores.get("faithfulness", 0),
                        scores.get("answer_relevancy", 0),
                        scores.get("context_precision", 0),
                        scores.get("context_recall", 0),
                    ]) / 4, 3),
                })

            scores_df = pd.DataFrame(scores_data)
            st.dataframe(
                scores_df.style.highlight_max(
                    subset=["Faithfulness", "Answer Relevancy",
                            "Context Precision", "Context Recall", "Average"],
                    color="#D6E4F0",
                ),
                hide_index=True,
                use_container_width=True,
            )

        else:
            st.info("💡 Provide a ground truth answer to see RAGAS scores and the AI recommendation")

        # ── LangGraph Agent Recommendation ────────────────────────────────────
        agent_report = result.get("agent_report", {})

        if agent_report and agent_report.get("winner"):
            st.markdown("---")
            st.markdown("### 🤖 AI Recommendation (LangGraph Agent)")

            st.markdown(f"""
<div class="winner-box">
    <div class="winner-title">🏆 Winner: {agent_report.get('winner', 'N/A')}</div>
    <p style="margin: 0.5rem 0">{agent_report.get('reason', '')}</p>
</div>
""", unsafe_allow_html=True)

            detail_col1, detail_col2, detail_col3 = st.columns(3)

            with detail_col1:
                st.markdown("**⚠️ Trade-off**")
                st.info(agent_report.get("tradeoff", "N/A"))

            with detail_col2:
                st.markdown("**🎯 Best for Quality**")
                st.info(agent_report.get("best_for_quality", "N/A"))

            with detail_col3:
                st.markdown("**💰 Best for Cost**")
                st.info(agent_report.get("best_for_cost", "N/A"))

            with st.expander("📋 View Full Analysis"):
                st.markdown("**Production Advice:**")
                st.write(agent_report.get("production_advice", "N/A"))
                st.markdown("**Full Technical Analysis:**")
                st.write(agent_report.get("full_analysis", "N/A"))

        # ── Reset Button ───────────────────────────────────────────────────────
        st.markdown("---")
        if st.button("🔄 Run Another Question"):
            del st.session_state["eval_result"]
            st.rerun()