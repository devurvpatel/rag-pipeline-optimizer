#!/bin/bash

# Start FastAPI backend in background on port 8000
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 15

# Start Streamlit frontend on port 7860 (HuggingFace required port)
python -m streamlit run frontend/app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableXsrfProtection=false \
    --server.enableCORS=false