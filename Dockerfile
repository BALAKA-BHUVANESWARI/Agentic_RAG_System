# IT Support Copilot: one image for the API, the Streamlit UI and the one-off tools (ingest, eval, tests).
#
# Secrets are NOT part of the image: API keys are passed at run time through docker-compose (env_file .env).
# The embedding model is baked in, so containers start without a download and can run with the Hugging Face hub offline.
# Pinecone, Groq and Tavily are still external services: the containers need outbound internet for them.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only PyTorch. The default wheel pulls roughly 2 GB of CUDA libraries this demo never uses.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Bake the embedding model into the image (must match EMBEDDING_MODEL_NAME at run time).
ARG EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "import sys; from sentence_transformers import SentenceTransformer; SentenceTransformer(sys.argv[1])" "${EMBEDDING_MODEL_NAME}" \
    && chmod -R a+rX /opt/hf-cache

# Bake the optional reranker (cross-encoder) too: containers run with HF_HUB_OFFLINE=1 and cannot download it.
# Must match RERANK_MODEL_NAME at run time. It is only used when RERANK_ENABLED=true.
ARG RERANK_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "import sys; from sentence_transformers import CrossEncoder; CrossEncoder(sys.argv[1])" "${RERANK_MODEL_NAME}" \
    && chmod -R a+rX /opt/hf-cache

# Application code, the synthetic demo knowledge base and the labelled evaluation dataset.
# (.dockerignore keeps .env, virtual environments, databases, uploads and evaluation results out.)
COPY pyproject.toml ./
COPY src ./src
COPY frontend ./frontend
COPY .streamlit ./.streamlit
COPY scripts ./scripts
COPY tests ./tests
COPY data/kb_documents ./data/kb_documents
COPY data/eval/dataset.json ./data/eval/dataset.json

# Run as a non-root user. These folders are mounted as named volumes by docker-compose.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p data/sqlite data/uploads data/eval/results \
    && chown -R app:app /app
USER app

EXPOSE 8000 8501

# Default command: the API. docker-compose overrides it for the UI and the tools.
CMD ["python", "scripts/run_api.py"]
