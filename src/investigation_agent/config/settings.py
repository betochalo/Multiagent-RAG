"""Application settings, injected manually into clients and services."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Relative paths resolve against the project root, not the working directory: the course
# evaluator runs the solver from inside solver-v2/.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Environment(StrEnum):
    """Valid runtime environments. Used as ENVIRONMENT env var value."""

    LOCAL = "local"
    DEV = "dev"
    PROD = "prod"


class Settings(BaseSettings):
    """Settings shared by every agent and service. Read from env vars, then `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "Investigation Agent"
    environment: Environment = Environment.LOCAL

    # H200 LLM (vLLM). The model id is never configured: it is read from /v1/models,
    # because the lab changed the served model at least once this semester.
    h200_host: str = "172.28.230.10"
    h200_port: int = 12555
    h200_replica_port: int = 12559
    h200_timeout: float = 600.0
    h200_max_tokens: int = 16384
    # Reasoning is billed from the same output budget. When the model exhausts
    # max_tokens while reasoning, the client doubles it up to this cap before giving up.
    h200_max_tokens_cap: int = 40000
    h200_temperature: float = 0.0

    # H200 embeddings (Ollama). /v1/embeddings on the vLLM port returns 404.
    h200_embed_port: int = 11434
    h200_embed_model: str = "bge-m3:latest"
    h200_embed_dim: int = 1024
    h200_embed_batch: int = 64

    # Qdrant (local Docker)
    qdrant_url: str = "http://localhost:6333"
    qdrant_timeout: float = 30.0
    qdrant_collection: str = "investigation_agent"
    # Hybrid collection: one named dense vector (bge-m3) and one sparse vector (BM25).
    qdrant_dense_vector: str = "dense"
    qdrant_sparse_vector: str = "bm25"
    # BM25 is computed server-side by Qdrant (>= 1.15); no fastembed needed. The language
    # drives stemming and stopwords, and must be the same at ingestion and query time.
    qdrant_bm25_model: str = "Qdrant/bm25"
    qdrant_bm25_language: str = "spanish"

    # PDF reader
    # A line is a heading when its font is at least this much larger than the body font.
    pdf_heading_ratio: float = 1.15
    # Below this many extracted characters the PDF is rejected (scanned or empty).
    pdf_min_chars: int = 100

    # RAG
    rag_chunk_size: int = 1200  # characters
    rag_chunk_overlap: int = 200  # characters carried from the previous chunk
    rag_top_k: int = 5
    # Candidates each retriever (dense, sparse) contributes before RRF fusion.
    rag_prefetch_k: int = 20

    # Knowledge base: course material indexed next to every task statement.
    course_notes_dir: str = str(PROJECT_ROOT / "taller-03-v2-solver-multiagente" / "notas-teoricas")
    # Entities and communities of the course notes are extracted once and cached here.
    cache_dir: str = str(PROJECT_ROOT / ".cache")
    # Concurrent LLM calls while indexing: vLLM batches concurrent requests.
    index_workers: int = 8
    # Communities smaller than this get no LLM summary.
    community_min_size: int = 3

    # Solver loop
    max_plan_attempts: int = 3
    max_code_attempts: int = 3  # per subtask
    max_write_attempts: int = 3
    # Token budget per run, checked before every LLM call. The reserve is kept for the
    # writer: when the work reaches budget - reserve, the solver writes with what it has.
    token_budget: int = 400_000
    writer_reserve_tokens: int = 60_000
    results_file: str = "resultados.json"  # the fixed-name results contract of every script

    # Sandbox
    sandbox_timeout_s: int = 180
    # A script that wants the network (downloads) never runs without a human approving it.
    # With confirmation off, such scripts are rejected; with it on, the solver asks on stdin.
    network_confirmation: bool = False

    # Extension A (Parte 4): subtasks whose dependencies are done run at once (LangGraph Send),
    # round-robin over the two H200 replicas. Off, the graph is exactly the measured baseline.
    parallel_subtasks: bool = False

    # Ablations (Parte 2.b)
    ablation_no_graph: bool = False  # researcher gets only the top-k similar chunks
    ablation_no_critic: bool = False  # one attempt per script and no checks


@lru_cache
def get_settings() -> Settings:
    """Get the cached settings, to be passed explicitly to clients and services."""
    return Settings()
