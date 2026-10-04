"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys & LLM Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_llm_client():
    """Return an initialized OpenAI-compatible client (Groq or OpenAI)."""
    if GROQ_API_KEY:
        from openai import OpenAI
        return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    elif OPENAI_API_KEY:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    return None


def get_llm_model() -> str:
    """Return the active model name."""
    if GROQ_API_KEY:
        return DEFAULT_GROQ_MODEL
    return DEFAULT_OPENAI_MODEL


def has_llm_api() -> bool:
    """Check if any LLM API key is configured."""
    return bool(GROQ_API_KEY or OPENAI_API_KEY)

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
