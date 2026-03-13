"""
Shared fixtures for RAG chatbot tests.

Key design decisions:
- Adds backend/ to sys.path so bare imports (vector_store, search_tools, etc.) work
- Loads .env from project root for ANTHROPIC_API_KEY
- Overrides CHROMA_PATH with an absolute path — the relative ./chroma_db in config.py
  breaks when pytest runs from a directory other than backend/
- Session scope on fixtures to avoid reloading the sentence-transformer model (~2s) per test
"""
import sys
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
TESTS_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = TESTS_DIR.parent.resolve()
PROJECT_ROOT = BACKEND_DIR.parent.resolve()

# Make bare imports like `from vector_store import VectorStore` work
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load .env before importing anything that reads os.getenv at import time
load_dotenv(PROJECT_ROOT / ".env")

# Absolute path to ChromaDB — avoids CWD-relative ./chroma_db bug
CHROMA_PATH = str(BACKEND_DIR / "chroma_db")

# ── Imports (after sys.path patch) ────────────────────────────────────────────
from vector_store import VectorStore
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from config import Config


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def vector_store():
    """Real VectorStore pointed at the project's actual ChromaDB."""
    from config import Config as _Config
    cfg = _Config()
    return VectorStore(CHROMA_PATH, cfg.EMBEDDING_MODEL, cfg.MAX_RESULTS)


@pytest.fixture(scope="session")
def search_tool(vector_store):
    """CourseSearchTool backed by the real VectorStore."""
    return CourseSearchTool(vector_store)


@pytest.fixture(scope="session")
def ai_generator():
    """AIGenerator using the real API key from .env."""
    cfg = Config()
    return AIGenerator(cfg.ANTHROPIC_API_KEY, cfg.ANTHROPIC_MODEL)


@pytest.fixture(scope="session")
def rag_system():
    """Full RAGSystem with real ChromaDB and real API key."""
    cfg = Config()
    # Override the relative CHROMA_PATH with an absolute one
    cfg.CHROMA_PATH = CHROMA_PATH
    return RAGSystem(cfg)
