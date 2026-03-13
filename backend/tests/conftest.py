"""
Shared fixtures for the RAG chatbot test suite.

Contains two fixture sets:
- Integration fixtures: real VectorStore, CourseSearchTool, AIGenerator, RAGSystem
  backed by the actual ChromaDB and API key from .env.
- App fixtures: mock-based setup for testing FastAPI routes without heavy deps.

Import-time strategy for app fixtures
--------------------------------------
app.py has two module-level side effects that break in the test environment:

  1. rag_system = RAGSystem(config)  — tries to connect to ChromaDB and load
     sentence-transformer embeddings.
  2. app.mount("/", StaticFiles(directory="../frontend", html=True), ...)
     — fails because the frontend directory does not exist relative to the
     working directory used by pytest.

Both are neutralised by patching their sources *before* app.py is imported
for the first time.  The patches live in a module-level `with` block so they
are active exactly while the import executes and are automatically removed
afterwards (the already-imported module retains the mock references).
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# Minimal ASGI stand-in for StaticFiles
# ---------------------------------------------------------------------------

class _MockStaticFiles:
    """Lightweight ASGI app that replaces StaticFiles during tests.

    Using a real class (not a MagicMock instance) lets app.py's
    `class DevStaticFiles(StaticFiles)` inherit from it without TypeError.
    """

    def __init__(self, *args, **kwargs):
        pass

    async def __call__(self, scope, receive, send):
        pass


# ---------------------------------------------------------------------------
# Shared mock RAGSystem — configured before app.py is imported
# ---------------------------------------------------------------------------

_mock_rag = MagicMock()
_mock_rag.session_manager.create_session.return_value = "session_test_1"
_mock_rag.query.return_value = (
    "Test answer",
    [{"course": "Test Course", "lesson": 1}],
)
_mock_rag.get_course_analytics.return_value = {
    "total_courses": 2,
    "course_titles": ["Python Basics", "FastAPI Advanced"],
}

# Patch both heavy dependencies while app.py is imported for the first time.
with (
    patch("rag_system.RAGSystem", return_value=_mock_rag),
    patch("fastapi.staticfiles.StaticFiles", _MockStaticFiles),
):
    import app as _app_module  # noqa: E402

# Belt-and-suspenders: make sure the module-level variable points to our mock.
_app_module.rag_system = _mock_rag


# ── Integration fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def vector_store():
    """Real VectorStore pointed at the project's actual ChromaDB."""
    cfg = Config()
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
    cfg.CHROMA_PATH = CHROMA_PATH
    return RAGSystem(cfg)


# ── App / mock fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def mock_rag():
    """Reset call history and return the shared mock RAGSystem."""
    _mock_rag.reset_mock()
    _mock_rag.session_manager.create_session.return_value = "session_test_1"
    _mock_rag.query.return_value = (
        "Test answer",
        [{"course": "Test Course", "lesson": 1}],
    )
    _mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python Basics", "FastAPI Advanced"],
    }
    return _mock_rag


@pytest.fixture
def client(mock_rag):
    """Starlette TestClient wired to the patched FastAPI app."""
    from fastapi.testclient import TestClient

    with TestClient(_app_module.app) as c:
        yield c
