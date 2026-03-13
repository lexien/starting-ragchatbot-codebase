"""
Shared fixtures for the RAG chatbot test suite.

Import-time strategy
--------------------
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
from unittest.mock import MagicMock, patch

import pytest

# Make backend modules importable (mirrors [tool.pytest.ini_options] pythonpath,
# but also works when tests are run directly via `python -m pytest`).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


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
# * patch("rag_system.RAGSystem") — intercepted by `from rag_system import RAGSystem`
#   inside app.py; calling RAGSystem(config) returns _mock_rag.
# * patch("fastapi.staticfiles.StaticFiles", _MockStaticFiles) — intercepted by
#   `from fastapi.staticfiles import StaticFiles` inside app.py.
with (
    patch("rag_system.RAGSystem", return_value=_mock_rag),
    patch("fastapi.staticfiles.StaticFiles", _MockStaticFiles),
):
    import app as _app_module  # noqa: E402

# Belt-and-suspenders: make sure the module-level variable points to our mock
# so that the route handlers reference it correctly.
_app_module.rag_system = _mock_rag


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag():
    """Reset call history and return the shared mock RAGSystem.

    Return values survive reset_mock() (Python keeps them by default), but
    we reconfigure them explicitly so individual tests can rely on known
    defaults even after other tests mutate side_effect or return_value.
    """
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
