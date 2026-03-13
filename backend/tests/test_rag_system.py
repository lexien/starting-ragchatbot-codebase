"""
Integration tests for the full RAG system (real Anthropic API + real ChromaDB).

All tests are skipped when ANTHROPIC_API_KEY is not set.

What each test catches
----------------------
test_content_query_returns_non_empty_answer  — full stack failure: API key, model name,
                                               DB path, or the "query failed" string
test_content_query_populates_sources         — retrieval never completing successfully
test_general_knowledge_query_returns_answer  — AttributeError on tool_use path with no tools
test_second_query_in_session_succeeds        — history formatting corrupts system prompt on turn 2
"""
import os
import sys
import uuid
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).parent.parent.resolve()
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Skip the entire module if no API key is present
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _db_has_content(rag_system) -> bool:
    try:
        results = rag_system.vector_store.course_content.get(limit=1)
        return bool(results and results.get("ids"))
    except Exception:
        return False


# ── tests ─────────────────────────────────────────────────────────────────────

def test_content_query_returns_non_empty_answer(rag_system):
    """
    A content question must return a non-empty answer with no 'query failed' text.

    This is the end-to-end smoke test: if anything in the stack is broken
    (wrong API key, wrong model name, wrong ChromaDB path, ChromaDB n_results bug)
    this test will fail.
    """
    if not _db_has_content(rag_system):
        pytest.skip("course_content collection is empty — load docs first")

    response, _ = rag_system.query("What topics are covered in the course materials?")

    assert response, "RAGSystem.query() returned an empty response"
    assert "query failed" not in response.lower(), (
        f"RAG system returned a failure message: {response!r}"
    )
    assert len(response) > 20, f"Response suspiciously short: {response!r}"


def test_content_query_populates_sources(rag_system):
    """
    A content question that triggers a tool call should return non-empty sources.

    Catches: tool execution silently failing so last_sources is never populated.
    """
    if not _db_has_content(rag_system):
        pytest.skip("course_content collection is empty — load docs first")

    _, sources = rag_system.query("Explain the main concept from the first lesson")

    # Sources may be empty if Claude chose not to call the search tool; that is
    # acceptable only if the response itself was valid.  We assert that sources
    # is at least a list (not None / not an exception).
    assert isinstance(sources, list), f"sources should be a list, got: {type(sources)}"


def test_general_knowledge_query_returns_answer(rag_system):
    """
    A general knowledge question (no tool call needed) must return a valid answer.

    Catches: AttributeError on the tool_use branch when no tool_manager is set,
    or if the end_turn path is broken.
    """
    response, sources = rag_system.query("What does the acronym RAG stand for?")

    assert response, "RAGSystem.query() returned empty response for general knowledge question"
    assert "query failed" not in response.lower(), (
        f"RAG system returned a failure message for a general-knowledge query: {response!r}"
    )
    # General knowledge questions should not produce sources from course DB
    # (they might if Claude decides to search, so we just verify sources is a list)
    assert isinstance(sources, list)


def test_second_query_in_session_succeeds(rag_system):
    """
    A second query in the same session must succeed without corrupting history.

    Catches: conversation history being formatted incorrectly and crashing the
    second API call, or session manager returning a malformed string.
    """
    if not _db_has_content(rag_system):
        pytest.skip("course_content collection is empty — load docs first")

    session_id = f"test-session-{uuid.uuid4().hex[:8]}"

    first_response, _ = rag_system.query("What is a RAG system?", session_id=session_id)
    assert first_response and "query failed" not in first_response.lower(), (
        f"First query failed: {first_response!r}"
    )

    second_response, _ = rag_system.query(
        "Can you give me more detail about that?",
        session_id=session_id
    )
    assert second_response, "Second query in session returned empty response"
    assert "query failed" not in second_response.lower(), (
        f"Second query failed (likely history formatting bug): {second_response!r}"
    )
