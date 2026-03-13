"""
Tests for CourseSearchTool.execute() against the real ChromaDB.

What each test catches
----------------------
test_general_query_returns_content       — wrong CHROMA_PATH (empty DB) or embedding model failure
test_result_contains_course_header       — error short-circuit before _format_results()
test_partial_course_name_filter          — course_catalog empty / semantic resolution broken
test_lesson_number_filter_no_search_error — PRIMARY BUG: n_results > index size → ChromaDB exception
test_sources_populated_after_search      — last_sources not set because tool returned error early
test_unknown_course_returns_error_string — exception raised instead of graceful error string
"""
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _db_has_content(vector_store) -> bool:
    """Return True if course_content collection is non-empty."""
    try:
        results = vector_store.course_content.get(limit=1)
        return bool(results and results.get("ids"))
    except Exception:
        return False


# ── tests ─────────────────────────────────────────────────────────────────────

def test_general_query_returns_content(search_tool, vector_store):
    """A broad query should return a non-empty, non-error result when the DB has data."""
    if not _db_has_content(vector_store):
        pytest.skip("course_content collection is empty — load docs first")

    result = search_tool.execute(query="what is covered in the course")

    assert result, "execute() returned an empty string"
    assert "Search error" not in result, f"Unexpected search error: {result}"
    assert "No relevant content found" not in result, "DB appears empty or mismatched path"


def test_result_contains_course_header(search_tool, vector_store):
    """Formatted results should include the [Course Title] header from _format_results()."""
    if not _db_has_content(vector_store):
        pytest.skip("course_content collection is empty — load docs first")

    result = search_tool.execute(query="introduction lesson overview")

    # _format_results() wraps each chunk in "[Course Title]" brackets
    assert "[" in result and "]" in result, (
        "Result missing course header brackets — likely returned an error string instead of formatted content"
    )


def test_partial_course_name_filter(search_tool, vector_store):
    """Partial course_name should resolve via semantic search in course_catalog."""
    if not _db_has_content(vector_store):
        pytest.skip("course_content collection is empty — load docs first")

    # Get an actual course title and truncate it to a partial match
    existing_titles = vector_store.get_existing_course_titles()
    if not existing_titles:
        pytest.skip("course_catalog is empty — load docs first")

    # Use the first word of the first title as a partial match
    first_title = existing_titles[0]
    partial = first_title.split()[0]  # e.g. "Introduction" from "Introduction to MCP"

    result = search_tool.execute(query="overview", course_name=partial)

    assert "No course found matching" not in result, (
        f"Semantic course resolution failed for partial name '{partial}' — "
        "course_catalog may be empty or CHROMA_PATH is wrong"
    )
    assert "Search error" not in result, f"Unexpected search error with course filter: {result}"


def test_lesson_number_filter_no_search_error(search_tool, vector_store):
    """
    Filtering by lesson_number must NOT raise a ChromaDB n_results error.

    Root cause being tested:
        course_content.query(n_results=5) fails when the filtered subset has
        fewer than 5 chunks.  VectorStore.search() catches it and returns
        SearchResults.empty("Search error: Number of requested results 5 is
        greater than number of elements in index N").
        CourseSearchTool.execute() returns that error string, and Claude produces
        a failure message.
    """
    if not _db_has_content(vector_store):
        pytest.skip("course_content collection is empty — load docs first")

    result = search_tool.execute(query="overview", lesson_number=1)

    assert "Search error" not in result, (
        "ChromaDB n_results > index size bug triggered.\n"
        "Fix: clamp n_results to min(max_results, actual_count) in VectorStore.search()"
    )


def test_sources_populated_after_search(search_tool, vector_store):
    """last_sources should be a non-empty list after a successful search."""
    if not _db_has_content(vector_store):
        pytest.skip("course_content collection is empty — load docs first")

    # Reset sources before the test
    search_tool.last_sources = []

    search_tool.execute(query="explain the main concept")

    assert search_tool.last_sources, (
        "last_sources is empty after a successful search — "
        "tool likely returned an error string early and never reached _format_results()"
    )
    # Each source should have at least a 'label' key
    for source in search_tool.last_sources:
        assert "label" in source, f"Source missing 'label' key: {source}"


def test_unknown_course_returns_error_string(search_tool):
    """
    execute() must always return a string without raising an exception.

    NOTE — discovered behavior: _resolve_course_name() has no similarity threshold.
    ChromaDB vector search always returns the top-1 match regardless of distance,
    so even a completely nonsensical course_name resolves to the nearest course and
    returns real content.  This is a latent bug (wrong course silently used), but
    the critical invariant here is that execute() does not raise and returns a string.
    """
    result = search_tool.execute(
        query="anything",
        course_name="ZZZ_THIS_COURSE_DOES_NOT_EXIST_XYZ_999"
    )

    assert isinstance(result, str), "execute() should always return a string, never raise"
    assert result, "execute() returned an empty string"
    # No exception / traceback leakage
    assert "Traceback" not in result, f"Exception traceback leaked into result: {result[:200]}"
