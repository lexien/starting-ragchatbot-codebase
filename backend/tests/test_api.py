"""
API endpoint tests for the RAG chatbot backend.

Covers:
  POST /api/query         — query processing and session handling
  GET  /api/courses       — course catalogue stats
  DELETE /api/sessions/{id} — session history clearing
"""

import pytest


class TestQueryEndpoint:
    def test_returns_200_with_answer(self, client, mock_rag):
        response = client.post("/api/query", json={"query": "What is Python?"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer"
        assert isinstance(data["sources"], list)

    def test_creates_session_when_none_provided(self, client, mock_rag):
        response = client.post("/api/query", json={"query": "What is Python?"})
        assert response.status_code == 200
        assert response.json()["session_id"] == "session_test_1"
        mock_rag.session_manager.create_session.assert_called_once()

    def test_uses_provided_session_id(self, client, mock_rag):
        response = client.post(
            "/api/query",
            json={"query": "What is Python?", "session_id": "my-session"},
        )
        assert response.status_code == 200
        assert response.json()["session_id"] == "my-session"
        mock_rag.session_manager.create_session.assert_not_called()

    def test_passes_query_and_session_to_rag(self, client, mock_rag):
        client.post("/api/query", json={"query": "Tell me about FastAPI"})
        mock_rag.query.assert_called_once_with("Tell me about FastAPI", "session_test_1")

    def test_returns_sources_from_rag(self, client, mock_rag):
        sources = [
            {"course": "Python 101", "lesson": 2},
            {"course": "FastAPI", "lesson": 1},
        ]
        mock_rag.query.return_value = ("Answer with sources", sources)
        response = client.post("/api/query", json={"query": "Give me sources"})
        assert response.status_code == 200
        assert response.json()["sources"] == sources

    def test_returns_500_when_rag_raises(self, client, mock_rag):
        mock_rag.query.side_effect = RuntimeError("DB connection failed")
        response = client.post("/api/query", json={"query": "anything"})
        assert response.status_code == 500
        assert "DB connection failed" in response.json()["detail"]

    def test_returns_422_when_query_field_missing(self, client, mock_rag):
        response = client.post("/api/query", json={"session_id": "s1"})
        assert response.status_code == 422

    def test_returns_422_for_empty_body(self, client, mock_rag):
        response = client.post("/api/query", json={})
        assert response.status_code == 422


class TestCoursesEndpoint:
    def test_returns_200_with_stats(self, client, mock_rag):
        response = client.get("/api/courses")
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Python Basics", "FastAPI Advanced"]

    def test_calls_get_course_analytics(self, client, mock_rag):
        client.get("/api/courses")
        mock_rag.get_course_analytics.assert_called_once()

    def test_returns_empty_catalog(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        response = client.get("/api/courses")
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_returns_500_when_rag_raises(self, client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("DB error")
        response = client.get("/api/courses")
        assert response.status_code == 500
        assert "DB error" in response.json()["detail"]


class TestSessionEndpoint:
    def test_delete_returns_ok(self, client, mock_rag):
        response = client.delete("/api/sessions/session_123")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_delete_calls_clear_session_with_id(self, client, mock_rag):
        client.delete("/api/sessions/my-session-id")
        mock_rag.session_manager.clear_session.assert_called_once_with("my-session-id")

    def test_delete_different_session_ids(self, client, mock_rag):
        for session_id in ("abc", "session_42", "user-xyz-session"):
            mock_rag.reset_mock()
            response = client.delete(f"/api/sessions/{session_id}")
            assert response.status_code == 200
            mock_rag.session_manager.clear_session.assert_called_once_with(session_id)
