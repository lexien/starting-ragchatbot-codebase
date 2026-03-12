# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the app:**
```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Install dependencies:**
```bash
uv sync
```

**Add a dependency:**
```bash
uv add <package>
```

**Never use `pip` directly** — always use `uv` to manage dependencies so `pyproject.toml` and `uv.lock` stay in sync.

The app is available at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

**Required:** A `.env` file in the project root with `ANTHROPIC_API_KEY=...`. The server will start without it but all queries will fail.

## Architecture

This is a full-stack RAG (Retrieval-Augmented Generation) app. The backend is in `backend/`, the frontend is static files in `frontend/`, and course source documents live in `docs/`.

### Query flow

1. Browser POSTs `{ query, session_id }` to `/api/query`
2. `app.py` delegates to `RAGSystem.query()`
3. `RAGSystem` fetches session history, then calls `AIGenerator.generate_response()` with the query, history, and tool definitions
4. Claude decides whether to call the `search_course_content` tool:
   - **No tool call** → direct answer returned (one API call total)
   - **Tool call** → `ToolManager` executes `CourseSearchTool`, which hits ChromaDB for semantic search, returns chunks; Claude makes a second API call to synthesize the answer
5. Sources (course + lesson) are collected from `CourseSearchTool.last_sources`, session history is updated, response returned to browser

### Key architectural decisions

- **Tool-based retrieval**: Claude autonomously decides when to search. The tool (`search_course_content`) supports optional `course_name` and `lesson_number` filters — partial course name matches are supported.
- **Two ChromaDB collections**: `course_catalog` stores course metadata; `course_content` stores chunked lesson text with embeddings (`all-MiniLM-L6-v2`).
- **Session history**: Stored in-memory in `SessionManager`. Only the last `MAX_HISTORY=2` exchanges are passed to Claude to keep context short.
- **Startup document loading**: On server start, `app.py` loads all `.txt/.pdf/.docx` files from `../docs/`. Already-loaded courses are skipped (deduped by title).
- **No framework on the frontend**: Plain HTML/CSS/JS. Uses `fetch` for API calls and `marked.js` (CDN) for markdown rendering.

### Adding a new tool for Claude

1. Create a class extending `Tool` (ABC in `search_tools.py`) implementing `get_tool_definition()` and `execute()`
2. Register it: `tool_manager.register_tool(MyTool())` in `RAGSystem.__init__()`

### Adding new course documents

Drop `.txt`, `.pdf`, or `.docx` files into `docs/`. The document parser (`document_processor.py`) expects this structure:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 1: <title>
<lesson content>

Lesson 2: <title>
...
```

Restart the server to ingest new files (or call `rag_system.add_course_folder()` directly).

### Configuration (`backend/config.py`)

All tunable settings are in `Config`:
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — text chunking for vector storage
- `MAX_RESULTS` — number of ChromaDB results returned per search
- `MAX_HISTORY` — conversation turns kept in session
- `CHROMA_PATH` — where ChromaDB persists to disk (`./chroma_db` relative to `backend/`)
- `ANTHROPIC_MODEL` — Claude model used for generation
