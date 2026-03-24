# Project: MovieMatch

## Quick Facts
- Stack: FastAPI + SentenceTransformers + ChromaDB (local) / Pinecone (prod) + Anthropic API.
- Purpose: Semantic movie recommendation engine. Users submit a natural language query; the app embeds it, retrieves vector-similar candidates, then uses Claude to rerank/filter them.

## Architecture at a Glance

| File | Role |
|------|------|
| `config.py` | All constants: models, thresholds, pricing, env vars |
| `claude.py` | Anthropic integration: `rerank()`, `rerank_stream()`, tool definitions |
| `search.py` | Orchestrates embedding → vector query → Claude rerank |
| `db.py` | Singletons for embedding model + vector DB (`get_model`, `vector_query`, `vector_upsert_batch`, `vector_count`) |
| `embeddings.py` | Text embedding pipeline, batch upsert |
| `tmdb.py` | TMDB API: fetch, score, ingest movies; `search_person`, `get_filmography` |
| `richtext.py` | Builds `document` string for each movie (what gets embedded) |
| `pipeline.py` | Full init pipeline + single-movie ingestion |
| `api/app.py` | FastAPI app factory, CORS, lifespan |
| `api/routes/search.py` | `POST /recommend` — SSE streaming endpoint |
| `api/routes/admin.py` | `/initialize`, `/status`, `/logs` |
| `logger.py` | JSON request logging |

## Anthropic Integration
- **Round 1**: Always Haiku (`CLAUDE_FAST_MODEL`) — cheap, handles tool-free queries
- **Round 2+**: Switches to Opus (`CLAUDE_MODEL`) only if non-terminal tools are invoked
- **Tools**: `search_person`, `get_filmography`, `return_results` (terminal)
- **Anti-hallucination**: `_filter_results()` validates returned titles against candidate + filmography sets
- **Streaming**: `rerank_stream()` uses `_extract_result_objects()` to yield results as JSON chunks arrive

Key config values:
- `SEARCH_CANDIDATES = 15` — vector results passed to Claude
- `SEARCH_DOC_TRUNCATE = 200` — chars of each movie doc sent in prompt
- `AGENT_MAX_TOOL_ROUNDS = 4`

## Tests
Run with: `pytest` from project root.

| Test file | What it covers |
|-----------|---------------|
| `tests/test_claude.py` | Pure functions: `_filter_results`, `_extract_result_objects` |
| `tests/test_api.py` | FastAPI endpoints via TestClient; mocks `search_stream`, `vector_count`, `get_model` |
| `tests/test_tmdb.py` | Pure scoring functions: `_composite_score`, `select_top_n`, `filter_cast`, `filter_crew` |
| `tests/test_richtext.py` | `build_richtext()` edge cases |

**No test touches the real Anthropic or TMDB APIs.** API-level tests mock `search_stream` at the route layer.

## Environment
Required env vars: `ANTHROPIC_API_KEY`, `TMDB_READ_ACCESS_TOKEN`
Optional: `VECTOR_DB` (default `chroma`), `PINECONE_*` keys, `CORS_ORIGINS`, `RATE_LIMIT`

## Key Decisions
- Haiku-first strategy: most queries resolve in round 1 without Opus
- `tool_choice={"type": "any"}` forces Claude to always call a tool (prevents prose responses)
- Background thread ingests filmography discoveries into vector DB for future queries
- `rerank` and `rerank_stream` share the same prompt/logic; `rerank` is kept for non-streaming use

## Model Use
- Use default model for planning and orchestration, but launch parallel sub-agents with Haiku for execution and research

## Forbidden Directories
Do not read or modify files in these directories:
- `venv/` — Python environment
- `embeddings/` — ChromaDB binary data files
- `data/` — raw TMDB JSON dumps
- `local/` — scratch notes and design proposals (already digested into this file)
- `.git/` — use `git` CLI commands instead
