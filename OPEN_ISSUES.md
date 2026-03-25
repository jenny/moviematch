# Open Issues

Issues identified during code review that are deferred for later.

---

## Issue 9 — `search_endpoint` blocks the uvicorn thread pool (architectural)

**File:** `api/routes/search.py:28`

`search_endpoint` is a synchronous `def`, so FastAPI runs it in a threadpool executor. The `generate()` generator inside it performs blocking I/O (embedding, vector DB query, Claude streaming) while holding that thread for the full duration of a streaming response — which can be several seconds.

Under concurrent load, this exhausts uvicorn's default threadpool (40 threads), queuing or dropping new requests.

**Fix:** Convert the route to `async def` and move the blocking `_fetch_candidates()` call into `asyncio.get_event_loop().run_in_executor()`. The Claude streaming loop would need to be wrapped in a thread executor as well, since the Anthropic SDK's streaming context manager is synchronous.

---

## Minor

- **`model_log` computed twice on exhausted-loop path** (`claude.py`): `"→".join(...)` is built once before the `usage` dict and then implicitly again in the log message. Trivial to deduplicate but no functional impact.
