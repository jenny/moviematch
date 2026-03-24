import json
import logging
import os
import glob
import threading

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from pipeline import initialize_all
from db import vector_count
from config import DATA_DIR

router = APIRouter()

_init_lock = threading.Lock()
_init_status = {"running": False, "last_result": None}


class InitializeRequest(BaseModel):
    n: int


class LastInitResult(BaseModel):
    movie_count: int
    indexed_count: int
    embedded_count: int
    failed_count: int
    failed_ids: list[int]


class StatusResponse(BaseModel):
    movie_count: int
    chroma_count: int
    initializing: bool
    last_init_result: LastInitResult | None = None


def _run_pipeline(n: int):
    try:
        result = initialize_all(n)
        _init_status["last_result"] = result
    finally:
        with _init_lock:
            _init_status["running"] = False


@router.post("/initialize", status_code=202)
def initialize(request: InitializeRequest, background_tasks: BackgroundTasks):
    with _init_lock:
        if _init_status["running"]:
            raise HTTPException(status_code=409, detail="Initialization already in progress.")
        _init_status["running"] = True
    try:
        background_tasks.add_task(_run_pipeline, request.n)
    except Exception:
        with _init_lock:
            _init_status["running"] = False
        raise
    return {"message": "Initialization started", "n": request.n}


@router.get("/status", response_model=StatusResponse)
def status():
    movie_files = [
        f for f in glob.glob(os.path.join(DATA_DIR, "*.json"))
        if os.path.basename(f) != "index.json"
    ]
    try:
        chroma_count = vector_count()
    except Exception:
        chroma_count = 0
    return StatusResponse(
        movie_count=len(movie_files),
        chroma_count=chroma_count,
        initializing=_init_status["running"],
        last_init_result=_init_status["last_result"]
    )


@router.get("/logs")
def logs():
    entries = []
    log_path = "logs/search.log"
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Malformed JSON in log file, skipping line: {line[:100]!r}")

    ok = [e for e in entries if e.get("status") == "ok"]

    def avg(values):
        return round(sum(values) / len(values)) if values else None

    def pct(sorted_values, p):
        if not sorted_values:
            return None
        return sorted_values[min(int(len(sorted_values) * p / 100), len(sorted_values) - 1)]

    def collect(key):
        return sorted([e[key] for e in ok if e.get(key) is not None])

    tool_queries = [e for e in ok if e.get("tools_called")]
    costs = [e["estimated_cost_usd"] for e in ok if e.get("estimated_cost_usd") is not None]
    recent_costs = [e["estimated_cost_usd"] for e in ok[-50:] if e.get("estimated_cost_usd") is not None]

    total_ms   = collect("total_ms")
    claude_ms  = collect("claude_ms")
    embed_ms   = collect("embedding_ms")
    chroma_ms  = collect("chroma_ms")

    stats = {
        "total_requests": len(entries),
        "error_count": len([e for e in entries if e.get("status") == "error"]),
        "avg_total_ms":      avg(total_ms),
        "p50_total_ms":      pct(total_ms, 50),
        "p90_total_ms":      pct(total_ms, 90),
        "p99_total_ms":      pct(total_ms, 99),
        "p50_claude_ms":     pct(claude_ms, 50),
        "p90_claude_ms":     pct(claude_ms, 90),
        "p99_claude_ms":     pct(claude_ms, 99),
        "p50_embedding_ms":  pct(embed_ms, 50),
        "p90_embedding_ms":  pct(embed_ms, 90),
        "p99_embedding_ms":  pct(embed_ms, 99),
        "p50_chroma_ms":     pct(chroma_ms, 50),
        "p90_chroma_ms":     pct(chroma_ms, 90),
        "p99_chroma_ms":     pct(chroma_ms, 99),
        "avg_embedding_ms":  avg(embed_ms),
        "avg_chroma_ms":     avg(chroma_ms),
        "avg_claude_ms":     avg(claude_ms),
        "avg_input_tokens":  avg([e["input_tokens"] for e in ok if e.get("input_tokens") is not None]),
        "avg_output_tokens": avg([e["output_tokens"] for e in ok if e.get("output_tokens") is not None]),
        "tool_use_rate": round(len(tool_queries) / len(ok) * 100) if ok else None,
        "avg_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
        "total_cost_usd": round(sum(costs), 6) if costs else None,
        "recent_avg_cost_usd": round(sum(recent_costs) / len(recent_costs), 6) if recent_costs else None,
        "recent_query_count": len(recent_costs),
    }

    return {"entries": list(reversed(entries[-50:])), "stats": stats}
