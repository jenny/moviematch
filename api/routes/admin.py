import ipaddress
import json
import logging
import os
import glob
import threading

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from api.auth import require_admin
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from pipeline import initialize_all
from db import vector_count
from config import DATA_DIR, LOG_DIR
import watchmode as watchmode_module

router = APIRouter()

# Keyed by IP string; values are the most specific location string resolved.
# Survives for the process lifetime so each unique IP is only looked up once.
_ip_location_cache: dict[str, str | None] = {}


def _resolve_location(ip: str) -> str | None:
    """Return the most specific location string available for a public IP.

    Resolution order (most → least specific):
      postal code → city → region (state) → country code

    Returns None for private/loopback addresses or when the lookup fails.
    Uses ipinfo.io free tier (no key required, 50k requests/month).
    """
    if ip in _ip_location_cache:
        return _ip_location_cache[ip]

    # Skip private / loopback addresses without making a network call.
    try:
        addr = ipaddress.ip_address(ip)
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            _ip_location_cache[ip] = None
            return None
    except ValueError:
        _ip_location_cache[ip] = None
        return None

    try:
        resp = httpx.get(f"https://ipinfo.io/{ip}/json", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            # Pick the most specific non-empty field available.
            location = (
                data.get("postal")
                or data.get("city")
                or data.get("region")
                or data.get("country")
                or None
            )
            # Only cache successful resolutions; transient failures (network
            # errors, non-200 responses) are left uncached so the next admin
            # refresh can retry.
            _ip_location_cache[ip] = location
            return location
    except Exception:
        pass

    return None


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


@router.post("/initialize", status_code=202, dependencies=[Depends(require_admin)])
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


@router.get("/status", response_model=StatusResponse, dependencies=[Depends(require_admin)])
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


@router.get("/watchmode", dependencies=[Depends(require_admin)])
def watchmode_stats():
    """Return Watchmode API usage stats for the admin panel.

    Counters are session-lifetime (reset on process restart / Railway deploy).
    """
    return watchmode_module.get_stats()


@router.get("/logs", dependencies=[Depends(require_admin)])
def logs():
    entries = []
    log_path = os.path.join(LOG_DIR, "search.log")
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

    # Resolve unique IPs to location strings (postal → city → region → country).
    # Deduplicate first so N entries with the same IP only cost one lookup.
    recent = list(reversed(entries[-50:]))
    unique_ips = {e["client_ip"] for e in recent if e.get("client_ip")}
    ip_to_location = {ip: _resolve_location(ip) for ip in unique_ips}
    for entry in recent:
        if ip := entry.get("client_ip"):
            entry["location"] = ip_to_location.get(ip)

    return {"entries": recent, "stats": stats}
