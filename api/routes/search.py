import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import HAIKU_INPUT_PRICE, HAIKU_OUTPUT_PRICE, OPUS_INPUT_PRICE, OPUS_OUTPUT_PRICE
from logger import log_request
from search import search

router = APIRouter()


class SearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    title: str
    explanation: str
    movie_poster: str = ""


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    message: str | None = None
    usage: dict | None = None


@router.post("/recommend", response_model=SearchResponse)
def search_endpoint(request: SearchRequest):
    timestamp = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    try:
        results, usage, timing = search(request.query)
    except RuntimeError as e:
        total_ms = round((time.perf_counter() - t0) * 1000)
        log_request({
            "timestamp": timestamp,
            "query": request.query,
            "status": "error",
            "error": str(e),
            "total_ms": total_ms,
        })
        raise HTTPException(status_code=503, detail=str(e))

    total_ms = round((time.perf_counter() - t0) * 1000)

    estimated_cost_usd = None
    if usage:
        estimated_cost_usd = round(
            usage.get("haiku_input_tokens", 0)  * HAIKU_INPUT_PRICE  +
            usage.get("haiku_output_tokens", 0) * HAIKU_OUTPUT_PRICE +
            usage.get("opus_input_tokens", 0)   * OPUS_INPUT_PRICE   +
            usage.get("opus_output_tokens", 0)  * OPUS_OUTPUT_PRICE,
            6,
        )

    log_request({
        "timestamp": timestamp,
        "query": request.query,
        "status": "ok",
        "result_count": len(results),
        "embedding_ms": timing["embedding_ms"],
        "chroma_ms": timing["chroma_ms"],
        "claude_ms": timing["claude_ms"],
        "total_ms": total_ms,
        "input_tokens": usage.get("input_tokens") if usage else None,
        "output_tokens": usage.get("output_tokens") if usage else None,
        "haiku_input_tokens": usage.get("haiku_input_tokens") if usage else None,
        "haiku_output_tokens": usage.get("haiku_output_tokens") if usage else None,
        "opus_input_tokens": usage.get("opus_input_tokens") if usage else None,
        "opus_output_tokens": usage.get("opus_output_tokens") if usage else None,
        "claude_rounds": usage.get("rounds") if usage else None,
        "tools_called": usage.get("tools_called") if usage else None,
        "estimated_cost_usd": estimated_cost_usd,
    })

    if not results:
        return SearchResponse(query=request.query, results=[], message="No relevant matches found.", usage=usage)
    return SearchResponse(query=request.query, results=results, usage=usage)
