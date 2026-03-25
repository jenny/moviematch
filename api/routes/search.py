import json
import logging
import time
import traceback
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from api.limiter import limiter
from config import HAIKU_INPUT_PRICE, HAIKU_OUTPUT_PRICE, OPUS_INPUT_PRICE, OPUS_OUTPUT_PRICE
from config import RATE_LIMIT
from logger import log_request
from main import search_stream

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


@router.post("/recommend")
@limiter.limit(RATE_LIMIT)
def search_endpoint(request: Request, body: SearchRequest):
    timestamp = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    def generate():
        result_count = 0
        try:
            for item in search_stream(body.query):
                if "__meta" in item:
                    meta = item["__meta"]
                    usage = meta.get("usage")
                    error = meta.get("error")
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
                        "query": body.query,
                        "status": "error" if error else "ok",
                        "result_count": result_count,
                        "embedding_ms": meta.get("embedding_ms"),
                        "chroma_ms": meta.get("chroma_ms"),
                        "claude_ms": meta.get("claude_ms"),
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
                        "error": error,
                    })

                    if error:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Could not connect to the AI service. Please try again.'})}\n\n"
                    else:
                        done = {"type": "done", "result_count": result_count}
                        if not result_count:
                            done["message"] = "No relevant matches found."
                        yield f"data: {json.dumps(done)}\n\n"
                else:
                    result_count += 1
                    yield f"data: {json.dumps({'type': 'result', **item})}\n\n"

        except Exception as e:
            total_ms = round((time.perf_counter() - t0) * 1000)
            tb = traceback.format_exc()
            logger.error("Unhandled exception in search_endpoint:\n%s", tb)
            log_request({
                "timestamp": timestamp,
                "query": body.query,
                "status": "error",
                "error": str(e),
                "traceback": tb,
                "total_ms": total_ms,
            })
            yield f"data: {json.dumps({'type': 'error', 'message': 'An error occurred. Please try again.'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
