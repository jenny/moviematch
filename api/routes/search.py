import json
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import HAIKU_INPUT_PRICE, HAIKU_OUTPUT_PRICE, OPUS_INPUT_PRICE, OPUS_OUTPUT_PRICE
from logger import log_request
from search import search_stream

router = APIRouter()


class SearchRequest(BaseModel):
    query: str


@router.post("/recommend")
def search_endpoint(request: SearchRequest):
    timestamp = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    def generate():
        result_count = 0
        try:
            for item in search_stream(request.query):
                if "__meta" in item:
                    meta = item["__meta"]
                    usage = meta.get("usage")
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
                        "result_count": result_count,
                        "embedding_ms": meta["embedding_ms"],
                        "chroma_ms": meta["chroma_ms"],
                        "claude_ms": meta["claude_ms"],
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

                    done = {"type": "done", "result_count": result_count}
                    if not result_count:
                        done["message"] = "No relevant matches found."
                    yield f"data: {json.dumps(done)}\n\n"
                else:
                    result_count += 1
                    yield f"data: {json.dumps({'type': 'result', **item})}\n\n"

        except RuntimeError as e:
            total_ms = round((time.perf_counter() - t0) * 1000)
            log_request({
                "timestamp": timestamp,
                "query": request.query,
                "status": "error",
                "error": str(e),
                "total_ms": total_ms,
            })
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
