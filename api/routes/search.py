from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from search import search

router = APIRouter()


class SearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    title: str
    explanation: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


@router.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest):
    try:
        results = search(request.query)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return SearchResponse(query=request.query, results=results)
