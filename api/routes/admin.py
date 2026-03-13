import os
import glob
import threading

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from pipeline import initialize_all
from db import get_or_create_collection
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


class StatusResponse(BaseModel):
    movie_count: int
    chroma_count: int
    initializing: bool
    last_init_result: LastInitResult | None = None


def _run_pipeline(n: int):
    with _init_lock:
        _init_status["running"] = True
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
    background_tasks.add_task(_run_pipeline, request.n)
    return {"message": "Initialization started", "n": request.n}


@router.get("/status", response_model=StatusResponse)
def status():
    movie_files = [
        f for f in glob.glob(os.path.join(DATA_DIR, "*.json"))
        if "index" not in f
    ]
    try:
        chroma_count = get_or_create_collection().count()
    except Exception:
        chroma_count = 0
    return StatusResponse(
        movie_count=len(movie_files),
        chroma_count=chroma_count,
        initializing=_init_status["running"],
        last_init_result=_init_status["last_result"]
    )
