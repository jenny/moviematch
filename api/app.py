import logger  # noqa: F401 — configures logging (stdout + file) before other modules
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.limiter import limiter
from api.routes import search, admin, streaming
from config import CORS_ORIGINS

_app_html: str = ""
_admin_html: str = ""
_hints: list = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app_html, _admin_html, _hints
    from db import get_model
    from claude import get_client
    get_model()
    get_client()  # validates ANTHROPIC_API_KEY early; raises ValueError if missing
    _root = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(_root, "app.html")) as f:
        _app_html = f.read()
    with open(os.path.join(_root, "admin.html")) as f:
        _admin_html = f.read()
    hints_path = os.path.join(_root, "hints.json")
    if os.path.exists(hints_path):
        with open(hints_path) as f:
            _hints = json.load(f)
    yield


app = FastAPI(
    title="moviematch",
    description="Semantic movie search powered by Claude",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def redirect_www(request: Request, call_next):
    host = request.headers.get("host", "")
    if host.startswith("www."):
        url = request.url.replace(netloc=host[4:])
        return RedirectResponse(url=str(url), status_code=301)
    return await call_next(request)

app.include_router(search.router)
app.include_router(admin.router, prefix="/admin")
app.include_router(streaming.router)


@app.get("/")
def root():
    return HTMLResponse(_app_html)


@app.get("/admin.html")
def admin_ui():
    return HTMLResponse(_admin_html)


@app.get("/hints.json")
def hints():
    return JSONResponse(_hints)


@app.get("/health")
def health():
    return {"status": "ok"}
