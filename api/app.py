import logger  # noqa: F401 — configures logging (stdout + file) before other modules
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.limiter import limiter
from api.routes import search, admin
from config import CORS_ORIGINS

_app_html: str = ""
_admin_html: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app_html, _admin_html
    from db import get_model
    get_model()
    _root = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(_root, "app.html")) as f:
        _app_html = f.read()
    with open(os.path.join(_root, "admin.html")) as f:
        _admin_html = f.read()
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

app.include_router(search.router)
app.include_router(admin.router, prefix="/admin")


@app.get("/")
def root():
    return HTMLResponse(_app_html)


@app.get("/admin.html")
def admin_ui():
    return HTMLResponse(_admin_html)


@app.get("/health")
def health():
    return {"status": "ok"}
