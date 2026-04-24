import logger  # noqa: F401 — configures logging (stdout + file) before other modules
import json
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.auth import SESSION_COOKIE, _cookie_kwargs, set_session_cookie, verify_session_cookie
from api.limiter import limiter
from api.routes import search, admin, streaming
from config import ADMIN_PASSWORD, ADMIN_SECRET_KEY, ADMIN_USERNAME, CORS_ORIGINS, validate_config

_app_html: str = ""
_admin_html: str = ""
_login_html: str = ""
_hints: list = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app_html, _admin_html, _login_html, _hints
    from db import get_model
    from claude import get_client
    validate_config()  # raises ValueError early if ANTHROPIC_API_KEY or TMDB_READ_ACCESS_TOKEN are unset
    get_model()
    get_client()
    _root = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(_root, "app.html")) as f:
        _app_html = f.read()
    with open(os.path.join(_root, "admin.html")) as f:
        _admin_html = f.read()
    with open(os.path.join(_root, "login.html")) as f:
        _login_html = f.read()
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
def admin_ui(request: Request, response: Response):
    if not verify_session_cookie(request.cookies.get(SESSION_COOKIE)):
        return RedirectResponse("/admin/login", status_code=303)
    set_session_cookie(response)
    return HTMLResponse(_admin_html)


class _LoginRequest(BaseModel):
    username: str
    password: str


@app.get("/admin/login")
def login_page(request: Request):
    # Already authenticated — skip the login page.
    if verify_session_cookie(request.cookies.get(SESSION_COOKIE)):
        return RedirectResponse("/admin.html", status_code=303)
    return HTMLResponse(_login_html)


@app.post("/admin/login")
def login_submit(body: _LoginRequest, response: Response):
    """
    Validate credentials and issue a session cookie.
    Both comparisons always run to prevent timing-based username enumeration.
    Fails closed: if any credential env var is unset, access is denied.
    """
    user_ok = secrets.compare_digest(body.username, ADMIN_USERNAME or "\x00")
    pass_ok = secrets.compare_digest(body.password, ADMIN_PASSWORD or "\x00")
    if not (user_ok and pass_ok and ADMIN_USERNAME and ADMIN_PASSWORD and ADMIN_SECRET_KEY):
        return JSONResponse({"detail": "Invalid credentials"}, status_code=401)
    set_session_cookie(response)
    return {"ok": True}


@app.get("/admin/logout")
def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE, **{k: v for k, v in _cookie_kwargs().items() if k != "max_age"})
    return RedirectResponse("/admin/login", status_code=303)


@app.get("/hints.json")
def hints():
    return JSONResponse(_hints)


@app.get("/health")
def health():
    return {"status": "ok"}
