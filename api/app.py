from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.limiter import limiter
from api.routes import search, admin
from config import CORS_ORIGINS


@asynccontextmanager
async def lifespan(app: FastAPI):
    from db import get_model
    get_model()
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
    with open("app.html") as f:
        return HTMLResponse(f.read())


@app.get("/admin.html")
def admin_ui():
    with open("admin.html") as f:
        return HTMLResponse(f.read())


@app.get("/health")
def health():
    return {"status": "ok"}
