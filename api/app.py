from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.routes import search, admin

app = FastAPI(
    title="moviematch",
    description="Semantic movie search powered by Claude"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
