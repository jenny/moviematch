from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/health")
def health():
    return {"status": "ok"}
