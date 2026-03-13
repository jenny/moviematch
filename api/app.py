from fastapi import FastAPI

from api.routes import search, admin

app = FastAPI(
    title="moviematch",
    description="Semantic movie search powered by Claude"
)

app.include_router(search.router)
app.include_router(admin.router, prefix="/admin")
