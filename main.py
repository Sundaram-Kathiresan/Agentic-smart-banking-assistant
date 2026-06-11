# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1.routes.query_routes import router as rag_router

# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="RAG PDF Upload API",
    version="1.0.0"
)

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# INCLUDE ROUTES
# =========================

app.include_router(rag_router)

# =========================
# ROOT ROUTE
# =========================

@app.get("/")
async def root():
    return {
        "message": "RAG API Running Successfully"
    }