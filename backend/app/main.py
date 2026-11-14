"""
backend/app/main.py
FastAPI application entry point — Phase 1.

Endpoints
---------
GET /health  →  liveness probe; returns service name, version, and status.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Resolve configuration relative to the repository, not the shell's working
# directory. A non-empty process environment value still takes precedence.
PROJECT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=PROJECT_ENV_FILE, override=False)
if not os.getenv("GEMINI_API_KEY", "").strip():
    dotenv_key = dotenv_values(PROJECT_ENV_FILE, interpolate=False).get("GEMINI_API_KEY")
    if isinstance(dotenv_key, str) and dotenv_key.strip():
        os.environ["GEMINI_API_KEY"] = dotenv_key.strip()

from backend.app.routers import prescriptions_router
from backend.app.routers.medicine_chat import router as medicine_chat_router

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medical Prescription OCR API",
    description=(
        "REST backend for the Medical Prescription OCR & Drug Interaction Checker. "
        "Review-gated extraction and normalization workflows are available. "
        "No drug-interaction evidence provider is configured by default."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the Streamlit frontend (running on a different port) to call the API.
_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["accept", "content-type"],
)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    service: str
    version: str
    timestamp: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["system"],
)
async def health() -> HealthResponse:
    """
    Liveness probe.

    Returns the service name, API version, and a UTC timestamp so that
    upstream load-balancers and the Streamlit frontend can verify the
    backend is reachable.
    """
    return HealthResponse(
        status="ok",
        service="medical-prescription-ocr",
        version=app.version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# Register sub-routers
app.include_router(prescriptions_router)
app.include_router(medicine_chat_router)
