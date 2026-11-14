"""
backend/app/routers/__init__.py
Sub-routers for the FastAPI backend.
"""

from __future__ import annotations

from backend.app.routers.prescriptions import router as prescriptions_router

__all__ = ["prescriptions_router"]

