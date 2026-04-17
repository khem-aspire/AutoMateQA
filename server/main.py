"""
AutoMateQA Dashboard — FastAPI application entry point.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.database import engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle events."""
    # -- Startup --
    logger.info("AutoMateQA Dashboard starting up...")

    # Start the APScheduler (lazy import to avoid circular deps)
    from server.services.scheduler import scheduler_service
    await scheduler_service.start()

    yield

    # -- Shutdown --
    logger.info("AutoMateQA Dashboard shutting down...")
    await scheduler_service.stop()
    await engine.dispose()


app = FastAPI(
    title="AutoMateQA Dashboard",
    description="API for managing and executing self-healing test automation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("AQA_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
from server.routers import categories, runs, schedules, stats, teams, tests, ws  # noqa: E402

app.include_router(teams.router, prefix="/api/teams", tags=["teams"])
app.include_router(categories.router, prefix="/api/categories", tags=["categories"])
app.include_router(tests.router, prefix="/api/tests", tags=["tests"])
app.include_router(runs.router, prefix="/api/runs", tags=["runs"])
app.include_router(schedules.router, prefix="/api/schedules", tags=["schedules"])
app.include_router(stats.router, prefix="/api/stats", tags=["stats"])
app.include_router(ws.router, prefix="/api/ws", tags=["websocket"])


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "automateqa-dashboard"}
