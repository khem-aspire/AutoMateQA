"""
SQLAlchemy ORM models for the AutoMateQA dashboard.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------
# Teams
# ------------------------------------------------------------------

class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    color: Mapped[str] = mapped_column(String(7), default="#6366f1")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    tests: Mapped[list[Test]] = relationship("Test", back_populates="team")


# ------------------------------------------------------------------
# Categories
# ------------------------------------------------------------------

class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    color: Mapped[str] = mapped_column(String(7), default="#6366f1")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    tests: Mapped[list[Test]] = relationship("Test", back_populates="category")


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class Test(Base):
    __tablename__ = "tests"
    __table_args__ = (
        Index("idx_tests_team", "team_id"),
        Index("idx_tests_category", "category_id"),
        Index("idx_tests_owner", "owner_email"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    test_id: Mapped[str] = mapped_column(String(16), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    base_url: Mapped[str] = mapped_column(String(2048), default="")
    owner_email: Mapped[str] = mapped_column(String(255), nullable=False)
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    category_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("categories.id"), nullable=True)
    s3_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    step_count: Mapped[int] = mapped_column(Integer, default=0)
    tags: Mapped[str] = mapped_column(Text, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    team: Mapped[Team] = relationship("Team", back_populates="tests")
    category: Mapped[Category | None] = relationship("Category", back_populates="tests")
    versions: Mapped[list[TestVersion]] = relationship("TestVersion", back_populates="test", cascade="all, delete-orphan")
    schedule: Mapped[Schedule | None] = relationship("Schedule", back_populates="test", uselist=False, cascade="all, delete-orphan")
    runs: Mapped[list[TestRun]] = relationship("TestRun", back_populates="test", cascade="all, delete-orphan")


# ------------------------------------------------------------------
# Test Versions (S3 file versioning)
# ------------------------------------------------------------------

class TestVersion(Base):
    __tablename__ = "test_versions"
    __table_args__ = (
        Index("idx_versions_test_version", "test_id", "version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    test_id: Mapped[int] = mapped_column(Integer, ForeignKey("tests.id", ondelete="CASCADE"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    s3_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    change_reason: Mapped[str] = mapped_column(String(50), default="upload")  # upload / self_healed / manual_edit
    healed_steps: Mapped[str] = mapped_column(Text, default="")  # JSON list of step_ids
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    test: Mapped[Test] = relationship("Test", back_populates="versions")


# ------------------------------------------------------------------
# Schedules
# ------------------------------------------------------------------

class Schedule(Base):
    __tablename__ = "schedules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    test_id: Mapped[int] = mapped_column(Integer, ForeignKey("tests.id", ondelete="CASCADE"), unique=True, nullable=False)
    cron_expr: Mapped[str] = mapped_column(String(100), default="0 6 * * *")
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    test: Mapped[Test] = relationship("Test", back_populates="schedule")


# ------------------------------------------------------------------
# Test Runs
# ------------------------------------------------------------------

class TestRun(Base):
    __tablename__ = "test_runs"
    __table_args__ = (
        Index("idx_runs_test", "test_id"),
        Index("idx_runs_status", "status"),
        Index("idx_runs_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    test_id: Mapped[int] = mapped_column(Integer, ForeignKey("tests.id", ondelete="CASCADE"), nullable=False)
    test_version: Mapped[int] = mapped_column(Integer, default=1)
    trigger_type: Mapped[str] = mapped_column(String(20), default="manual")  # manual / scheduled / api
    status: Mapped[str] = mapped_column(String(10), default="pending")  # pending/running/passed/failed/healed/error
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    total_duration_ms: Mapped[float] = mapped_column(Double, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    passed_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    healed_count: Mapped[int] = mapped_column(Integer, default=0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    result_json: Mapped[str | None] = mapped_column(LONGTEXT, nullable=True)
    error_message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    test: Mapped[Test] = relationship("Test", back_populates="runs")
    step_results: Mapped[list[StepResult]] = relationship("StepResult", back_populates="run", cascade="all, delete-orphan")


# ------------------------------------------------------------------
# Step Results
# ------------------------------------------------------------------

class StepResult(Base):
    __tablename__ = "step_results"
    __table_args__ = (
        Index("idx_step_results_run", "run_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("test_runs.id", ondelete="CASCADE"), nullable=False)
    step_id: Mapped[int] = mapped_column(Integer, nullable=False)
    action_type: Mapped[str] = mapped_column(String(30), default="")
    status: Mapped[str] = mapped_column(String(10), default="")  # passed / failed / healed
    confidence: Mapped[float] = mapped_column(Double, default=0)
    duration_ms: Mapped[float] = mapped_column(Double, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    healed: Mapped[bool] = mapped_column(Boolean, default=False)
    healing_details: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")

    run: Mapped[TestRun] = relationship("TestRun", back_populates="step_results")
