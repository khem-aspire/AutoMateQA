"""
Pydantic v2 request/response schemas for the AutoMateQA dashboard API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


# ------------------------------------------------------------------
# Teams
# ------------------------------------------------------------------

class TeamCreate(BaseModel):
    name: str = Field(..., max_length=100)
    slug: str = Field(..., max_length=100)
    description: str = ""
    color: str = Field(default="#6366f1", max_length=7)


class TeamUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    slug: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    color: Optional[str] = Field(None, max_length=7)


class TeamResponse(BaseModel):
    id: int
    name: str
    slug: str
    description: str
    color: str
    created_at: datetime
    updated_at: datetime
    test_count: int = 0

    model_config = {"from_attributes": True}


# ------------------------------------------------------------------
# Categories
# ------------------------------------------------------------------

class CategoryCreate(BaseModel):
    name: str = Field(..., max_length=100)
    description: str = ""
    color: str = Field(default="#6366f1", max_length=7)


class CategoryUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    color: Optional[str] = Field(None, max_length=7)


class CategoryResponse(BaseModel):
    id: int
    name: str
    description: str
    color: str
    created_at: datetime
    test_count: int = 0

    model_config = {"from_attributes": True}


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestUploadMeta(BaseModel):
    name: str = Field(..., max_length=255)
    description: str = ""
    owner_email: EmailStr
    team_id: int
    category_id: Optional[int] = None
    tags: str = ""


class TestUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    owner_email: Optional[EmailStr] = None
    team_id: Optional[int] = None
    category_id: Optional[int] = None
    tags: Optional[str] = None
    is_active: Optional[bool] = None


class TestResponse(BaseModel):
    id: int
    test_id: str
    name: str
    description: str
    base_url: str
    owner_email: str
    team_id: int
    team_name: str = ""
    category_id: Optional[int] = None
    category_name: str = ""
    s3_key: str
    current_version: int
    step_count: int
    tags: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_run_status: Optional[str] = None

    model_config = {"from_attributes": True}


class TestVersionResponse(BaseModel):
    id: int
    version: int
    s3_key: str
    change_reason: str
    healed_steps: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ------------------------------------------------------------------
# Schedules
# ------------------------------------------------------------------

class ScheduleCreate(BaseModel):
    test_uuid: str
    cron_expr: str = Field(default="0 6 * * *", max_length=100)
    timezone: str = Field(default="UTC", max_length=50)
    enabled: bool = True


class ScheduleUpdate(BaseModel):
    cron_expr: Optional[str] = Field(None, max_length=100)
    timezone: Optional[str] = Field(None, max_length=50)
    enabled: Optional[bool] = None


class ScheduleResponse(BaseModel):
    id: int
    test_id: int
    test_uuid: str = ""
    test_name: str = ""
    team_name: str = ""
    cron_expr: str
    timezone: str
    enabled: bool
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ------------------------------------------------------------------
# Test Runs
# ------------------------------------------------------------------

class RunTriggerRequest(BaseModel):
    healing_mode: str = "disabled"
    llm_enabled: bool = False
    llm_model: str = "gpt-4o"
    llm_provider: str = "openai"
    confidence_threshold: float = 0.75
    retry_strategy: str = "exponential"
    max_step_retries: int = 3


class RunResponse(BaseModel):
    id: int
    run_id: str
    test_id: int
    test_uuid: str = ""
    test_name: str = ""
    team_name: str = ""
    test_version: int
    trigger_type: str
    status: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    total_duration_ms: float
    total_steps: int
    passed_count: int
    failed_count: int
    healed_count: int
    tokens_used: int
    error_message: str
    created_at: datetime

    model_config = {"from_attributes": True}


class RunDetailResponse(RunResponse):
    result_json: Optional[str] = None


class StepResultResponse(BaseModel):
    id: int
    step_id: int
    action_type: str
    status: str
    confidence: float
    duration_ms: float
    retry_count: int
    healed: bool
    healing_details: str
    error: str

    model_config = {"from_attributes": True}


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------

class OverviewStats(BaseModel):
    total_tests: int
    total_runs: int
    runs_today: int
    pass_rate: float
    avg_duration_ms: float
    total_healed: int


class TrendPoint(BaseModel):
    date: str
    passed: int
    failed: int
    healed: int


class CategoryBreakdown(BaseModel):
    category_name: str
    color: str
    test_count: int
    pass_rate: float


class TeamBreakdown(BaseModel):
    team_name: str
    color: str
    test_count: int
    run_count: int
    pass_rate: float


# ------------------------------------------------------------------
# Pagination
# ------------------------------------------------------------------

class PaginatedResponse(BaseModel):
    items: list
    total: int
    page: int
    per_page: int
    pages: int
