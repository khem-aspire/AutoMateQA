"""
Runs router — trigger execution, list runs, get details, export reports.

Test-specific endpoints use the test UUID (test_id string), not the DB integer id.
"""

from __future__ import annotations

import logging
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from server.dependencies import get_db
from server.models import StepResult as StepResultModel
from server.models import Test, TestRun
from server.routers.tests import _get_test_by_uuid
from server.schemas import (
    PaginatedResponse,
    RunDetailResponse,
    RunResponse,
    RunTriggerRequest,
    StepResultResponse,
)
from server.services.executor import execution_manager

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_run_response(run: TestRun) -> RunResponse:
    return RunResponse(
        id=run.id,
        run_id=run.run_id,
        test_id=run.test_id,
        test_uuid=run.test.test_id if run.test else "",
        test_name=run.test.name if run.test else "",
        team_name=run.test.team.name if run.test and run.test.team else "",
        test_version=run.test_version,
        trigger_type=run.trigger_type,
        status=run.status,
        started_at=run.started_at,
        finished_at=run.finished_at,
        total_duration_ms=run.total_duration_ms,
        total_steps=run.total_steps,
        passed_count=run.passed_count,
        failed_count=run.failed_count,
        healed_count=run.healed_count,
        tokens_used=run.tokens_used,
        error_message=run.error_message,
        created_at=run.created_at,
    )


@router.post("/{test_uuid}/trigger", status_code=202)
async def trigger_run(
    test_uuid: str,
    body: RunTriggerRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Trigger a manual test execution by test UUID. Returns run_id immediately."""
    test = await _get_test_by_uuid(db, test_uuid)

    config_overrides = body.model_dump() if body else {}
    run_id = await execution_manager.trigger_run(
        test_db_id=test.id,
        test_id_str=test.test_id,
        s3_key=test.s3_key,
        current_version=test.current_version,
        owner_email=test.owner_email,
        trigger_type="manual",
        config_overrides=config_overrides,
    )
    return {"run_id": run_id, "status": "pending"}


@router.get("", response_model=PaginatedResponse)
async def list_runs(
    test_uuid: Optional[str] = Query(None),
    team_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(TestRun)
        .options(selectinload(TestRun.test).selectinload(Test.team))
    )
    count_stmt = select(func.count(TestRun.id))

    if test_uuid is not None:
        test = await _get_test_by_uuid(db, test_uuid)
        stmt = stmt.where(TestRun.test_id == test.id)
        count_stmt = count_stmt.where(TestRun.test_id == test.id)
    if team_id is not None:
        stmt = stmt.join(Test).where(Test.team_id == team_id)
        count_stmt = count_stmt.join(Test).where(Test.team_id == team_id)
    if status:
        stmt = stmt.where(TestRun.status == status)
        count_stmt = count_stmt.where(TestRun.status == status)

    total = (await db.execute(count_stmt)).scalar() or 0
    stmt = stmt.order_by(TestRun.created_at.desc()).offset((page - 1) * per_page).limit(per_page)
    runs = (await db.execute(stmt)).scalars().all()

    return PaginatedResponse(
        items=[_build_run_response(r) for r in runs],
        total=total,
        page=page,
        per_page=per_page,
        pages=(total + per_page - 1) // per_page if total > 0 else 0,
    )


@router.get("/{run_id}")
async def get_run(run_id: str, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(TestRun)
        .options(selectinload(TestRun.test).selectinload(Test.team))
        .where(TestRun.run_id == run_id)
    )
    run = (await db.execute(stmt)).scalar_one_or_none()
    if not run:
        raise HTTPException(404, "Run not found")
    resp = _build_run_response(run)
    return RunDetailResponse(**resp.model_dump(), result_json=run.result_json)


@router.get("/{run_id}/steps", response_model=list[StepResultResponse])
async def get_run_steps(run_id: str, db: AsyncSession = Depends(get_db)):
    run = (await db.execute(select(TestRun).where(TestRun.run_id == run_id))).scalar_one_or_none()
    if not run:
        raise HTTPException(404, "Run not found")
    stmt = select(StepResultModel).where(StepResultModel.run_id == run.id).order_by(StepResultModel.step_id)
    steps = (await db.execute(stmt)).scalars().all()
    return [StepResultResponse.model_validate(s) for s in steps]


@router.get("/{run_id}/report")
async def export_report(
    run_id: str,
    format: str = Query("html", pattern="^(html|json|junit)$"),
    db: AsyncSession = Depends(get_db),
):
    """Export a test run report in HTML, JSON, or JUnit XML format."""
    run = (await db.execute(select(TestRun).where(TestRun.run_id == run_id))).scalar_one_or_none()
    if not run:
        raise HTTPException(404, "Run not found")
    if not run.result_json:
        raise HTTPException(400, "Run has no results yet")

    from engine.models import TestResult
    from engine.reporter import ReportGenerator

    result = TestResult.model_validate_json(run.result_json)
    reporter = ReportGenerator()

    ext_map = {"html": ".html", "json": ".json", "junit": ".xml"}
    content_type_map = {"html": "text/html", "json": "application/json", "junit": "application/xml"}

    suffix = ext_map[format]
    import os
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    await reporter.generate(result, format, tmp_path)

    return FileResponse(
        tmp_path,
        media_type=content_type_map[format],
        filename=f"report_{run_id}{suffix}",
        headers={"Content-Disposition": f'attachment; filename="report_{run_id}{suffix}"'},
    )


@router.get("/test/{test_uuid}/runs", response_model=PaginatedResponse)
async def get_test_runs(
    test_uuid: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    test = await _get_test_by_uuid(db, test_uuid)
    stmt = (
        select(TestRun)
        .options(selectinload(TestRun.test).selectinload(Test.team))
        .where(TestRun.test_id == test.id)
    )
    count_stmt = select(func.count(TestRun.id)).where(TestRun.test_id == test.id)

    total = (await db.execute(count_stmt)).scalar() or 0
    stmt = stmt.order_by(TestRun.created_at.desc()).offset((page - 1) * per_page).limit(per_page)
    runs = (await db.execute(stmt)).scalars().all()

    return PaginatedResponse(
        items=[_build_run_response(r) for r in runs],
        total=total,
        page=page,
        per_page=per_page,
        pages=(total + per_page - 1) // per_page if total > 0 else 0,
    )


@router.get("/test/{test_uuid}/flakiness")
async def get_test_flakiness(test_uuid: str, last_n: int = Query(20, ge=5, le=100), db: AsyncSession = Depends(get_db)):
    """Per-step flakiness data across recent runs."""
    test = await _get_test_by_uuid(db, test_uuid)

    run_ids_stmt = (
        select(TestRun.id)
        .where(TestRun.test_id == test.id, TestRun.status.in_(["passed", "failed", "healed"]))
        .order_by(TestRun.created_at.desc())
        .limit(last_n)
    )
    run_ids = (await db.execute(run_ids_stmt)).scalars().all()
    if not run_ids:
        return []

    stmt = (
        select(StepResultModel)
        .where(StepResultModel.run_id.in_(run_ids))
        .order_by(StepResultModel.step_id)
    )
    steps = (await db.execute(stmt)).scalars().all()

    from collections import defaultdict
    step_data: dict[int, list[str]] = defaultdict(list)
    for s in steps:
        step_data[s.step_id].append(s.status)

    result = []
    for step_id, statuses in sorted(step_data.items()):
        if len(statuses) < 2:
            flakiness = 0.0
        else:
            flips = sum(1 for i in range(1, len(statuses)) if statuses[i] != statuses[i - 1])
            flakiness = flips / (len(statuses) - 1)
        result.append({
            "step_id": step_id,
            "total_runs": len(statuses),
            "passed": statuses.count("passed"),
            "failed": statuses.count("failed"),
            "healed": statuses.count("healed"),
            "flakiness_score": round(flakiness, 3),
        })

    return result
