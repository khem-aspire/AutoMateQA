"""
Schedules CRUD router — manage cron-scheduled test executions.

Uses test UUID for schedule creation.
"""

from __future__ import annotations

from datetime import datetime, timezone

from apscheduler.triggers.cron import CronTrigger
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from server.dependencies import get_db
from server.models import Schedule, Test
from server.routers.tests import _get_test_by_uuid
from server.schemas import ScheduleCreate, ScheduleResponse, ScheduleUpdate
from server.services.scheduler import scheduler_service

router = APIRouter()


def _compute_next_run(cron_expr: str, tz: str = "UTC") -> datetime | None:
    """Compute the next fire time for a cron expression."""
    try:
        trigger = CronTrigger.from_crontab(cron_expr, timezone=tz)
        return trigger.get_next_fire_time(None, datetime.now(timezone.utc))
    except Exception:
        return None


def _build_response(sched: Schedule) -> ScheduleResponse:
    return ScheduleResponse(
        id=sched.id,
        test_id=sched.test_id,
        test_uuid=sched.test.test_id if sched.test else "",
        test_name=sched.test.name if sched.test else "",
        team_name=sched.test.team.name if sched.test and sched.test.team else "",
        cron_expr=sched.cron_expr,
        timezone=sched.timezone,
        enabled=sched.enabled,
        last_run_at=sched.last_run_at,
        next_run_at=sched.next_run_at,
        created_at=sched.created_at,
        updated_at=sched.updated_at,
    )


@router.post("", response_model=ScheduleResponse, status_code=201)
async def create_schedule(body: ScheduleCreate, db: AsyncSession = Depends(get_db)):
    test = await _get_test_by_uuid(db, body.test_uuid)

    existing = (await db.execute(select(Schedule).where(Schedule.test_id == test.id))).scalar_one_or_none()
    if existing:
        raise HTTPException(409, "Schedule already exists for this test. Update the existing one.")

    next_run = _compute_next_run(body.cron_expr, body.timezone)

    sched = Schedule(
        test_id=test.id,
        cron_expr=body.cron_expr,
        timezone=body.timezone,
        enabled=body.enabled,
        next_run_at=next_run,
    )
    db.add(sched)
    await db.commit()
    await db.refresh(sched, ["test"])

    sched.test = test
    if sched.enabled:
        scheduler_service.add_schedule(sched)

    return _build_response(sched)


@router.get("", response_model=list[ScheduleResponse])
async def list_schedules(db: AsyncSession = Depends(get_db)):
    stmt = (
        select(Schedule)
        .options(selectinload(Schedule.test).selectinload(Test.team))
        .order_by(Schedule.created_at.desc())
    )
    schedules = (await db.execute(stmt)).scalars().all()
    return [_build_response(s) for s in schedules]


@router.get("/{schedule_id}", response_model=ScheduleResponse)
async def get_schedule(schedule_id: int, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(Schedule)
        .options(selectinload(Schedule.test).selectinload(Test.team))
        .where(Schedule.id == schedule_id)
    )
    sched = (await db.execute(stmt)).scalar_one_or_none()
    if not sched:
        raise HTTPException(404, "Schedule not found")
    return _build_response(sched)


@router.put("/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(schedule_id: int, body: ScheduleUpdate, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(Schedule)
        .options(selectinload(Schedule.test).selectinload(Test.team))
        .where(Schedule.id == schedule_id)
    )
    sched = (await db.execute(stmt)).scalar_one_or_none()
    if not sched:
        raise HTTPException(404, "Schedule not found")

    for key, val in body.model_dump(exclude_unset=True).items():
        setattr(sched, key, val)

    # Recompute next_run_at if cron or timezone changed
    sched.next_run_at = _compute_next_run(sched.cron_expr, sched.timezone)

    await db.commit()
    await db.refresh(sched, ["test"])

    if sched.enabled:
        scheduler_service.add_schedule(sched)
    else:
        scheduler_service.remove_schedule(sched.id)

    return _build_response(sched)


@router.delete("/{schedule_id}", status_code=204)
async def delete_schedule(schedule_id: int, db: AsyncSession = Depends(get_db)):
    sched = await db.get(Schedule, schedule_id)
    if not sched:
        raise HTTPException(404, "Schedule not found")
    scheduler_service.remove_schedule(sched.id)
    await db.delete(sched)
    await db.commit()


@router.post("/{schedule_id}/toggle", response_model=ScheduleResponse)
async def toggle_schedule(schedule_id: int, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(Schedule)
        .options(selectinload(Schedule.test).selectinload(Test.team))
        .where(Schedule.id == schedule_id)
    )
    sched = (await db.execute(stmt)).scalar_one_or_none()
    if not sched:
        raise HTTPException(404, "Schedule not found")

    sched.enabled = not sched.enabled
    if sched.enabled:
        sched.next_run_at = _compute_next_run(sched.cron_expr, sched.timezone)
    await db.commit()
    await db.refresh(sched, ["test"])

    if sched.enabled:
        scheduler_service.add_schedule(sched)
    else:
        scheduler_service.remove_schedule(sched.id)

    return _build_response(sched)


@router.get("/debug/jobs")
async def debug_jobs():
    """Debug endpoint: list all registered APScheduler jobs."""
    return scheduler_service.get_jobs_info()
