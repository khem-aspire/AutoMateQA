"""
Stats router — dashboard overview, trends, breakdowns.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, cast, func, select, Date
from sqlalchemy.ext.asyncio import AsyncSession

from server.dependencies import get_db
from server.models import Category, Team, Test, TestRun
from server.schemas import (
    CategoryBreakdown,
    OverviewStats,
    TeamBreakdown,
    TrendPoint,
)

router = APIRouter()


@router.get("/overview", response_model=OverviewStats)
async def get_overview(db: AsyncSession = Depends(get_db)):
    total_tests = (await db.execute(select(func.count(Test.id)))).scalar() or 0

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    total_runs = (await db.execute(select(func.count(TestRun.id)))).scalar() or 0
    runs_today = (
        await db.execute(
            select(func.count(TestRun.id)).where(TestRun.created_at >= today_start)
        )
    ).scalar() or 0

    # Pass rate (passed + healed) / total completed runs
    completed = (
        await db.execute(
            select(func.count(TestRun.id)).where(TestRun.status.in_(["passed", "failed", "healed"]))
        )
    ).scalar() or 0
    passed = (
        await db.execute(
            select(func.count(TestRun.id)).where(TestRun.status.in_(["passed", "healed"]))
        )
    ).scalar() or 0
    pass_rate = (passed / completed * 100) if completed > 0 else 0.0

    avg_duration = (
        await db.execute(
            select(func.avg(TestRun.total_duration_ms)).where(
                TestRun.status.in_(["passed", "failed", "healed"])
            )
        )
    ).scalar() or 0.0

    total_healed = (
        await db.execute(select(func.sum(TestRun.healed_count)))
    ).scalar() or 0

    return OverviewStats(
        total_tests=total_tests,
        total_runs=total_runs,
        runs_today=runs_today,
        pass_rate=round(pass_rate, 1),
        avg_duration_ms=round(avg_duration, 0),
        total_healed=total_healed,
    )


@router.get("/trends", response_model=list[TrendPoint])
async def get_trends(days: int = Query(30, ge=1, le=90), db: AsyncSession = Depends(get_db)):
    since = datetime.now(timezone.utc) - timedelta(days=days)

    stmt = (
        select(
            cast(TestRun.created_at, Date).label("date"),
            func.sum(case((TestRun.status == "passed", 1), else_=0)).label("passed"),
            func.sum(case((TestRun.status == "failed", 1), else_=0)).label("failed"),
            func.sum(case((TestRun.status == "healed", 1), else_=0)).label("healed"),
        )
        .where(TestRun.created_at >= since, TestRun.status.in_(["passed", "failed", "healed"]))
        .group_by(cast(TestRun.created_at, Date))
        .order_by(cast(TestRun.created_at, Date))
    )
    rows = (await db.execute(stmt)).all()
    return [
        TrendPoint(date=str(row.date), passed=row.passed, failed=row.failed, healed=row.healed)
        for row in rows
    ]


@router.get("/category-breakdown", response_model=list[CategoryBreakdown])
async def get_category_breakdown(db: AsyncSession = Depends(get_db)):
    stmt = (
        select(
            Category.name,
            Category.color,
            func.count(Test.id).label("test_count"),
        )
        .outerjoin(Test, Test.category_id == Category.id)
        .group_by(Category.id)
        .order_by(Category.name)
    )
    rows = (await db.execute(stmt)).all()

    result = []
    for row in rows:
        # Get pass rate for this category's tests
        cat_test_ids = (
            await db.execute(select(Test.id).where(Test.category_id == (
                await db.execute(select(Category.id).where(Category.name == row.name))
            ).scalar()))
        )
        test_ids = cat_test_ids.scalars().all()

        pass_rate = 0.0
        if test_ids:
            completed = (
                await db.execute(
                    select(func.count(TestRun.id)).where(
                        TestRun.test_id.in_(test_ids),
                        TestRun.status.in_(["passed", "failed", "healed"]),
                    )
                )
            ).scalar() or 0
            passed = (
                await db.execute(
                    select(func.count(TestRun.id)).where(
                        TestRun.test_id.in_(test_ids),
                        TestRun.status.in_(["passed", "healed"]),
                    )
                )
            ).scalar() or 0
            pass_rate = (passed / completed * 100) if completed > 0 else 0.0

        result.append(CategoryBreakdown(
            category_name=row.name,
            color=row.color,
            test_count=row.test_count,
            pass_rate=round(pass_rate, 1),
        ))
    return result


@router.get("/team-breakdown", response_model=list[TeamBreakdown])
async def get_team_breakdown(db: AsyncSession = Depends(get_db)):
    stmt = (
        select(
            Team.name,
            Team.color,
            func.count(func.distinct(Test.id)).label("test_count"),
        )
        .outerjoin(Test, Test.team_id == Team.id)
        .group_by(Team.id)
        .order_by(Team.name)
    )
    rows = (await db.execute(stmt)).all()

    result = []
    for row in rows:
        test_ids = (
            await db.execute(
                select(Test.id).join(Team).where(Team.name == row.name)
            )
        ).scalars().all()

        run_count = 0
        pass_rate = 0.0
        if test_ids:
            run_count = (
                await db.execute(
                    select(func.count(TestRun.id)).where(TestRun.test_id.in_(test_ids))
                )
            ).scalar() or 0
            completed = (
                await db.execute(
                    select(func.count(TestRun.id)).where(
                        TestRun.test_id.in_(test_ids),
                        TestRun.status.in_(["passed", "failed", "healed"]),
                    )
                )
            ).scalar() or 0
            passed = (
                await db.execute(
                    select(func.count(TestRun.id)).where(
                        TestRun.test_id.in_(test_ids),
                        TestRun.status.in_(["passed", "healed"]),
                    )
                )
            ).scalar() or 0
            pass_rate = (passed / completed * 100) if completed > 0 else 0.0

        result.append(TeamBreakdown(
            team_name=row.name,
            color=row.color,
            test_count=row.test_count,
            run_count=run_count,
            pass_rate=round(pass_rate, 1),
        ))
    return result
