"""
Teams CRUD router.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from server.dependencies import get_db
from server.models import Team, Test
from server.schemas import TeamCreate, TeamResponse, TeamUpdate

router = APIRouter()


@router.post("", response_model=TeamResponse, status_code=201)
async def create_team(body: TeamCreate, db: AsyncSession = Depends(get_db)):
    team = Team(**body.model_dump())
    db.add(team)
    await db.commit()
    await db.refresh(team)
    return TeamResponse(**team.__dict__, test_count=0)


@router.get("", response_model=list[TeamResponse])
async def list_teams(db: AsyncSession = Depends(get_db)):
    stmt = (
        select(Team, func.count(Test.id).label("test_count"))
        .outerjoin(Test, Test.team_id == Team.id)
        .group_by(Team.id)
        .order_by(Team.name)
    )
    rows = (await db.execute(stmt)).all()
    return [
        TeamResponse(**team.__dict__, test_count=count)
        for team, count in rows
    ]


@router.get("/{team_id}", response_model=TeamResponse)
async def get_team(team_id: int, db: AsyncSession = Depends(get_db)):
    team = await db.get(Team, team_id)
    if not team:
        raise HTTPException(404, "Team not found")
    count = (await db.execute(select(func.count(Test.id)).where(Test.team_id == team_id))).scalar() or 0
    return TeamResponse(**team.__dict__, test_count=count)


@router.put("/{team_id}", response_model=TeamResponse)
async def update_team(team_id: int, body: TeamUpdate, db: AsyncSession = Depends(get_db)):
    team = await db.get(Team, team_id)
    if not team:
        raise HTTPException(404, "Team not found")
    for key, val in body.model_dump(exclude_unset=True).items():
        setattr(team, key, val)
    await db.commit()
    await db.refresh(team)
    count = (await db.execute(select(func.count(Test.id)).where(Test.team_id == team_id))).scalar() or 0
    return TeamResponse(**team.__dict__, test_count=count)


@router.delete("/{team_id}", status_code=204)
async def delete_team(team_id: int, db: AsyncSession = Depends(get_db)):
    team = await db.get(Team, team_id)
    if not team:
        raise HTTPException(404, "Team not found")
    count = (await db.execute(select(func.count(Test.id)).where(Test.team_id == team_id))).scalar() or 0
    if count > 0:
        raise HTTPException(400, f"Cannot delete team with {count} existing tests. Reassign tests first.")
    await db.delete(team)
    await db.commit()
