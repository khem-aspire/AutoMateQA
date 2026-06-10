"""
Categories CRUD router.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from server.dependencies import get_db
from server.models import Category, Test
from server.schemas import CategoryCreate, CategoryResponse, CategoryUpdate

router = APIRouter()


@router.post("", response_model=CategoryResponse, status_code=201)
async def create_category(body: CategoryCreate, db: AsyncSession = Depends(get_db)):
    cat = Category(**body.model_dump())
    db.add(cat)
    await db.commit()
    await db.refresh(cat)
    return CategoryResponse(**cat.__dict__, test_count=0)


@router.get("", response_model=list[CategoryResponse])
async def list_categories(db: AsyncSession = Depends(get_db)):
    stmt = (
        select(Category, func.count(Test.id).label("test_count"))
        .outerjoin(Test, Test.category_id == Category.id)
        .group_by(Category.id)
        .order_by(Category.name)
    )
    rows = (await db.execute(stmt)).all()
    return [
        CategoryResponse(**cat.__dict__, test_count=count)
        for cat, count in rows
    ]


@router.put("/{category_id}", response_model=CategoryResponse)
async def update_category(category_id: int, body: CategoryUpdate, db: AsyncSession = Depends(get_db)):
    cat = await db.get(Category, category_id)
    if not cat:
        raise HTTPException(404, "Category not found")
    for key, val in body.model_dump(exclude_unset=True).items():
        setattr(cat, key, val)
    await db.commit()
    await db.refresh(cat)
    count = (await db.execute(select(func.count(Test.id)).where(Test.category_id == category_id))).scalar() or 0
    return CategoryResponse(**cat.__dict__, test_count=count)


@router.delete("/{category_id}", status_code=204)
async def delete_category(category_id: int, db: AsyncSession = Depends(get_db)):
    cat = await db.get(Category, category_id)
    if not cat:
        raise HTTPException(404, "Category not found")
    # Set tests to uncategorized
    stmt = select(Test).where(Test.category_id == category_id)
    tests = (await db.execute(stmt)).scalars().all()
    for t in tests:
        t.category_id = None
    await db.delete(cat)
    await db.commit()
