"""
Tests router — upload, list, CRUD, download, inspect, versions.

All detail endpoints use the test UUID (test_id string), not the DB integer id.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from server.dependencies import get_db
from server.models import Category, Team, Test, TestRun, TestVersion
from server.schemas import (
    PaginatedResponse,
    TestResponse,
    TestUpdate,
    TestVersionResponse,
)
from server.services.s3_manager import S3Manager, parse_test_file

logger = logging.getLogger(__name__)
router = APIRouter()

s3 = S3Manager()


async def _get_test_by_uuid(db: AsyncSession, test_uuid: str) -> Test:
    """Resolve a test UUID string to a Test ORM object or raise 404."""
    stmt = (
        select(Test)
        .options(selectinload(Test.team), selectinload(Test.category))
        .where(Test.test_id == test_uuid)
    )
    test = (await db.execute(stmt)).scalar_one_or_none()
    if not test:
        raise HTTPException(404, f"Test with uuid '{test_uuid}' not found")
    return test


def _build_test_response(test: Test, last_run_status: str | None = None) -> TestResponse:
    return TestResponse(
        id=test.id,
        test_id=test.test_id,
        name=test.name,
        description=test.description,
        base_url=test.base_url,
        owner_email=test.owner_email,
        team_id=test.team_id,
        team_name=test.team.name if test.team else "",
        category_id=test.category_id,
        category_name=test.category.name if test.category else "",
        s3_key=test.s3_key,
        current_version=test.current_version,
        step_count=test.step_count,
        tags=test.tags,
        is_active=test.is_active,
        created_at=test.created_at,
        updated_at=test.updated_at,
        last_run_status=last_run_status,
    )


@router.post("/upload", response_model=TestResponse, status_code=201)
async def upload_test(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    owner_email: str = Form(...),
    team_id: int = Form(...),
    category_id: Optional[int] = Form(None),
    tags: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """Upload a .aqa or .json test file with metadata."""
    if not file.filename or not (file.filename.endswith(".aqa") or file.filename.endswith(".json")):
        raise HTTPException(400, "File must be .aqa or .json")

    team = await db.get(Team, team_id)
    if not team:
        raise HTTPException(400, f"Team with id {team_id} not found")

    if category_id:
        cat = await db.get(Category, category_id)
        if not cat:
            raise HTTPException(400, f"Category with id {category_id} not found")

    file_bytes = await file.read()
    try:
        test_data = parse_test_file(file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse test file: {e}")

    test_id = test_data.get("test_id", "")
    if not test_id:
        raise HTTPException(400, "Test file missing test_id field")

    existing = (await db.execute(select(Test).where(Test.test_id == test_id))).scalar_one_or_none()
    if existing:
        raise HTTPException(409, f"Test with test_id '{test_id}' already exists")

    s3_key = s3.upload_file(file_bytes, test_id, version=1, filename=file.filename)

    test = Test(
        test_id=test_id,
        name=name or test_data.get("name", "Unnamed Test"),
        description=description or test_data.get("description", ""),
        base_url=test_data.get("base_url", ""),
        owner_email=owner_email,
        team_id=team_id,
        category_id=category_id,
        s3_key=s3_key,
        current_version=1,
        step_count=len(test_data.get("steps", [])),
        tags=tags,
    )
    db.add(test)
    await db.flush()

    version = TestVersion(
        test_id=test.id,
        version=1,
        s3_key=s3_key,
        change_reason="upload",
    )
    db.add(version)
    await db.commit()
    await db.refresh(test, ["team", "category"])

    return _build_test_response(test)


@router.get("", response_model=PaginatedResponse)
async def list_tests(
    team_id: Optional[int] = Query(None),
    category_id: Optional[int] = Query(None),
    owner_email: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Test).options(selectinload(Test.team), selectinload(Test.category))
    count_stmt = select(func.count(Test.id))

    if team_id is not None:
        stmt = stmt.where(Test.team_id == team_id)
        count_stmt = count_stmt.where(Test.team_id == team_id)
    if category_id is not None:
        stmt = stmt.where(Test.category_id == category_id)
        count_stmt = count_stmt.where(Test.category_id == category_id)
    if owner_email:
        stmt = stmt.where(Test.owner_email == owner_email)
        count_stmt = count_stmt.where(Test.owner_email == owner_email)
    if is_active is not None:
        stmt = stmt.where(Test.is_active == is_active)
        count_stmt = count_stmt.where(Test.is_active == is_active)
    if search:
        pattern = f"%{search}%"
        search_filter = Test.name.ilike(pattern) | Test.description.ilike(pattern) | Test.test_id.ilike(pattern)
        stmt = stmt.where(search_filter)
        count_stmt = count_stmt.where(search_filter)

    total = (await db.execute(count_stmt)).scalar() or 0
    stmt = stmt.order_by(Test.updated_at.desc()).offset((page - 1) * per_page).limit(per_page)
    tests = (await db.execute(stmt)).scalars().all()

    items = []
    for t in tests:
        last_run = (
            await db.execute(
                select(TestRun.status)
                .where(TestRun.test_id == t.id)
                .order_by(TestRun.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        items.append(_build_test_response(t, last_run_status=last_run))

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        per_page=per_page,
        pages=(total + per_page - 1) // per_page if total > 0 else 0,
    )


@router.get("/{test_uuid}", response_model=TestResponse)
async def get_test(test_uuid: str, db: AsyncSession = Depends(get_db)):
    test = await _get_test_by_uuid(db, test_uuid)
    last_run = (
        await db.execute(
            select(TestRun.status).where(TestRun.test_id == test.id).order_by(TestRun.created_at.desc()).limit(1)
        )
    ).scalar_one_or_none()
    return _build_test_response(test, last_run_status=last_run)


@router.put("/{test_uuid}", response_model=TestResponse)
async def update_test(test_uuid: str, body: TestUpdate, db: AsyncSession = Depends(get_db)):
    test = await _get_test_by_uuid(db, test_uuid)
    for key, val in body.model_dump(exclude_unset=True).items():
        setattr(test, key, val)
    await db.commit()
    await db.refresh(test, ["team", "category"])
    return _build_test_response(test)


@router.delete("/{test_uuid}", status_code=204)
async def delete_test(test_uuid: str, db: AsyncSession = Depends(get_db)):
    test = await _get_test_by_uuid(db, test_uuid)
    s3.delete_prefix(f"tests/{test.test_id}/")
    await db.delete(test)
    await db.commit()


@router.get("/{test_uuid}/download")
async def download_test(test_uuid: str, version: Optional[int] = Query(None), db: AsyncSession = Depends(get_db)):
    test = await _get_test_by_uuid(db, test_uuid)

    s3_key = test.s3_key
    if version is not None:
        ver = (
            await db.execute(
                select(TestVersion).where(TestVersion.test_id == test.id, TestVersion.version == version)
            )
        ).scalar_one_or_none()
        if not ver:
            raise HTTPException(404, f"Version {version} not found")
        s3_key = ver.s3_key

    file_bytes = s3.download_bytes(s3_key)
    filename = s3_key.split("/")[-1]
    content_type = "application/gzip" if filename.endswith(".aqa") else "application/json"
    return Response(content=file_bytes, media_type=content_type, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/{test_uuid}/inspect")
async def inspect_test(test_uuid: str, db: AsyncSession = Depends(get_db)):
    """Parse and return the full TestModel JSON."""
    test = await _get_test_by_uuid(db, test_uuid)
    file_bytes = s3.download_bytes(test.s3_key)
    filename = test.s3_key.split("/")[-1]
    return parse_test_file(file_bytes, filename)


@router.get("/{test_uuid}/versions", response_model=list[TestVersionResponse])
async def list_versions(test_uuid: str, db: AsyncSession = Depends(get_db)):
    test = await _get_test_by_uuid(db, test_uuid)
    stmt = select(TestVersion).where(TestVersion.test_id == test.id).order_by(TestVersion.version.desc())
    versions = (await db.execute(stmt)).scalars().all()
    return [TestVersionResponse.model_validate(v) for v in versions]


@router.post("/{test_uuid}/versions", response_model=TestResponse, status_code=201)
async def upload_new_version(
    test_uuid: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a new version of an existing test's .aqa/.json file."""
    test = await _get_test_by_uuid(db, test_uuid)

    if not file.filename or not (file.filename.endswith(".aqa") or file.filename.endswith(".json")):
        raise HTTPException(400, "File must be .aqa or .json")

    file_bytes = await file.read()
    try:
        test_data = parse_test_file(file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse test file: {e}")

    # Validate the uploaded file belongs to the same test
    file_test_id = test_data.get("test_id", "")
    if file_test_id and file_test_id != test.test_id:
        raise HTTPException(
            400,
            f"File test_id '{file_test_id}' does not match existing test '{test.test_id}'. "
            "Upload the correct file or use /upload for a new test.",
        )

    new_version = test.current_version + 1
    new_s3_key = s3.upload_file(file_bytes, test.test_id, new_version, file.filename)

    # Update test record
    test.s3_key = new_s3_key
    test.current_version = new_version
    test.step_count = len(test_data.get("steps", []))

    # Create version record
    version = TestVersion(
        test_id=test.id,
        version=new_version,
        s3_key=new_s3_key,
        change_reason="manual_edit",
    )
    db.add(version)
    await db.commit()
    await db.refresh(test, ["team", "category"])

    return _build_test_response(test)
