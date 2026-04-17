"""
Async MySQL database engine and session factory via SQLAlchemy + aiomysql.
"""

from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = (
    "mysql+aiomysql://"
    f"{os.getenv('AQA_DB_USER', 'root')}:"
    f"{os.getenv('AQA_DB_PASSWORD', '')}@"
    f"{os.getenv('AQA_DB_HOST', 'localhost')}:"
    f"{os.getenv('AQA_DB_PORT', '3306')}/"
    f"{os.getenv('AQA_DB_NAME', 'automateqa')}"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("AQA_DB_ECHO", "false").lower() == "true",
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass
