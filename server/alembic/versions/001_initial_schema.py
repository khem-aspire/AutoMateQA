"""Initial schema — teams, categories, tests, test_versions, schedules, test_runs, step_results

Revision ID: 001
Revises: None
Create Date: 2026-04-02
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Teams
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(100), unique=True, nullable=False),
        sa.Column("slug", sa.String(100), unique=True, nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("color", sa.String(7), server_default="#6366f1"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Categories
    op.create_table(
        "categories",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(100), unique=True, nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("color", sa.String(7), server_default="#6366f1"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Tests
    op.create_table(
        "tests",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("test_id", sa.String(16), unique=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("base_url", sa.String(2048), server_default=""),
        sa.Column("owner_email", sa.String(255), nullable=False),
        sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("category_id", sa.Integer, sa.ForeignKey("categories.id"), nullable=True),
        sa.Column("s3_key", sa.String(1024), nullable=False),
        sa.Column("current_version", sa.Integer, server_default="1"),
        sa.Column("step_count", sa.Integer, server_default="0"),
        sa.Column("tags", sa.Text, nullable=True),
        sa.Column("is_active", sa.Boolean, server_default=sa.text("1")),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("idx_tests_team", "tests", ["team_id"])
    op.create_index("idx_tests_category", "tests", ["category_id"])
    op.create_index("idx_tests_owner", "tests", ["owner_email"])

    # Test Versions
    op.create_table(
        "test_versions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("test_id", sa.Integer, sa.ForeignKey("tests.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("s3_key", sa.String(1024), nullable=False),
        sa.Column("change_reason", sa.String(50), server_default="upload"),
        sa.Column("healed_steps", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_versions_test_version", "test_versions", ["test_id", "version"])

    # Schedules
    op.create_table(
        "schedules",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("test_id", sa.Integer, sa.ForeignKey("tests.id", ondelete="CASCADE"), unique=True, nullable=False),
        sa.Column("cron_expr", sa.String(100), server_default="0 6 * * *"),
        sa.Column("timezone", sa.String(50), server_default="UTC"),
        sa.Column("enabled", sa.Boolean, server_default=sa.text("1")),
        sa.Column("last_run_at", sa.DateTime, nullable=True),
        sa.Column("next_run_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Test Runs
    op.create_table(
        "test_runs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(32), unique=True, nullable=False),
        sa.Column("test_id", sa.Integer, sa.ForeignKey("tests.id", ondelete="CASCADE"), nullable=False),
        sa.Column("test_version", sa.Integer, server_default="1"),
        sa.Column("trigger_type", sa.String(20), server_default="manual"),
        sa.Column("status", sa.String(10), server_default="pending"),
        sa.Column("started_at", sa.DateTime, nullable=True),
        sa.Column("finished_at", sa.DateTime, nullable=True),
        sa.Column("total_duration_ms", sa.Double, server_default="0"),
        sa.Column("total_steps", sa.Integer, server_default="0"),
        sa.Column("passed_count", sa.Integer, server_default="0"),
        sa.Column("failed_count", sa.Integer, server_default="0"),
        sa.Column("healed_count", sa.Integer, server_default="0"),
        sa.Column("tokens_used", sa.Integer, server_default="0"),
        sa.Column("result_json", mysql.LONGTEXT, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_runs_test", "test_runs", ["test_id"])
    op.create_index("idx_runs_status", "test_runs", ["status"])
    op.create_index("idx_runs_created", "test_runs", ["created_at"])

    # Step Results
    op.create_table(
        "step_results",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("test_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("step_id", sa.Integer, nullable=False),
        sa.Column("action_type", sa.String(30), server_default=""),
        sa.Column("status", sa.String(10), server_default=""),
        sa.Column("confidence", sa.Double, server_default="0"),
        sa.Column("duration_ms", sa.Double, server_default="0"),
        sa.Column("retry_count", sa.Integer, server_default="0"),
        sa.Column("healed", sa.Boolean, server_default=sa.text("0")),
        sa.Column("healing_details", sa.Text, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
    )
    op.create_index("idx_step_results_run", "step_results", ["run_id"])

    # Seed default categories
    op.execute(
        "INSERT INTO categories (name, description, color) VALUES "
        "('smoke', 'Quick sanity checks', '#22c55e'), "
        "('regression', 'Full regression suite', '#3b82f6'), "
        "('e2e', 'End-to-end user flows', '#8b5cf6'), "
        "('integration', 'Integration tests', '#f59e0b'), "
        "('performance', 'Performance and load tests', '#ef4444')"
    )

    # Seed default teams
    op.execute(
        "INSERT INTO teams (name, slug, description, color) VALUES "
        "('AspireOS', 'aspireos', 'AspireOS team', '#6366f1'), "
        "('PFX', 'pfx', 'PFX team', '#ec4899'), "
        "('Spend', 'spend', 'Spend team', '#14b8a6')"
    )


def downgrade() -> None:
    op.drop_table("step_results")
    op.drop_table("test_runs")
    op.drop_table("schedules")
    op.drop_table("test_versions")
    op.drop_table("tests")
    op.drop_table("categories")
    op.drop_table("teams")
