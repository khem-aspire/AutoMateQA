"""
APScheduler integration for scheduled test execution.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger("automateqa.scheduler")


class SchedulerService:
    """Manages cron-scheduled test executions via APScheduler."""

    def __init__(self):
        self._scheduler = AsyncIOScheduler()

    async def start(self):
        """Start scheduler and load all enabled schedules from DB."""
        self._scheduler.start()
        print("[Scheduler] APScheduler started")

        from server.database import async_session
        from server.models import Schedule
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        try:
            async with async_session() as db:
                stmt = (
                    select(Schedule)
                    .options(selectinload(Schedule.test))
                    .where(Schedule.enabled == True)
                )
                schedules = (await db.execute(stmt)).scalars().all()
                for sched in schedules:
                    if sched.test:
                        self._add_job(sched)
                print(f"[Scheduler] Loaded {len(schedules)} scheduled jobs")
        except Exception as e:
            print(f"[Scheduler] WARNING: Failed to load schedules on startup: {e}")

    async def stop(self):
        """Gracefully shut down the scheduler."""
        self._scheduler.shutdown(wait=False)
        print("[Scheduler] APScheduler stopped")

    def _add_job(self, schedule):
        """Add a cron job for a schedule record."""
        job_id = f"schedule_{schedule.id}"
        try:
            trigger = CronTrigger.from_crontab(schedule.cron_expr, timezone=schedule.timezone)
        except Exception as e:
            print(f"[Scheduler] ERROR: Invalid cron '{schedule.cron_expr}' for schedule {schedule.id}: {e}")
            return

        self._scheduler.add_job(
            self._run_scheduled_test,
            trigger=trigger,
            id=job_id,
            replace_existing=True,
            kwargs={
                "schedule_id": schedule.id,
                "test_db_id": schedule.test.id,
                "test_id_str": schedule.test.test_id,
                "s3_key": schedule.test.s3_key,
                "current_version": schedule.test.current_version,
                "owner_email": schedule.test.owner_email,
            },
        )
        next_fire = trigger.get_next_fire_time(None, datetime.now(timezone.utc))
        print(f"[Scheduler] Added job {job_id}: cron='{schedule.cron_expr}' next_fire={next_fire}")

    def add_schedule(self, schedule):
        """Add or replace a cron job."""
        self._add_job(schedule)

    def remove_schedule(self, schedule_id: int):
        """Remove a cron job."""
        job_id = f"schedule_{schedule_id}"
        try:
            self._scheduler.remove_job(job_id)
            print(f"[Scheduler] Removed job {job_id}")
        except Exception:
            pass

    def get_jobs_info(self) -> list[dict]:
        """Return info about all registered jobs (for debugging)."""
        jobs = self._scheduler.get_jobs()
        return [
            {
                "job_id": job.id,
                "next_run_time": str(job.next_run_time) if job.next_run_time else None,
                "trigger": str(job.trigger),
            }
            for job in jobs
        ]

    @staticmethod
    async def _run_scheduled_test(
        schedule_id: int,
        test_db_id: int,
        test_id_str: str,
        s3_key: str,
        current_version: int,
        owner_email: str,
    ):
        """Callback executed by APScheduler for each scheduled test."""
        from server.database import async_session
        from server.models import Schedule
        from server.services.executor import execution_manager

        print(f"[Scheduler] Firing scheduled run for test {test_id_str} (schedule {schedule_id})")

        run_id = await execution_manager.trigger_run(
            test_db_id=test_db_id,
            test_id_str=test_id_str,
            s3_key=s3_key,
            current_version=current_version,
            owner_email=owner_email,
            trigger_type="scheduled",
        )

        async with async_session() as db:
            sched = await db.get(Schedule, schedule_id)
            if sched:
                sched.last_run_at = datetime.now(timezone.utc)
                await db.commit()

        print(f"[Scheduler] Scheduled run {run_id} triggered for test {test_id_str}")


# Global singleton
scheduler_service = SchedulerService()
