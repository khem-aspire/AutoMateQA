"""
ExecutionManager — bridges the dashboard API with the AutoMateQA engine.

Downloads test files from S3, runs them via TestEngine, stores results,
and broadcasts real-time progress over WebSocket queues.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone

from server.services.s3_manager import S3Manager

logger = logging.getLogger(__name__)

# Max concurrent test executions
_MAX_CONCURRENT = int(os.getenv("AQA_MAX_CONCURRENT_RUNS", "2"))
_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)


class ExecutionManager:
    """Manages test execution lifecycle and WebSocket broadcasting."""

    def __init__(self):
        self._active_runs: dict[str, asyncio.Task] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._s3 = S3Manager()

    async def trigger_run(
        self,
        test_db_id: int,
        test_id_str: str,
        s3_key: str,
        current_version: int,
        owner_email: str,
        trigger_type: str = "manual",
        config_overrides: dict | None = None,
    ) -> str:
        """Create a run and start execution in background. Returns run_id."""
        run_id = uuid.uuid4().hex

        task = asyncio.create_task(
            self._execute_run(
                run_id=run_id,
                test_db_id=test_db_id,
                test_id_str=test_id_str,
                s3_key=s3_key,
                current_version=current_version,
                owner_email=owner_email,
                trigger_type=trigger_type,
                config_overrides=config_overrides or {},
            )
        )
        self._active_runs[run_id] = task
        return run_id

    async def _execute_run(
        self,
        run_id: str,
        test_db_id: int,
        test_id_str: str,
        s3_key: str,
        current_version: int,
        owner_email: str,
        trigger_type: str,
        config_overrides: dict,
    ):
        """Background task: download from S3, execute, store results."""
        # Import here to avoid circular imports and heavy module loading at import time
        from engine.core import TestEngine
        from engine.models import StepStatus

        from server.database import async_session
        from server.models import StepResult as StepResultModel
        from server.models import TestRun, TestVersion, Test

        tmp_path = None
        async with _semaphore:
            try:
                # Download test file from S3
                tmp_path = self._s3.download_to_tempfile(s3_key)

                # Create DB run record
                async with async_session() as db:
                    run = TestRun(
                        run_id=run_id,
                        test_id=test_db_id,
                        test_version=current_version,
                        trigger_type=trigger_type,
                        status="running",
                        started_at=datetime.now(timezone.utc),
                    )
                    db.add(run)
                    await db.commit()
                    await db.refresh(run)
                    run_db_id = run.id

                # Broadcast run started
                await self._broadcast(run_id, {
                    "type": "run_started",
                    "run_id": run_id,
                })

                # Build engine with config overrides
                engine_kwargs = {
                    "headless": True,  # always headless in server mode
                    "healing_mode": config_overrides.get("healing_mode", "disabled"),
                    "llm_enabled": config_overrides.get("llm_enabled", False),
                    "llm_model": config_overrides.get("llm_model", "gpt-4o"),
                    "llm_provider": config_overrides.get("llm_provider", "openai"),
                    "confidence_threshold": config_overrides.get("confidence_threshold", 0.75),
                    "retry_strategy": config_overrides.get("retry_strategy", "exponential"),
                    "max_step_retries": config_overrides.get("max_step_retries", 3),
                }
                engine = TestEngine(**engine_kwargs)

                # Step callback for real-time updates
                async def on_step_complete(step_result, total_steps):
                    await self._broadcast(run_id, {
                        "type": "step_completed",
                        "step_id": step_result.step_id,
                        "status": step_result.status.value,
                        "confidence": step_result.element_confidence,
                        "duration_ms": step_result.duration_ms,
                        "healed": step_result.healed,
                        "healing_details": step_result.healing_details,
                        "error": step_result.error,
                        "action_type": step_result.action_type or "",
                        "retry_count": step_result.retry_count,
                        "total_steps": total_steps,
                    })

                # Execute
                screenshot_dir = tempfile.mkdtemp(prefix="aqa_screenshots_")
                result = await engine.execute(
                    test_path=tmp_path,
                    screenshot_dir=screenshot_dir,
                    on_step_complete=on_step_complete,
                )

                # Store results in DB
                async with async_session() as db:
                    run = await db.get(TestRun, run_db_id)
                    run.status = result.status.value
                    run.finished_at = datetime.now(timezone.utc)
                    run.total_duration_ms = result.total_duration_ms
                    run.total_steps = len(result.steps)
                    run.passed_count = sum(1 for s in result.steps if s.status == StepStatus.PASSED)
                    run.failed_count = result.failed_count
                    run.healed_count = result.healed_count
                    run.tokens_used = result.tokens_used
                    run.result_json = result.model_dump_json()

                    # Insert step results
                    for sr in result.steps:
                        step = StepResultModel(
                            run_id=run_db_id,
                            step_id=sr.step_id,
                            action_type=sr.action_type or "",
                            status=sr.status.value,
                            confidence=sr.element_confidence,
                            duration_ms=sr.duration_ms,
                            retry_count=sr.retry_count,
                            healed=sr.healed,
                            healing_details=sr.healing_details,
                            error=sr.error,
                        )
                        db.add(step)

                    # If self-healing occurred, upload new version to S3
                    if (
                        engine_kwargs.get("healing_mode") == "auto_update"
                        and result.healed_count > 0
                        and os.path.exists(tmp_path)
                    ):
                        with open(tmp_path, "rb") as f:
                            healed_bytes = f.read()
                        filename = s3_key.split("/")[-1]
                        new_version = current_version + 1
                        new_s3_key = self._s3.upload_file(healed_bytes, test_id_str, new_version, filename)

                        # Update test record
                        test = await db.get(Test, test_db_id)
                        test.current_version = new_version
                        test.s3_key = new_s3_key

                        # Create version record
                        healed_step_ids = [sr.step_id for sr in result.steps if sr.healed]
                        version = TestVersion(
                            test_id=test_db_id,
                            version=new_version,
                            s3_key=new_s3_key,
                            change_reason="self_healed",
                            healed_steps=json.dumps(healed_step_ids),
                        )
                        db.add(version)

                    await db.commit()

                # Broadcast run completed
                await self._broadcast(run_id, {
                    "type": "run_completed",
                    "run_id": run_id,
                    "status": result.status.value,
                    "total_duration_ms": result.total_duration_ms,
                    "passed": run.passed_count,
                    "failed": result.failed_count,
                    "healed": result.healed_count,
                })

                # Send failure notification
                if result.status == StepStatus.FAILED:
                    try:
                        from server.services.notifier import notifier_service
                        await notifier_service.send_failure_email(
                            to_email=owner_email,
                            test_name=result.test_name,
                            run_id=run_id,
                            failed_count=result.failed_count,
                            total_steps=len(result.steps),
                        )
                    except Exception as e:
                        logger.warning("Failed to send notification: %s", e)

            except Exception as e:
                logger.error("Run %s failed with error: %s", run_id, e)
                async with async_session() as db:
                    run = await db.get(TestRun, run_db_id)
                    if run:
                        run.status = "error"
                        run.finished_at = datetime.now(timezone.utc)
                        run.error_message = str(e)
                        await db.commit()

                await self._broadcast(run_id, {
                    "type": "run_error",
                    "run_id": run_id,
                    "error": str(e),
                })
            finally:
                self._active_runs.pop(run_id, None)
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    async def _broadcast(self, run_id: str, message: dict):
        """Send a message to all WebSocket subscribers for a given run."""
        queues = self._subscribers.get(run_id, [])
        for q in queues:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                pass

    def subscribe(self, run_id: str) -> asyncio.Queue:
        """Subscribe to real-time updates for a run."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.setdefault(run_id, []).append(q)
        return q

    def unsubscribe(self, run_id: str, queue: asyncio.Queue):
        """Unsubscribe from updates."""
        queues = self._subscribers.get(run_id, [])
        if queue in queues:
            queues.remove(queue)
        if not queues:
            self._subscribers.pop(run_id, None)

    def is_running(self, run_id: str) -> bool:
        return run_id in self._active_runs


# Global singleton
execution_manager = ExecutionManager()
