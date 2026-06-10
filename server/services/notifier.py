"""
Email notification service for test failure alerts.
"""

from __future__ import annotations

import logging
import os
from email.message import EmailMessage

logger = logging.getLogger(__name__)


class NotifierService:
    """Sends email notifications on test failures."""

    def __init__(self):
        self._host = os.getenv("AQA_SMTP_HOST", "")
        self._port = int(os.getenv("AQA_SMTP_PORT", "587"))
        self._user = os.getenv("AQA_SMTP_USER", "")
        self._password = os.getenv("AQA_SMTP_PASS", "")
        self._from = os.getenv("AQA_SMTP_FROM", "noreply@automateqa.dev")

    @property
    def is_configured(self) -> bool:
        return bool(self._host and self._user)

    async def send_failure_email(
        self,
        to_email: str,
        test_name: str,
        run_id: str,
        failed_count: int,
        total_steps: int,
    ):
        """Send a failure notification email."""
        if not self.is_configured:
            logger.debug("SMTP not configured, skipping email notification")
            return

        dashboard_url = os.getenv("AQA_DASHBOARD_URL", "http://localhost:5173")
        run_url = f"{dashboard_url}/runs/{run_id}"

        msg = EmailMessage()
        msg["Subject"] = f"[AutoMateQA] Test Failed: {test_name}"
        msg["From"] = self._from
        msg["To"] = to_email
        msg.set_content(
            f"Test '{test_name}' has failed.\n\n"
            f"Failed steps: {failed_count}/{total_steps}\n"
            f"Run ID: {run_id}\n\n"
            f"View details: {run_url}\n\n"
            f"— AutoMateQA Dashboard"
        )
        msg.add_alternative(
            f"""<html><body style="font-family: -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem;">
            <div style="max-width: 600px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 2rem;">
                <h2 style="color: #ef4444;">Test Failed: {test_name}</h2>
                <p style="color: #94a3b8;">
                    <strong>{failed_count}</strong> of <strong>{total_steps}</strong> steps failed.
                </p>
                <p>Run ID: <code>{run_id}</code></p>
                <a href="{run_url}" style="display: inline-block; background: #3b82f6; color: white; padding: 10px 20px;
                   border-radius: 8px; text-decoration: none; margin-top: 1rem;">View Run Details</a>
            </div>
            </body></html>""",
            subtype="html",
        )

        try:
            import aiosmtplib
            await aiosmtplib.send(
                msg,
                hostname=self._host,
                port=self._port,
                username=self._user,
                password=self._password,
                start_tls=True,
            )
            logger.info("Failure notification sent to %s for run %s", to_email, run_id)
        except Exception as e:
            logger.error("Failed to send email to %s: %s", to_email, e)


# Global singleton
notifier_service = NotifierService()
