"""
DriftDetector — tracks selector confidence across runs.
FlakinessTracker — tracks per-step pass/fail history for flakiness scoring.

Detects gradual degradation of selectors before they break.
Stores history in JSON files alongside the test files.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DriftDetector:
    """Tracks selector confidence trends and detects drift."""

    def __init__(self, history_path: str = ".automateqa_drift.json") -> None:
        self._path = Path(history_path)
        self._history: dict[str, list[dict[str, Any]]] = self._load()

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._history, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Failed to save drift history: %s", e)

    def record_confidence(
        self, test_id: str, step_id: int, strategy: str, confidence: float,
    ) -> None:
        key = f"{test_id}:{step_id}"
        self._history.setdefault(key, []).append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "confidence": round(confidence, 4),
        })
        self._history[key] = self._history[key][-50:]
        self._save()

    def get_drift_report(self, test_id: str) -> list[dict[str, Any]]:
        """Identify steps with declining confidence over recent runs."""
        alerts: list[dict[str, Any]] = []
        for key, runs in self._history.items():
            if not key.startswith(f"{test_id}:"):
                continue
            if len(runs) < 5:
                continue
            recent = [r["confidence"] for r in runs[-10:]]
            older = [r["confidence"] for r in runs[-20:-10]]
            if not older or not recent:
                continue
            avg_recent = sum(recent) / len(recent)
            avg_older = sum(older) / len(older)
            drop = avg_older - avg_recent
            if drop > 0.15:
                step_id = key.split(":")[1]
                severity = "critical" if avg_recent < 0.5 else "warning"
                alerts.append({
                    "step_id": step_id,
                    "confidence_trend": f"{avg_older:.2f} -> {avg_recent:.2f}",
                    "drop": round(drop, 2),
                    "severity": severity,
                    "latest_strategy": runs[-1].get("strategy", ""),
                    "recommendation": (
                        "Re-record this step or add data-testid"
                        if severity == "critical"
                        else "Monitor — consider adding stable selectors"
                    ),
                })
        return sorted(alerts, key=lambda a: a["drop"], reverse=True)

    def get_step_history(self, test_id: str, step_id: int) -> list[dict[str, Any]]:
        return self._history.get(f"{test_id}:{step_id}", [])

    def clear(self, test_id: str = "") -> None:
        if test_id:
            for k in [k for k in self._history if k.startswith(f"{test_id}:")]:
                del self._history[k]
        else:
            self._history.clear()
        self._save()


class FlakinessTracker:
    """Tracks per-step pass/fail history for flakiness scoring."""

    def __init__(self, history_path: str = ".automateqa_flakiness.json") -> None:
        self._path = Path(history_path)
        self._history: dict[str, list[bool]] = self._load()

    def _load(self) -> dict[str, list[bool]]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save(self) -> None:
        try:
            self._path.write_text(json.dumps(self._history), encoding="utf-8")
        except OSError:
            pass

    def record(self, step_key: str, passed: bool) -> None:
        self._history.setdefault(step_key, []).append(passed)
        self._history[step_key] = self._history[step_key][-20:]
        self._save()

    def flakiness_score(self, step_key: str) -> float:
        """0.0 = stable; 1.0 = maximally flaky (alternates every run)."""
        runs = self._history.get(step_key, [])
        if len(runs) < 3:
            return 0.0
        flips = sum(1 for i in range(1, len(runs)) if runs[i] != runs[i - 1])
        return round(flips / (len(runs) - 1), 2)
