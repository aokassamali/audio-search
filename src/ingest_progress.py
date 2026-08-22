from __future__ import annotations

from copy import deepcopy
from threading import Lock


_lock = Lock()
_progress: dict[str, dict] = {}


def initialize(source_key: str) -> None:
    with _lock:
        _progress[source_key] = {
            "status": "queued",
            "active_stage": None,
            "stages": {
                "normalize": {"status": "pending", "progress": None},
                "transcribe": {"status": "pending", "progress": None},
                "speakers": {"status": "pending", "progress": None},
                "chunk": {"status": "pending", "progress": None},
                "embed": {"status": "pending", "progress": None},
            },
            "error": None,
        }


def mark_running(source_key: str) -> None:
    with _lock:
        if source_key in _progress:
            _progress[source_key]["status"] = "running"


def start_stage(source_key: str, stage: str) -> None:
    with _lock:
        record = _progress.get(source_key)
        if record is None:
            return
        record["status"] = "running"
        record["active_stage"] = stage
        record["stages"][stage]["status"] = "running"
        record["stages"][stage]["progress"] = None


def update_stage(source_key: str, stage: str, progress: float) -> None:
    with _lock:
        record = _progress.get(source_key)
        if record is None:
            return
        record["active_stage"] = stage
        record["stages"][stage]["status"] = "running"
        record["stages"][stage]["progress"] = max(0.0, min(100.0, float(progress)))


def complete_stage(source_key: str, stage: str) -> None:
    with _lock:
        record = _progress.get(source_key)
        if record is None:
            return
        record["stages"][stage]["status"] = "complete"
        record["stages"][stage]["progress"] = 100.0
        if record["active_stage"] == stage:
            record["active_stage"] = None


def mark_complete(source_key: str) -> None:
    with _lock:
        record = _progress.get(source_key)
        if record is None:
            return
        record["status"] = "complete"
        record["active_stage"] = None
        for stage in record["stages"].values():
            stage["status"] = "complete"
            stage["progress"] = 100.0


def mark_failed(source_key: str, error: str) -> None:
    with _lock:
        record = _progress.get(source_key)
        if record is None:
            return
        record["status"] = "failed"
        record["error"] = error


def snapshot(source_key: str) -> dict | None:
    with _lock:
        record = _progress.get(source_key)
        return deepcopy(record) if record is not None else None
