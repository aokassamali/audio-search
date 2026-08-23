from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import dagster as dg

from src.config import PROJECT_ROOT, Settings, SourceSettings

_STAGE_ORDER = ["normalize", "transcribe", "speakers", "chunk", "embed"]
_STAGE_WEIGHTS = {
    "normalize": 5.0,
    "transcribe": 45.0,
    "speakers": 35.0,
    "chunk": 5.0,
    "embed": 10.0,
}
_TERMINAL_STATUSES = {"complete", "failed", "cancelled"}


class AudioIngestManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs: dict[str, dict] = {}
        self._lock = Lock()
        self._futures: dict[str, Future] = {}
        self._processes: dict[str, subprocess.Popen] = {}

        self.max_workers = max(1, int(os.getenv("AUDIO_SEARCH_INGEST_WORKERS", "1")))
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="audio-ingest",
        )

        self.dagster_home = Path(
            os.getenv("DAGSTER_HOME", str(PROJECT_ROOT / "data" / "dagster_home"))
        )
        self.dagster_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("DAGSTER_HOME", str(self.dagster_home))
        self.instance = dg.DagsterInstance.get()

        self.log_dir = PROJECT_ROOT / "data" / "ingest_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
        return slug or "audio-source"

    def _unique_source_key(self, display_name: str) -> str:
        base = self._slugify(display_name)
        while True:
            key = f"{base}-{uuid.uuid4().hex[:8]}"
            with self._lock:
                known = set(self.settings.sources)
                known.update(job["source_key"] for job in self.jobs.values())
            if key not in known:
                return key

    def submit(self, audio_path: Path, display_name: str) -> dict:
        source_key = self._unique_source_key(display_name)
        source = SourceSettings(
            key=source_key,
            source_id=source_key,
            audio_filename=audio_path.name,
            chunk_variant=self.settings.chunk_variant,
            speaker_labels={},
            paths=self.settings.paths,
        )
        self.settings.sources[source_key] = source

        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "source_key": source_key,
            "source_id": source_key,
            "display_name": display_name,
            "filename": audio_path.name,
            "audio_path": str(audio_path),
            "status": "queued",
            "submitted_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "run_id": None,
            "log_path": str(self.log_dir / f"{job_id}.log"),
        }
        with self._lock:
            self.jobs[job_id] = job

        future = self.executor.submit(self._run, job_id)
        with self._lock:
            self._futures[job_id] = future
        return self.status(job_id)

    def _write_runtime_config(self, job: dict) -> Path:
        text = (PROJECT_ROOT / "audio_search.toml").read_text(encoding="utf-8").rstrip()
        source_key = job["source_key"]
        filename = job["filename"].replace('"', '\\"')
        source_id = job["source_id"].replace('"', '\\"')
        text += (
            "\n\n"
            f"[sources.{source_key}]\n"
            f"audio_filename = \"{filename}\"\n"
            f"source_id = \"{source_id}\"\n"
        )
        path = PROJECT_ROOT / f".audio_search_runtime_{job['job_id']}.toml"
        path.write_text(text, encoding="utf-8")
        return path

    @staticmethod
    def _dagster_executable() -> str:
        discovered = shutil.which("dagster")
        if discovered:
            return discovered
        candidate = Path(sys.executable).parent / ("dagster.exe" if os.name == "nt" else "dagster")
        if candidate.exists():
            return str(candidate)
        raise RuntimeError("Could not find the Dagster CLI in the active environment.")

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=8,
                )
                return
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.wait(timeout=4)
                    return
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    return
            except (OSError, ProcessLookupError):
                pass
        try:
            process.terminate()
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass

    def _set_cancelled(self, job_id: str) -> None:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None or job["status"] in {"complete", "failed"}:
                return
            job["status"] = "cancelled"
            job["finished_at"] = job.get("finished_at") or time.time()
            job["error"] = None

    def cancel(self, job_id: str) -> dict:
        # status() acquires the same non-reentrant lock, so never call it while
        # holding _lock. This matters when a cancel request races with completion.
        with self._lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = self.jobs[job_id]
            terminal = job["status"] in _TERMINAL_STATUSES
            if not terminal:
                job["status"] = "cancelling"
            future = self._futures.get(job_id)
            process = self._processes.get(job_id)

        if terminal:
            return self.status(job_id)

        if future is not None and future.cancel():
            self._set_cancelled(job_id)
            return self.status(job_id)

        if process is not None:
            self._terminate_process_tree(process)

        return self.status(job_id)

    def cancel_all(self) -> list[dict]:
        with self._lock:
            ids = [
                job_id
                for job_id, job in self.jobs.items()
                if job["status"] not in _TERMINAL_STATUSES
            ]
        return [self.cancel(job_id) for job_id in ids]

    def shutdown(self) -> None:
        """Stop live Dagster subprocesses when FastAPI shuts down."""
        try:
            self.cancel_all()
        finally:
            self.executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self.jobs[job_id]
            if job["status"] in {"cancelling", "cancelled"}:
                job["status"] = "cancelled"
                job["finished_at"] = time.time()
                return
            job["status"] = "running"
            job["started_at"] = time.time()

        source_key = job["source_key"]
        runtime_config = None

        try:
            with self._lock:
                if self.jobs[job_id]["status"] in {"cancelling", "cancelled"}:
                    self.jobs[job_id]["status"] = "cancelled"
                    self.jobs[job_id]["finished_at"] = time.time()
                    return

            self.instance.add_dynamic_partitions("audio_files", [source_key])
            runtime_config = self._write_runtime_config(job)
            env = os.environ.copy()
            env["DAGSTER_HOME"] = str(self.dagster_home)
            env["AUDIO_SEARCH_CONFIG"] = str(runtime_config)

            command = [
                self._dagster_executable(),
                "asset",
                "materialize",
                "-f",
                str(PROJECT_ROOT / "src" / "dagster_assets.py"),
                "--select",
                "*embeddings",
                "--partition",
                source_key,
            ]
            popen_kwargs = (
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                if os.name == "nt"
                else {"start_new_session": True}
            )

            log_path = Path(job["log_path"])
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    **popen_kwargs,
                )
                with self._lock:
                    self._processes[job_id] = process
                    cancel_requested = self.jobs[job_id]["status"] in {"cancelling", "cancelled"}
                if cancel_requested:
                    self._terminate_process_tree(process)
                returncode = process.wait()

            with self._lock:
                cancelled = self.jobs[job_id]["status"] in {"cancelling", "cancelled"}
            if cancelled:
                self._set_cancelled(job_id)
            elif returncode == 0:
                with self._lock:
                    job["status"] = "complete"
                    job["finished_at"] = time.time()
            else:
                error = f"Dagster materialization failed. See {log_path}."
                with self._lock:
                    job["status"] = "failed"
                    job["error"] = error
                    job["finished_at"] = time.time()
        except Exception as exc:
            with self._lock:
                cancelled = self.jobs[job_id]["status"] in {"cancelling", "cancelled"}
            if cancelled:
                self._set_cancelled(job_id)
            else:
                error = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    job["status"] = "failed"
                    job["error"] = error
                    job["finished_at"] = time.time()
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
                self._futures.pop(job_id, None)
            if runtime_config is not None:
                try:
                    runtime_config.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _artifact_is_current(path: Path, started_at: float | None) -> bool:
        if started_at is None or not path.exists():
            return False
        try:
            return path.stat().st_mtime >= started_at - 1.0
        except OSError:
            return False

    def _artifact_stages(self, job: dict) -> dict[str, dict]:
        source = self.settings.sources[job["source_key"]]
        embedding_path = source.embedding_cache_dir / "chunk_embeddings.npy"
        started_at = job.get("started_at")
        complete = {
            "normalize": self._artifact_is_current(source.normalized_audio_path, started_at),
            "transcribe": self._artifact_is_current(source.transcript_path, started_at),
            "speakers": self._artifact_is_current(source.speaker_roles_path, started_at),
            "chunk": self._artifact_is_current(source.speaker_chunks_path, started_at),
            "embed": self._artifact_is_current(embedding_path, started_at),
        }

        stages: dict[str, dict] = {}
        first_incomplete = None
        for stage in _STAGE_ORDER:
            if complete[stage]:
                stages[stage] = {"status": "complete", "progress": 100.0}
            else:
                if first_incomplete is None:
                    first_incomplete = stage
                stages[stage] = {"status": "pending", "progress": None}
        if job["status"] == "running" and first_incomplete is not None:
            stages[first_incomplete] = {"status": "running", "progress": None}
        return stages

    def status(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = dict(self.jobs[job_id])

        stages = self._artifact_stages(job)
        overall = 0.0
        active_stage = None
        for stage in _STAGE_ORDER:
            info = stages.get(stage, {})
            status = info.get("status")
            progress = info.get("progress")
            weight = _STAGE_WEIGHTS[stage]
            if status == "complete":
                overall += weight
            elif status == "running":
                active_stage = stage
                if progress is not None:
                    overall += weight * (float(progress) / 100.0)

        job["overall_progress"] = round(overall, 1)
        job["active_stage"] = active_stage
        job["stages"] = stages
        job["progress_status"] = job["status"]
        job["dagster_home"] = str(self.dagster_home)
        job["max_workers"] = self.max_workers
        return job

    def list_jobs(self) -> list[dict]:
        with self._lock:
            ids = list(self.jobs)
        return [self.status(job_id) for job_id in ids]
