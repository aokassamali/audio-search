from __future__ import annotations

import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import dagster as dg

from src.config import PROJECT_ROOT, Settings, SourceSettings
from src.ingest_progress import (
    initialize,
    mark_complete,
    mark_failed,
    mark_running,
    snapshot,
)


_STAGE_ORDER = [
    "normalize",
    "transcribe",
    "speakers",
    "chunk",
    "embed",
]

_STAGE_WEIGHTS = {
    "normalize": 5.0,
    "transcribe": 45.0,
    "speakers": 35.0,
    "chunk": 5.0,
    "embed": 10.0,
}


class AudioIngestManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs: dict[str, dict] = {}
        self._lock = Lock()

        max_workers = max(
            1,
            int(os.getenv("AUDIO_SEARCH_INGEST_WORKERS", "1")),
        )
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="audio-ingest",
        )

        dagster_home = Path(
            os.getenv(
                "DAGSTER_HOME",
                str(PROJECT_ROOT / "data" / "dagster_home"),
            )
        )
        dagster_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("DAGSTER_HOME", str(dagster_home))
        self.dagster_home = dagster_home
        self.instance = dg.DagsterInstance.get()
        self.max_workers = max_workers

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "-",
            value.strip(),
        ).strip("-").lower()
        return slug or "audio-source"

    def _unique_source_key(self, display_name: str) -> str:
        base = self._slugify(display_name)
        key = base
        suffix = 2
        with self._lock:
            known = set(self.settings.sources)
            known.update(
                job["source_key"]
                for job in self.jobs.values()
            )
            while key in known:
                key = f"{base}-{suffix}"
                suffix += 1
        return key

    def submit(
        self,
        audio_path: Path,
        display_name: str,
    ) -> dict:
        source_key = self._unique_source_key(display_name)
        source_id = source_key
        source = SourceSettings(
            key=source_key,
            source_id=source_id,
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
            "source_id": source_id,
            "display_name": display_name,
            "filename": audio_path.name,
            "audio_path": str(audio_path),
            "status": "queued",
            "submitted_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "run_id": None,
        }

        with self._lock:
            self.jobs[job_id] = job

        initialize(source_key)
        self.executor.submit(self._run, job_id)
        return self.status(job_id)

    def _run(self, job_id: str) -> None:
        from src import dagster_assets as pipeline

        with self._lock:
            job = self.jobs[job_id]
            job["status"] = "running"
            job["started_at"] = time.time()

        source_key = job["source_key"]
        mark_running(source_key)

        source = self.settings.sources[source_key]
        pipeline.SETTINGS.sources[source_key] = source

        try:
            self.instance.add_dynamic_partitions(
                pipeline.audio_partitions.name,
                [source_key],
            )

            resources = {
                "whisper": pipeline.WhisperResource(
                    model_size=self.settings.models.whisper_model,
                    device="cuda",
                    compute_type="int8",
                ),
                "diarizer": pipeline.DiarizationResource(
                    model_name=self.settings.models.diarization_model,
                ),
                "embedding": pipeline.EmbeddingResource(
                    model_name=self.settings.models.embedding_model,
                ),
                "llm": pipeline.LLMResource(
                    base_url=self.settings.llm.base_url,
                    model_name=self.settings.llm.model,
                    timeout_seconds=self.settings.llm.timeout_seconds,
                ),
            }

            assets = [
                pipeline.raw_audio,
                pipeline.normalized_audio,
                pipeline.transcript,
                pipeline.diarization,
                pipeline.chunks,
                pipeline.speaker_transcript,
                pipeline.speaker_roles,
                pipeline.speaker_chunks,
                pipeline.embeddings,
            ]

            result = dg.materialize(
                assets=assets,
                resources=resources,
                partition_key=source_key,
                instance=self.instance,
                raise_on_error=False,
                tags={
                    "audio_search/source_key": source_key,
                    "audio_search/job_id": job_id,
                    "audio_search/origin": "frontend",
                },
            )

            with self._lock:
                job["run_id"] = getattr(result, "run_id", None)

            if result.success:
                mark_complete(source_key)
                with self._lock:
                    job["status"] = "complete"
                    job["finished_at"] = time.time()
            else:
                error = "Dagster materialization failed. Check Dagster run logs."
                mark_failed(source_key, error)
                with self._lock:
                    job["status"] = "failed"
                    job["error"] = error
                    job["finished_at"] = time.time()

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            mark_failed(source_key, error)
            with self._lock:
                job["status"] = "failed"
                job["error"] = error
                job["finished_at"] = time.time()

    def _artifact_stages(self, source_key: str, job_status: str) -> dict[str, dict]:
        source = self.settings.sources[source_key]
        embedding_path = source.embedding_cache_dir / "chunk_embeddings.npy"

        complete = {
            "normalize": source.normalized_audio_path.exists(),
            "transcribe": source.transcript_path.exists(),
            "speakers": source.speaker_roles_path.exists(),
            "chunk": source.speaker_chunks_path.exists(),
            "embed": embedding_path.exists(),
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

        if job_status == "running" and first_incomplete is not None:
            stages[first_incomplete] = {"status": "running", "progress": None}

        live = snapshot(source_key)
        if live:
            for stage, info in live.get("stages", {}).items():
                if info.get("status") == "running" and info.get("progress") is not None:
                    stages[stage] = dict(info)

        return stages

    def status(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = dict(self.jobs[job_id])

        stages = self._artifact_stages(job["source_key"], job["status"])

        overall = 0.0
        active_stage = None
        for stage in _STAGE_ORDER:
            info = stages.get(stage, {})
            stage_status = info.get("status")
            stage_progress = info.get("progress")
            weight = _STAGE_WEIGHTS[stage]
            if stage_status == "complete":
                overall += weight
            elif stage_status == "running":
                active_stage = stage
                if stage_progress is not None:
                    overall += weight * (float(stage_progress) / 100.0)

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
