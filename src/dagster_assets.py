import json
from pathlib import Path

import dagster as dg
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydantic import PrivateAttr
from sentence_transformers import SentenceTransformer

from src.audio_utils import normalize_audio
from src.chunk import (
    chunk_transcript,
    load_speaker_labels,
)
from src.config import (
    SourceSettings,
    load_settings,
)
from src.diarize import (
    diarize_audio,
    load_diarization_pipeline,
)
from src.llm_clients import LlamaCppClient
from src.search import (
    build_dense_index,
    extract_texts,
    load_chunks,
)
from src.speaker_alignment import (
    align_transcript_speakers,
)
from src.speaker_roles import (
    infer_speaker_roles,
    load_speaker_samples,
    save_speaker_roles,
)
from src.transcribe import transcribe_audio


SETTINGS = load_settings()


audio_partitions = dg.DynamicPartitionsDefinition(
    name="audio_files"
)


def get_partition_source(
    context: dg.AssetExecutionContext,
) -> SourceSettings:
    return SETTINGS.get_source(
        context.partition_key
    )


class WhisperResource(dg.ConfigurableResource):
    model_size: str
    device: str = "cuda"
    compute_type: str = "int8"

    _model: WhisperModel = PrivateAttr()

    def setup_for_execution(
        self,
        context: dg.InitResourceContext,
    ) -> None:
        context.log.info(
            f"Loading Whisper model: "
            f"{self.model_size}"
        )

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    @property
    def model(self) -> WhisperModel:
        return self._model


class DiarizationResource(
    dg.ConfigurableResource
):
    model_name: str

    _pipeline: Pipeline = PrivateAttr()

    def setup_for_execution(
        self,
        context: dg.InitResourceContext,
    ) -> None:
        context.log.info(
            f"Loading diarization model: "
            f"{self.model_name}"
        )

        self._pipeline = (
            load_diarization_pipeline(
                model_name=self.model_name,
            )
        )

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline


class EmbeddingResource(
    dg.ConfigurableResource
):
    model_name: str

    _model: SentenceTransformer = (
        PrivateAttr()
    )

    def setup_for_execution(
        self,
        context: dg.InitResourceContext,
    ) -> None:
        context.log.info(
            f"Loading embedding model: "
            f"{self.model_name}"
        )

        self._model = SentenceTransformer(
            self.model_name
        )

    @property
    def model(self) -> SentenceTransformer:
        return self._model


class LLMResource(dg.ConfigurableResource):
    base_url: str
    model_name: str
    timeout_seconds: float

    _client: LlamaCppClient = PrivateAttr()

    def setup_for_execution(
        self,
        context: dg.InitResourceContext,
    ) -> None:
        context.log.info(
            f"Configuring LLM client: "
            f"{self.base_url}"
        )

        self._client = LlamaCppClient(
            base_url=self.base_url,
            model=self.model_name,
            timeout=self.timeout_seconds,
        )

    @property
    def client(self) -> LlamaCppClient:
        return self._client


@dg.asset(
    partitions_def=audio_partitions,
)
def raw_audio(
    context: dg.AssetExecutionContext,
) -> str:
    source = get_partition_source(context)
    audio_path = source.raw_audio_path

    context.add_output_metadata(
        {
            "source_key": source.key,
            "source_id": source.source_id,
            "filename": source.audio_filename,
            "path": str(audio_path),
            "file_exists": audio_path.exists(),
            "size_bytes": (
                audio_path.stat().st_size
                if audio_path.exists()
                else 0
            ),
        }
    )

    return str(audio_path)


@dg.asset(
    partitions_def=audio_partitions,
)
def normalized_audio(
    context: dg.AssetExecutionContext,
    raw_audio: str,
) -> str:
    source = get_partition_source(context)

    output_path = normalize_audio(
        audio_path=raw_audio,
        output_path=(
            source.normalized_audio_path
        ),
    )

    context.add_output_metadata(
        {
            "source_key": source.key,
            "source_id": source.source_id,
            "path": str(output_path),
            "size_bytes": (
                output_path.stat().st_size
            ),
        }
    )

    return str(output_path)


@dg.asset(
    partitions_def=audio_partitions,
)
def transcript(
    context: dg.AssetExecutionContext,
    normalized_audio: str,
    whisper: WhisperResource,
) -> str:
    source = get_partition_source(context)

    output_path = transcribe_audio(
        audio=normalized_audio,
        output_path=source.transcript_path,
        model=whisper.model,
    )

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        segments = json.load(file)

    context.add_output_metadata(
        {
            "source_key": source.key,
            "source_id": source.source_id,
            "path": str(output_path),
            "segment_count": len(segments),
            "size_bytes": (
                output_path.stat().st_size
            ),
        }
    )

    return str(output_path)


@dg.asset(
    partitions_def=audio_partitions,
)
def diarization(
    context: dg.AssetExecutionContext,
    normalized_audio: str,
    diarizer: DiarizationResource,
) -> str:
    source = get_partition_source(context)

    output_path = diarize_audio(
        audio_path=normalized_audio,
        output_path=source.diarization_path,
        pipeline=diarizer.pipeline,
    )

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        turns = json.load(file)

    speakers = {
        turn["speaker"]
        for turn in turns
    }

    context.add_output_metadata(
        {
            "source_key": source.key,
            "source_id": source.source_id,
            "path": str(output_path),
            "speaker_count": len(speakers),
            "turn_count": len(turns),
            "size_bytes": (
                output_path.stat().st_size
            ),
        }
    )

    return str(output_path)


@dg.asset(
    partitions_def=audio_partitions,
)
def chunks(
    context: dg.AssetExecutionContext,
    transcript: str,
) -> str:
    source = get_partition_source(context)

    output_path, chunk_count = (
        chunk_transcript(
            transcription=transcript,
            output_path=source.chunks_path,
            source_id=source.source_id,
        )
    )

    context.add_output_metadata(
        {
            "source_key": source.key,
            "source_id": source.source_id,
            "path": str(output_path),
            "chunk_count": chunk_count,
            "size_bytes": (
                output_path.stat().st_size
            ),
        }
    )

    return str(output_path)


@dg.asset(
    partitions_def=audio_partitions,
)
def speaker_transcript(
    context: dg.AssetExecutionContext,
    transcript: str,
    diarization: str,
) -> str:
    source = get_partition_source(context)

    output_path = (
        align_transcript_speakers(
            transcript_path=transcript,
            diarization_path=diarization,
            output_path=(
                source.speaker_transcript_path
            ),
        )
    )

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        segments = json.load(file)

    unknown_count = sum(
        segment["speaker"] == "UNKNOWN"
        for segment in segments
    )

    below_point_nine = sum(
        segment[
            "speaker_overlap_ratio"
        ] < 0.9
        for segment in segments
    )

    context.add_output_metadata(
        {
            "source_key": source.key,
            "source_id": source.source_id,
            "path": str(output_path),
            "segment_count": len(segments),
            "unknown_count": unknown_count,
            "overlap_below_0_9": (
                below_point_nine
            ),
            "size_bytes": (
                output_path.stat().st_size
            ),
        }
    )

    return str(output_path)


@dg.asset(
    partitions_def=audio_partitions,
)
def speaker_roles(
    context: dg.AssetExecutionContext,
    speaker_transcript: str,
    llm: LLMResource,
) -> str:
    source = get_partition_source(context)

    samples = load_speaker_samples(
        speaker_transcript
    )

    artifact = infer_speaker_roles(
        samples_by_speaker=samples,
        source_id=source.source_id,
        llm_client=llm.client,
        manual_labels=(
            source.speaker_labels
        ),
    )

    output_path = save_speaker_roles(
        artifact=artifact,
        output_path=(
            source.speaker_roles_path
        ),
    )

    inferred_identity_count = sum(
        role.label_source
        == "inferred_identity"
        for role in artifact.roles.values()
    )

    inferred_role_count = sum(
        role.label_source
        == "inferred_role"
        for role in artifact.roles.values()
    )

    manual_override_count = sum(
        role.label_source
        == "manual_override"
        for role in artifact.roles.values()
    )

    context.add_output_metadata(
        {
            "source_key": source.key,
            "source_id": source.source_id,
            "path": str(output_path),
            "speaker_count": (
                len(artifact.roles)
            ),
            "inferred_identity_count": (
                inferred_identity_count
            ),
            "inferred_role_count": (
                inferred_role_count
            ),
            "manual_override_count": (
                manual_override_count
            ),
            "size_bytes": (
                output_path.stat().st_size
            ),
        }
    )

    return str(output_path)


@dg.asset(
    partitions_def=audio_partitions,
)
def speaker_chunks(
    context: dg.AssetExecutionContext,
    speaker_transcript: str,
    speaker_roles: str,
) -> str:
    source = get_partition_source(context)

    speaker_labels = load_speaker_labels(
        speaker_roles
    )

    output_path, chunk_count = (
        chunk_transcript(
            transcription=(
                speaker_transcript
            ),
            output_path=(
                source.speaker_chunks_path
            ),
            source_id=source.source_id,
            speaker_labels=speaker_labels,
        )
    )

    context.add_output_metadata(
        {
            "source_key": source.key,
            "source_id": source.source_id,
            "path": str(output_path),
            "chunk_count": chunk_count,
            "speaker_label_count": (
                len(speaker_labels)
            ),
            "size_bytes": (
                output_path.stat().st_size
            ),
        }
    )

    return str(output_path)


@dg.asset(
    partitions_def=audio_partitions,
)
def embeddings(
    context: dg.AssetExecutionContext,
    speaker_chunks: str,
    embedding: EmbeddingResource,
) -> str:
    source = get_partition_source(context)

    chunk_data = load_chunks(
        speaker_chunks
    )

    texts = extract_texts(
        chunk_data
    )

    cache_dir = (
        source.embedding_cache_dir
    )

    _, chunk_embeddings = (
        build_dense_index(
            texts,
            cache_dir=cache_dir,
            embedding_model=(
                embedding.model
            ),
            model_name=(
                embedding.model_name
            ),
        )
    )

    embeddings_path = (
        cache_dir
        / "chunk_embeddings.npy"
    )

    context.add_output_metadata(
        {
            "source_key": source.key,
            "source_id": source.source_id,
            "path": str(embeddings_path),
            "chunk_count": len(texts),
            "embedding_dimensions": (
                chunk_embeddings.shape[1]
            ),
            "size_bytes": (
                embeddings_path.stat().st_size
            ),
        }
    )

    return str(embeddings_path)


defs = dg.Definitions(
    assets=[
        raw_audio,
        normalized_audio,
        transcript,
        diarization,
        chunks,
        speaker_transcript,
        speaker_roles,
        speaker_chunks,
        embeddings,
    ],
    resources={
        "whisper": WhisperResource(
            model_size=(
                SETTINGS.models.whisper_model
            ),
            device="cuda",
            compute_type="int8",
        ),
        "diarizer": DiarizationResource(
            model_name=(
                SETTINGS.models
                .diarization_model
            ),
        ),
        "embedding": EmbeddingResource(
            model_name=(
                SETTINGS.models
                .embedding_model
            ),
        ),
        "llm": LLMResource(
            base_url=SETTINGS.llm.base_url,
            model_name=SETTINGS.llm.model,
            timeout_seconds=(
                SETTINGS.llm.timeout_seconds
            ),
        ),
    },
)