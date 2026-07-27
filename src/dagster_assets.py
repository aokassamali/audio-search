from pathlib import Path

import dagster as dg
from faster_whisper import WhisperModel
from pydantic import PrivateAttr
from sentence_transformers import SentenceTransformer

from src.chunk import chunk_transcript
from src.config import (
    SourceSettings,
    load_settings,
)
from src.search import (
    build_dense_index,
    extract_texts,
    load_chunks,
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


class EmbeddingResource(dg.ConfigurableResource):
    model_name: str

    _model: SentenceTransformer = PrivateAttr()

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
def transcript(
    context: dg.AssetExecutionContext,
    raw_audio: str,
    whisper: WhisperResource,
) -> str:
    source = get_partition_source(context)
    audio_path = Path(raw_audio)

    output_path = transcribe_audio(
        audio=audio_path,
        output_path=source.transcript_path,
        model=whisper.model,
    )

    context.add_output_metadata(
        {
            "source_key": source.key,
            "source_id": source.source_id,
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
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
    transcript_path = Path(transcript)

    output_path, chunk_count = chunk_transcript(
        transcription=transcript_path,
        output_path=source.chunks_path,
        source_id=source.source_id,
    )

    context.add_output_metadata(
        {
            "source_key": source.key,
            "source_id": source.source_id,
            "path": str(output_path),
            "chunk_count": chunk_count,
            "size_bytes": output_path.stat().st_size,
        }
    )

    return str(output_path)


@dg.asset(
    partitions_def=audio_partitions,
)
def embeddings(
    context: dg.AssetExecutionContext,
    chunks: str,
    embedding: EmbeddingResource,
) -> str:
    source = get_partition_source(context)

    chunk_data = load_chunks(chunks)
    texts = extract_texts(chunk_data)

    cache_dir = source.embedding_cache_dir

    _, chunk_embeddings = build_dense_index(
        texts,
        cache_dir=cache_dir,
        embedding_model=embedding.model,
        model_name=embedding.model_name,
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
        transcript,
        chunks,
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
        "embedding": EmbeddingResource(
            model_name=(
                SETTINGS.models.embedding_model
            ),
        ),
    },
)