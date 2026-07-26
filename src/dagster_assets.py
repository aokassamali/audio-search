from pathlib import Path
from src.transcribe import transcribe_audio
import dagster as dg
from faster_whisper import WhisperModel
from pydantic import PrivateAttr
from src.chunk import chunk_transcript
from sentence_transformers import SentenceTransformer

from src.search import (
    EMBEDDING_MODEL_NAME,
    build_dense_index,
    extract_texts,
    load_chunks,
)


AUDIO_DIR = Path("data/raw")
TRANSCRIPT_DIR = Path("data\processed\jsons")
CHUNKS_DIR = Path("data\processed\chunks")
EMBEDDING_CACHE_ROOT = Path("data\cache")

audio_partitions = dg.DynamicPartitionsDefinition(
    name="audio_files"
)

class WhisperResource(dg.ConfigurableResource):
    model_size: str = "medium"
    device: str = "cuda"
    compute_type: str = "int8"

    _model: WhisperModel = PrivateAttr()

    def setup_for_execution(
        self,
        context: dg.InitResourceContext,
    ) -> None:
        context.log.info(
            f"Loading Whisper model: {self.model_size}"
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
    model_name: str = EMBEDDING_MODEL_NAME

    _model: SentenceTransformer = PrivateAttr()

    def setup_for_execution(
        self,
        context: dg.InitResourceContext,
    ) -> None:
        context.log.info(
            f"Loading embedding model: {self.model_name}"
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
    audio_filename = context.partition_key
    audio_path = AUDIO_DIR / audio_filename

    context.add_output_metadata(
        {
            "filename": audio_filename,
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
    audio_path = Path(raw_audio)

    output_path = (
        TRANSCRIPT_DIR
        / f"{audio_path.stem}.json"
    )

    output_path = transcribe_audio(
        audio=audio_path,
        output_path=output_path,
        model=whisper.model,
    )

    context.add_output_metadata(
        {
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
    transcript_path = Path(transcript)

    output_path = (
        CHUNKS_DIR
        / f"{transcript_path.stem}_chunks.json"
    )

    output_path, chunk_count = chunk_transcript(
        transcription=transcript_path,
        output_path=output_path,
    )

    context.add_output_metadata(
        {
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
    chunk_data = load_chunks(chunks)
    texts = extract_texts(chunk_data)

    chunks_path = Path(chunks)

    cache_dir = (
        EMBEDDING_CACHE_ROOT
        / chunks_path.stem
    )

    _, chunk_embeddings = build_dense_index(
        texts,
        cache_dir=cache_dir,
        embedding_model=embedding.model,
    )

    embeddings_path = (
        cache_dir
        / "chunk_embeddings.npy"
    )

    context.add_output_metadata(
        {
            "path": str(embeddings_path),
            "chunk_count": len(texts),
            "embedding_dimensions": (
                chunk_embeddings.shape[1]
            ),
            "size_bytes": embeddings_path.stat().st_size,
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
            model_size="medium",
            device="cuda",
            compute_type="int8",
        ),
        "embedding": EmbeddingResource(
            model_name=EMBEDDING_MODEL_NAME,
        ),
    },
)