import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "audio_search.toml"
)


def resolve_path(
    value: str,
    base_dir: Path,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return base_dir / path


@dataclass(frozen=True)
class PathSettings:
    raw_audio_dir: Path
    normalized_audio_dir: Path
    transcript_dir: Path
    diarization_dir: Path
    speaker_transcript_dir: Path
    chunks_dir: Path
    speaker_chunks_dir: Path
    embedding_cache_root: Path
    eval_dir: Path


@dataclass(frozen=True)
class ModelSettings:
    embedding_model: str
    whisper_model: str
    diarization_model: str


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    model: str
    timeout_seconds: float


@dataclass(frozen=True)
class SourceSettings:
    key: str
    source_id: str
    audio_filename: str
    chunk_variant: str
    paths: PathSettings

    @property
    def raw_audio_path(self) -> Path:
        return (
            self.paths.raw_audio_dir
            / self.audio_filename
        )

    @property
    def normalized_audio_path(self) -> Path:
        return (
            self.paths.normalized_audio_dir
            / f"{self.source_id}.wav"
        )

    @property
    def transcript_path(self) -> Path:
        return (
            self.paths.transcript_dir
            / f"{self.source_id}.json"
        )

    @property
    def diarization_path(self) -> Path:
        return (
            self.paths.diarization_dir
            / f"{self.source_id}_diarization.json"
        )

    @property
    def speaker_transcript_path(self) -> Path:
        return (
            self.paths.speaker_transcript_dir
            / f"{self.source_id}_speakers.json"
        )

    @property
    def chunks_path(self) -> Path:
        return (
            self.paths.chunks_dir
            / f"{self.source_id}_chunks.json"
        )

    @property
    def speaker_chunks_path(self) -> Path:
        return (
            self.paths.speaker_chunks_dir
            / f"{self.source_id}_speaker_chunks.json"
        )

    @property
    def active_chunks_path(self) -> Path:
        if self.chunk_variant == "speaker":
            return self.speaker_chunks_path

        return self.chunks_path

    @property
    def embedding_cache_dir(self) -> Path:
        return (
            self.paths.embedding_cache_root
            / self.source_id
        )


@dataclass(frozen=True)
class Settings:
    default_source_key: str
    chunk_variant: str
    paths: PathSettings
    models: ModelSettings
    llm: LLMSettings
    sources: dict[str, SourceSettings]

    def get_source(
        self,
        source_key: str | None = None,
    ) -> SourceSettings:
        selected_key = (
            source_key
            or os.getenv("AUDIO_SEARCH_SOURCE")
            or self.default_source_key
        )

        return self.sources[selected_key]

    @property
    def rag_eval_path(self) -> Path:
        return self.paths.eval_dir / "rag_eval.json"

    @property
    def rag_eval_results_path(self) -> Path:
        return (
            self.paths.eval_dir
            / "rag_eval_results.json"
        )


def load_settings(
    config_path: str | Path | None = None,
) -> Settings:
    configured_path = (
        config_path
        or os.getenv("AUDIO_SEARCH_CONFIG")
        or DEFAULT_CONFIG_PATH
    )

    config_path = Path(configured_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open("rb") as file:
        data = tomllib.load(file)

    config_dir = config_path.parent

    app_data = data["app"]
    path_data = data["paths"]
    model_data = data["models"]
    llm_data = data["llm"]

    paths = PathSettings(
        raw_audio_dir=resolve_path(
            path_data["raw_audio_dir"],
            config_dir,
        ),
        normalized_audio_dir=resolve_path(
            path_data["normalized_audio_dir"],
            config_dir,
        ),
        transcript_dir=resolve_path(
            path_data["transcript_dir"],
            config_dir,
        ),
        diarization_dir=resolve_path(
            path_data["diarization_dir"],
            config_dir,
        ),
        speaker_transcript_dir=resolve_path(
            path_data["speaker_transcript_dir"],
            config_dir,
        ),
        chunks_dir=resolve_path(
            path_data["chunks_dir"],
            config_dir,
        ),
        speaker_chunks_dir=resolve_path(
            path_data["speaker_chunks_dir"],
            config_dir,
        ),
        embedding_cache_root=resolve_path(
            path_data["embedding_cache_root"],
            config_dir,
        ),
        eval_dir=resolve_path(
            path_data["eval_dir"],
            config_dir,
        ),
    )

    chunk_variant = os.getenv(
        "AUDIO_SEARCH_CHUNK_VARIANT",
        app_data.get("chunk_variant", "plain"),
    )

    sources = {}

    for source_key, source_data in data[
        "sources"
    ].items():
        audio_filename = source_data[
            "audio_filename"
        ]

        source_id = source_data.get(
            "source_id",
            Path(audio_filename).stem,
        )

        sources[source_key] = SourceSettings(
            key=source_key,
            source_id=source_id,
            audio_filename=audio_filename,
            chunk_variant=chunk_variant,
            paths=paths,
        )

    models = ModelSettings(
        embedding_model=model_data[
            "embedding_model"
        ],
        whisper_model=model_data[
            "whisper_model"
        ],
        diarization_model=model_data[
            "diarization_model"
        ],
    )

    llm = LLMSettings(
        base_url=os.getenv(
            "LLAMA_CPP_BASE_URL",
            llm_data["base_url"],
        ),
        model=os.getenv(
            "LLAMA_CPP_MODEL",
            llm_data["model"],
        ),
        timeout_seconds=float(
            os.getenv(
                "LLAMA_CPP_TIMEOUT_SECONDS",
                llm_data["timeout_seconds"],
            )
        ),
    )

    return Settings(
        default_source_key=app_data[
            "default_source_id"
        ],
        chunk_variant=chunk_variant,
        paths=paths,
        models=models,
        llm=llm,
        sources=sources,
    )