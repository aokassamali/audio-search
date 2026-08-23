import argparse
import json
import os
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel

from src.config import load_settings


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_output_path(
    audio: str | Path,
    output_directory: str | Path,
) -> Path:
    audio_path = Path(audio)
    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        output_directory
        / f"{audio_path.stem}.json"
    )


def transcribe_audio(
    audio,
    output_path=None,
    model_size="medium",
    device="cuda",
    compute_type="int8",
    model=None,
    progress_callback=None,
    batch_size: int | None = None,
    beam_size: int | None = None,
    vad_filter: bool | None = None,
):
    if model is None:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    if batch_size is None:
        batch_size = max(
            1,
            int(os.getenv("AUDIO_SEARCH_WHISPER_BATCH_SIZE", "1")),
        )

    if beam_size is None:
        beam_size = max(
            1,
            int(os.getenv("AUDIO_SEARCH_WHISPER_BEAM_SIZE", "5")),
        )

    if vad_filter is None:
        vad_filter = _env_bool(
            "AUDIO_SEARCH_WHISPER_VAD",
            default=False,
        )

    if batch_size > 1:
        inference = BatchedInferencePipeline(
            model=model,
        )
        segments, info = inference.transcribe(
            audio,
            batch_size=batch_size,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
    else:
        segments, info = model.transcribe(
            audio,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    json_list = []

    for segment in segments:
        json_list.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
        )

        if progress_callback is not None and duration > 0:
            progress_callback(
                min(
                    100.0,
                    max(
                        0.0,
                        (float(segment.end) / duration) * 100.0,
                    ),
                )
            )

    if progress_callback is not None:
        progress_callback(100.0)

    if output_path is None:
        settings = load_settings()

        output_path = create_output_path(
            audio=audio,
            output_directory=(
                settings.paths.transcript_dir
            ),
        )
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_list,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def main():
    settings = load_settings()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "audio",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--model-size",
        default=None,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--compute-type",
        default="int8",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--vad-filter",
        action="store_true",
        default=None,
    )

    args = parser.parse_args()

    model_size = (
        args.model_size
        or settings.models.whisper_model
    )

    output_path = transcribe_audio(
        audio=args.audio,
        output_path=args.output,
        model_size=model_size,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
    )

    print(f"Saved transcript to {output_path}")


if __name__ == "__main__":
    main()
