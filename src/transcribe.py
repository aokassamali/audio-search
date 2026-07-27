from faster_whisper import WhisperModel
import argparse
import json
from src.config import load_settings
from pathlib import Path

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
):
    if model is None:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    segments, info = model.transcribe(
        audio,
        beam_size=5,
    )

    json_list = [
        {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        for segment in segments
    ]

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
    )

    print(f"Saved transcript to {output_path}")

if __name__ == "__main__":
    main()