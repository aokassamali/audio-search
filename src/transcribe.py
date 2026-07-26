from faster_whisper import WhisperModel
import argparse
import json
from datetime import datetime
from pathlib import Path

def create_output_path(audio: str) -> Path:
    audio_path = Path(audio)
    output_directory = Path("data/processed/jsons")
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / f"{audio_path.stem}.json"

    if output_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_directory / (
            f"{audio_path.stem}_{timestamp}.json"
        )

    return output_path


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
        output_path = create_output_path(audio)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--model-size", default="medium")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()

    output_path = transcribe_audio(
        audio=args.audio,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )

    print(f"Saved transcript to {output_path}")

if __name__ == "__main__":
    main()