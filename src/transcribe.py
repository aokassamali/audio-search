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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--model-size", default="medium")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()

    model = WhisperModel(
        args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )

    segments, info = model.transcribe(args.audio, beam_size=5)

    json_list = [
        {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        for segment in segments
    ]

    output_path = create_output_path(args.audio)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(json_list, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()