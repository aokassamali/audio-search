import subprocess
from pathlib import Path


def normalize_audio(
    audio_path: str | Path,
    output_path: str | Path,
) -> Path:
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        check=True,
    )

    return output_path