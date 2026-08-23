import json
import os
from pathlib import Path

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


MODEL_NAME = "pyannote/speaker-diarization-community-1"


def load_diarization_pipeline(
    token: str | None = None,
    model_name: str = MODEL_NAME,
    device: str = "cuda",
) -> Pipeline:
    token = token or os.getenv("HF_TOKEN")

    pipeline = Pipeline.from_pretrained(
        model_name,
        token=token,
    )

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Diarization is configured for CUDA, but CUDA is unavailable. "
            "Set AUDIO_SEARCH_DIARIZATION_DEVICE=cpu to run on CPU."
        )

    pipeline.to(target_device)
    return pipeline


def diarize_audio(
    audio_path: str | Path,
    output_path: str | Path,
    pipeline: Pipeline | None = None,
    device: str = "cuda",
) -> Path:
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if pipeline is None:
        pipeline = load_diarization_pipeline(
            device=device,
        )

    with ProgressHook() as hook:
        output = pipeline(
            str(audio_path),
            hook=hook,
            preload=False,
        )

    turns = []

    for turn, speaker in (
        output.exclusive_speaker_diarization
    ):
        turns.append(
            {
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
            }
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            turns,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path
