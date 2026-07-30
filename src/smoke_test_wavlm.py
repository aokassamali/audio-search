import json
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    WavLMModel,
)

from src.config import load_settings
from src.speech_act_gold import (
    SpeechActGoldArtifact,
)


MODEL_NAME = "microsoft/wavlm-base-plus"

MINIMUM_DURATION = 5.0
MAXIMUM_DURATION = 30.0


def load_wav_slice(
    audio_path: Path,
    start: float,
    end: float,
) -> tuple[np.ndarray, int] | None:
    with wave.open(
        str(audio_path),
        "rb",
    ) as wav_file:
        channel_count = (
            wav_file.getnchannels()
        )
        sample_rate = (
            wav_file.getframerate()
        )
        sample_width = (
            wav_file.getsampwidth()
        )
        frame_count = (
            wav_file.getnframes()
        )

        if channel_count != 1:
            print(
                "Expected mono audio. Found "
                f"{channel_count} channels."
            )
            return None

        if sample_rate != 16_000:
            print(
                "Expected 16000 Hz audio. Found "
                f"{sample_rate} Hz."
            )
            return None

        if sample_width != 2:
            print(
                "Expected 16-bit PCM audio. Found "
                f"{sample_width * 8}-bit audio."
            )
            return None

        start_frame = max(
            0,
            int(start * sample_rate),
        )

        end_frame = min(
            frame_count,
            int(end * sample_rate),
        )

        if end_frame <= start_frame:
            print("Audio slice is empty.")
            return None

        wav_file.setpos(start_frame)

        raw_audio = wav_file.readframes(
            end_frame - start_frame
        )

    audio = np.frombuffer(
        raw_audio,
        dtype="<i2",
    ).astype(np.float32)

    audio /= 32768.0

    return audio, sample_rate


def main() -> None:
    settings = load_settings()

    gold_path = (
        settings.paths.eval_dir
        / "speech_act_gold_v24.json"
    )

    with gold_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = (
            SpeechActGoldArtifact.model_validate(
                json.load(file)
            )
        )

    candidates = [
        item
        for item in artifact.items
        if (
            not item.frozen
            and MINIMUM_DURATION
            <= item.duration
            <= MAXIMUM_DURATION
        )
    ]

    if not candidates:
        print(
            "No suitable smoke-test turn found."
        )
        return

    item = min(
        candidates,
        key=lambda candidate: abs(
            candidate.duration - 15.0
        ),
    )

    source = settings.get_source(
        item.source_key
    )

    loaded_audio = load_wav_slice(
        audio_path=(
            source.normalized_audio_path
        ),
        start=item.start,
        end=item.end,
    )

    if loaded_audio is None:
        return

    audio, sample_rate = loaded_audio

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Turn: {item.turn_id}")
    print(f"Source: {item.source_key}")
    print(f"Duration: {item.duration:.2f}s")
    print(f"Samples: {len(audio)}")
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    feature_extractor = (
        AutoFeatureExtractor.from_pretrained(
            MODEL_NAME
        )
    )

    model = WavLMModel.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    inputs = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    input_values = inputs[
        "input_values"
    ].to(device)

    with torch.inference_mode():
        outputs = model(
            input_values=input_values,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states

    if hidden_states is None:
        print("No hidden states returned.")
        return

    pooled_layers = torch.stack(
        [
            layer.mean(dim=1).squeeze(0)
            for layer in hidden_states
        ]
    ).cpu()

    expected_layers = (
        model.config.num_hidden_layers + 1
    )

    print(
        "\nTransformer layers: "
        f"{model.config.num_hidden_layers}"
    )
    print(
        "Hidden-state tensors: "
        f"{len(hidden_states)}"
    )
    print(
        "Expected hidden-state tensors: "
        f"{expected_layers}"
    )
    print(
        "Frame-level final shape: "
        f"{tuple(outputs.last_hidden_state.shape)}"
    )
    print(
        "Pooled layer shape: "
        f"{tuple(pooled_layers.shape)}"
    )
    print(
        "All values finite: "
        f"{bool(torch.isfinite(
            pooled_layers
        ).all())}"
    )


if __name__ == "__main__":
    main()