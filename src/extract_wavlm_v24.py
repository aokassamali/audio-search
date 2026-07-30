import argparse
import json
import time
import wave
from datetime import datetime, timezone
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
SAMPLE_RATE = 16_000


def load_wav_slice(
    audio_path: Path,
    start: float,
    end: float,
) -> np.ndarray | None:
    with wave.open(
        str(audio_path),
        "rb",
    ) as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()

        if channels != 1:
            print(
                f"Expected mono audio: {audio_path}"
            )
            return None

        if sample_rate != SAMPLE_RATE:
            print(
                f"Expected {SAMPLE_RATE} Hz: "
                f"{audio_path}"
            )
            return None

        if sample_width != 2:
            print(
                f"Expected 16-bit PCM: "
                f"{audio_path}"
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
            print(
                f"Empty audio interval: "
                f"{start:.2f}-{end:.2f}"
            )
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

    return audio


def create_windows(
    audio: np.ndarray,
    window_seconds: float,
) -> list[np.ndarray]:
    window_size = int(
        window_seconds * SAMPLE_RATE
    )

    minimum_tail_size = SAMPLE_RATE

    windows = []
    start = 0

    while start < len(audio):
        end = min(
            start + window_size,
            len(audio),
        )

        remaining = len(audio) - end

        if (
            remaining > 0
            and remaining < minimum_tail_size
        ):
            end = len(audio)

        windows.append(
            audio[start:end]
        )

        start = end

    return windows


def extract_turn_embedding(
    audio: np.ndarray,
    feature_extractor,
    model: WavLMModel,
    device: torch.device,
    window_seconds: float,
    expected_layers: int,
    hidden_size: int,
) -> np.ndarray | None:
    windows = create_windows(
        audio=audio,
        window_seconds=window_seconds,
    )

    layer_sums = torch.zeros(
        (
            expected_layers,
            hidden_size,
        ),
        dtype=torch.float64,
    )

    total_frames = 0

    for audio_window in windows:
        inputs = feature_extractor(
            audio_window,
            sampling_rate=SAMPLE_RATE,
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
            return None

        if len(hidden_states) != expected_layers:
            print(
                "Unexpected layer count. "
                f"Expected {expected_layers}, "
                f"found {len(hidden_states)}."
            )
            return None

        frame_count = int(
            hidden_states[0].shape[1]
        )

        window_sums = torch.stack(
            [
                layer.sum(
                    dim=1
                ).squeeze(0)
                for layer in hidden_states
            ]
        )

        window_sums = window_sums.to(
            device="cpu",
            dtype=torch.float64,
        )

        layer_sums += window_sums
        total_frames += frame_count

    if total_frames == 0:
        print("No encoder frames produced.")
        return None

    pooled = (
        layer_sums / total_frames
    ).to(dtype=torch.float32)

    if not torch.isfinite(pooled).all():
        print(
            "Non-finite embedding values found."
        )
        return None

    return pooled.numpy()


def load_cached_embedding(
    path: Path,
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    if not path.exists():
        return None

    try:
        embedding = np.load(
            path,
            allow_pickle=False,
        )
    except (OSError, ValueError):
        return None

    if embedding.shape != expected_shape:
        return None

    if not np.isfinite(embedding).all():
        return None

    return embedding


def save_array_atomic(
    path: Path,
    array: np.ndarray,
) -> None:
    temporary_path = path.with_suffix(
        ".tmp"
    )

    with temporary_path.open("wb") as file:
        np.save(
            file,
            array,
            allow_pickle=False,
        )

    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=[
            "cuda",
            "cpu",
        ],
        default="cuda",
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=20.0,
    )

    arguments = parser.parse_args()

    if (
        arguments.device == "cuda"
        and not torch.cuda.is_available()
    ):
        print(
            "CUDA is unavailable. "
            "Use --device cpu explicitly."
        )
        return

    settings = load_settings()
    eval_dir = settings.paths.eval_dir

    gold_path = (
        eval_dir
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

    device = torch.device(
        arguments.device
    )

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(
        "Window length: "
        f"{arguments.window_seconds:.1f}s"
    )

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

    expected_layers = (
        model.config.num_hidden_layers + 1
    )

    hidden_size = int(
        model.config.hidden_size
    )

    expected_shape = (
        expected_layers,
        hidden_size,
    )

    cache_dir = (
        settings.paths.embedding_cache_root
        / "wavlm_base_plus_v24"
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    started = time.perf_counter()
    newly_extracted = 0
    cached_count = 0

    for item_index, item in enumerate(
        artifact.items
    ):
        cache_path = (
            cache_dir
            / f"turn_{item_index:04d}.npy"
        )

        cached = load_cached_embedding(
            path=cache_path,
            expected_shape=expected_shape,
        )

        if cached is not None:
            cached_count += 1
            continue

        source = settings.get_source(
            item.source_key
        )

        audio = load_wav_slice(
            audio_path=(
                source.normalized_audio_path
            ),
            start=item.start,
            end=item.end,
        )

        if audio is None:
            print(
                f"Failed on {item.turn_id}"
            )
            return

        try:
            embedding = extract_turn_embedding(
                audio=audio,
                feature_extractor=(
                    feature_extractor
                ),
                model=model,
                device=device,
                window_seconds=(
                    arguments.window_seconds
                ),
                expected_layers=(
                    expected_layers
                ),
                hidden_size=hidden_size,
            )
        except torch.cuda.OutOfMemoryError:
            print(
                "\nCUDA ran out of memory."
                "\nRerun with:"
                "\n  --window-seconds 10"
            )
            return

        if embedding is None:
            print(
                f"Failed on {item.turn_id}"
            )
            return

        save_array_atomic(
            path=cache_path,
            array=embedding,
        )

        newly_extracted += 1

        elapsed = (
            time.perf_counter() - started
        )

        print(
            f"[{item_index + 1}/"
            f"{len(artifact.items)}] "
            f"{item.turn_id} "
            f"{item.duration:.1f}s "
            f"saved "
            f"({elapsed / 60:.1f} min)"
        )

    cached_embeddings = []

    for item_index in range(
        len(artifact.items)
    ):
        cache_path = (
            cache_dir
            / f"turn_{item_index:04d}.npy"
        )

        embedding = load_cached_embedding(
            path=cache_path,
            expected_shape=expected_shape,
        )

        if embedding is None:
            print(
                "Extraction incomplete. "
                f"Missing turn {item_index}."
            )
            return

        cached_embeddings.append(
            embedding
        )

    embedding_matrix = np.stack(
        cached_embeddings
    ).astype(np.float32)

    embeddings_path = (
        eval_dir
        / "wavlm_base_plus_v24.npy"
    )

    save_array_atomic(
        path=embeddings_path,
        array=embedding_matrix,
    )

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_name": MODEL_NAME,
        "sample_rate": SAMPLE_RATE,
        "window_seconds": (
            arguments.window_seconds
        ),
        "window_overlap_seconds": 0.0,
        "pooling": (
            "mean over encoder frames, "
            "weighted by frame count across "
            "non-overlapping windows"
        ),
        "transformer_layer_count": (
            model.config.num_hidden_layers
        ),
        "hidden_state_count": (
            expected_layers
        ),
        "hidden_size": hidden_size,
        "embedding_shape": list(
            embedding_matrix.shape
        ),
        "dtype": str(
            embedding_matrix.dtype
        ),
        "items": [
            {
                "row_index": item_index,
                "turn_id": item.turn_id,
                "source_key": (
                    item.source_key
                ),
                "source_id": item.source_id,
                "speaker": item.speaker,
                "provisional_group": (
                    f"{item.source_key}:"
                    f"{item.speaker}"
                ),
                "duration": item.duration,
                "label_permissive": (
                    item.label_permissive
                ),
                "label_strict": (
                    item.label_strict
                ),
                "difficulty": (
                    item.difficulty
                ),
                "frozen": item.frozen,
                "cache_file": (
                    f"turn_{item_index:04d}.npy"
                ),
            }
            for item_index, item
            in enumerate(artifact.items)
        ],
    }

    manifest_path = (
        eval_dir
        / "wavlm_base_plus_v24_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    elapsed = time.perf_counter() - started

    print("\nExtraction complete")
    print(
        f"Newly extracted: "
        f"{newly_extracted}"
    )
    print(
        f"Loaded from cache: "
        f"{cached_count}"
    )
    print(
        f"Shape: "
        f"{embedding_matrix.shape}"
    )
    print(
        f"Elapsed: "
        f"{elapsed / 60:.1f} minutes"
    )
    print(f"Saved: {embeddings_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()