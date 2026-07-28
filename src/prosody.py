from pathlib import Path

import opensmile

import wave

import numpy as np

def create_egemaps_extractor() -> opensmile.Smile:
    return opensmile.Smile(
        feature_set=(
            opensmile.FeatureSet.eGeMAPSv02
        ),
        feature_level=(
            opensmile.FeatureLevel.Functionals
        ),
    )


def extract_turn_features(
    audio_path: str | Path,
    start: float,
    end: float,
    word_count: int,
    extractor: opensmile.Smile,
) -> dict[str, float]:
    feature_frame = extractor.process_file(
        str(audio_path),
        start=start,
        end=end,
    )

    feature_row = feature_frame.iloc[0]

    features = {
        str(feature_name): float(value)
        for feature_name, value
        in feature_row.items()
    }

    duration = max(
        end - start,
        0.001,
    )

    features["duration_seconds"] = duration
    features["word_count"] = float(word_count)
    features["speaking_rate_wps"] = (
        word_count / duration
    )

    pause_features = extract_pause_features(
        audio_path=audio_path,
        start=start,
        end=end,
    )

    features.update(pause_features)

    return features

def count_internal_pauses(
    silent_frames: np.ndarray,
    minimum_pause_frames: int,
) -> int:
    voiced_indices = np.flatnonzero(
        ~silent_frames
    )

    if len(voiced_indices) == 0:
        return 0

    internal_silence = silent_frames[
        voiced_indices[0]:
        voiced_indices[-1] + 1
    ]

    pause_count = 0
    current_run = 0

    for is_silent in internal_silence:
        if is_silent:
            current_run += 1
        else:
            if current_run >= minimum_pause_frames:
                pause_count += 1

            current_run = 0

    if current_run >= minimum_pause_frames:
        pause_count += 1

    return pause_count


def extract_pause_features(
    audio_path: str | Path,
    start: float,
    end: float,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    silence_threshold_db: float = 20.0,
    minimum_pause_ms: float = 150.0,
) -> dict[str, float]:
    with wave.open(
        str(audio_path),
        "rb",
    ) as audio_file:
        sample_rate = (
            audio_file.getframerate()
        )

        start_frame = max(
            0,
            int(start * sample_rate),
        )

        end_frame = max(
            start_frame + 1,
            int(end * sample_rate),
        )

        start_frame = min(
            start_frame,
            audio_file.getnframes(),
        )

        audio_file.setpos(start_frame)

        raw_audio = audio_file.readframes(
            end_frame - start_frame
        )

    samples = np.frombuffer(
        raw_audio,
        dtype=np.int16,
    ).astype(np.float32)

    samples /= 32768.0

    if len(samples) == 0:
        return {
            "silence_proportion": 1.0,
            "pause_count": 0.0,
        }

    frame_length = max(
        1,
        int(sample_rate * frame_ms / 1000),
    )

    hop_length = max(
        1,
        int(sample_rate * hop_ms / 1000),
    )

    if len(samples) < frame_length:
        samples = np.pad(
            samples,
            (
                0,
                frame_length - len(samples),
            ),
        )

    frame_starts = range(
        0,
        len(samples) - frame_length + 1,
        hop_length,
    )

    rms_values = np.array(
        [
            np.sqrt(
                np.mean(
                    samples[
                        frame_start:
                        frame_start + frame_length
                    ] ** 2
                )
                + 1e-12
            )
            for frame_start in frame_starts
        ]
    )

    reference_rms = float(
        np.percentile(rms_values, 95)
    )

    silence_threshold = (
        reference_rms
        * 10 ** (
            -silence_threshold_db / 20
        )
    )

    silent_frames = (
        rms_values < silence_threshold
    )

    minimum_pause_frames = max(
        1,
        round(
            minimum_pause_ms / hop_ms
        ),
    )

    pause_count = count_internal_pauses(
        silent_frames,
        minimum_pause_frames,
    )

    return {
        "silence_proportion": float(
            silent_frames.mean()
        ),
        "pause_count": float(pause_count),
    }