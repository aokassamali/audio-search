from __future__ import annotations

import json
import re
from pathlib import Path

SUPPORTED_TRANSCRIPT_EXTENSIONS = {".json", ".srt", ".txt", ".vtt"}
TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[,.]\d{3})"
)
SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 ._'-]{0,60}):\s+(.+)$")
VTT_SPEAKER_RE = re.compile(r"^<v\s+([^>]+)>(.*)$", re.IGNORECASE)


def _timestamp_to_seconds(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Unsupported timestamp: {value}")
    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)


def _speaker_and_text(value: str) -> tuple[str | None, str]:
    value = value.strip()
    match = VTT_SPEAKER_RE.match(value)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    match = SPEAKER_RE.match(value)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, value


def _normalize_segment(item: dict, index: int) -> dict | None:
    text = str(item.get("text", "")).strip()
    if not text:
        return None
    speaker = item.get("speaker") or item.get("speaker_name")
    if speaker is None:
        speaker, text = _speaker_and_text(text)
    try:
        start = float(item.get("start", -1.0))
    except (TypeError, ValueError):
        start = -1.0
    try:
        end = float(item.get("end", -1.0))
    except (TypeError, ValueError):
        end = -1.0
    segment = {"text": text, "start": start, "end": end, "import_index": index}
    if speaker:
        segment["speaker"] = str(speaker).strip()
    return segment


def _parse_json(content: str) -> list[dict]:
    data = json.loads(content)
    if isinstance(data, dict):
        for key in ("segments", "transcript", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("JSON transcripts must be a list of segments, or contain a segments/transcript/items list.")
    segments = []
    for index, item in enumerate(data):
        if isinstance(item, str):
            item = {"text": item}
        if isinstance(item, dict):
            segment = _normalize_segment(item, index)
            if segment:
                segments.append(segment)
    return segments


def _parse_captions(content: str) -> list[dict]:
    lines = [line.rstrip("\ufeff") for line in content.splitlines()]
    segments = []
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor].strip()
        cursor += 1
        if not line or line.upper() == "WEBVTT" or line.isdigit():
            continue
        match = TIMESTAMP_RE.search(line)
        if not match:
            continue
        start = _timestamp_to_seconds(match.group("start"))
        end = _timestamp_to_seconds(match.group("end"))
        text_lines = []
        while cursor < len(lines) and lines[cursor].strip():
            text_lines.append(lines[cursor].strip())
            cursor += 1
        text = " ".join(text_lines).strip()
        if not text:
            continue
        speaker, text = _speaker_and_text(text)
        segment = {"text": text, "start": start, "end": end, "import_index": len(segments)}
        if speaker:
            segment["speaker"] = speaker
        segments.append(segment)
    return segments


def _parse_text(content: str) -> list[dict]:
    parts = [part.strip() for part in re.split(r"\n\s*\n|\n", content) if part.strip()]
    segments = []
    for index, part in enumerate(parts):
        speaker, text = _speaker_and_text(part)
        segment = {"text": text, "start": -1.0, "end": -1.0, "import_index": index}
        if speaker:
            segment["speaker"] = speaker
        segments.append(segment)
    return segments


def parse_transcript_bytes(filename: str, payload: bytes) -> tuple[list[dict], dict]:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_TRANSCRIPT_EXTENSIONS:
        raise ValueError(f"Unsupported transcript format '{extension}'. Use JSON, SRT, VTT, or TXT.")
    content = payload.decode("utf-8-sig")
    if extension == ".json":
        segments = _parse_json(content)
    elif extension in {".srt", ".vtt"}:
        segments = _parse_captions(content)
    else:
        segments = _parse_text(content)
    if not segments:
        raise ValueError("No transcript segments were found in the uploaded file.")
    timed = sum(segment["start"] >= 0 and segment["end"] >= 0 for segment in segments)
    speakers = sum("speaker" in segment for segment in segments)
    return segments, {
        "segment_count": len(segments),
        "has_timestamps": timed == len(segments),
        "timestamp_coverage": timed / len(segments),
        "has_speakers": speakers > 0,
        "speaker_coverage": speakers / len(segments),
    }
