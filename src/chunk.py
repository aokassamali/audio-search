#Walk segments in order, accumulating into the current chunk.
#Once the chunk reaches a soft target of ~120 words, keep extending until you hit a segment ending in . or ?, then close the chunk. Hard cap ~200 words — if no punctuation shows up by then, cut anyway (transcripts are messy; never trust punctuation to arrive).
#Overlap: start each new chunk with the last segment of the previous chunk.
#Each chunk: id (0, 1, 2...), text (segment texts joined), start, end, and word_count (you'll thank yourself during debugging).
#CLI mirrors yesterday: input path argument, output to data/processed/<stem>_chunks.json.


#plan
#parse JSON
# loop through json and add words to chunk plus extra data
# once reached soft target look for punctuation and then end chunk or hard cap of 200 words and pull end time
#calculate word count and store

import argparse
import json
from datetime import datetime
from pathlib import Path

SOFT_TARGET = 120
HARD_CAP = 200


def create_output_path(transcription: str) -> Path:
    transcription_path = Path(transcription)
    output_directory = Path("data/processed/chunks")
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / f"{transcription_path.stem}_chunks.json"

    if output_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_directory / (
            f"{transcription_path.stem}_chunks_{timestamp}.json"
        )

    return output_path


def build_chunk(segments: list[dict], chunk_id: int) -> dict:
    text = " ".join(segment["text"].strip() for segment in segments)

    return {
        "chunk_id": chunk_id,
        "text": text,
        "start": segments[0]["start"],
        "end": segments[-1]["end"],
        "word_count": len(text.split()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transcription")
    args = parser.parse_args()

    with open(args.transcription, "r", encoding="utf-8") as file:
        data = json.load(file)

    current_segments = []
    current_word_count = 0
    all_chunks = []
    chunk_id = 0
    just_closed_chunk = False

    for segment in data:
        just_closed_chunk = False
        segment_text = segment["text"].strip()

        current_segments.append(segment)
        current_word_count += len(segment_text.split())

        reached_soft_target = (
            current_word_count >= SOFT_TARGET
            and segment_text.endswith((".", "?", "!"))
        )
        reached_hard_cap = current_word_count >= HARD_CAP

        if reached_soft_target or reached_hard_cap:
            all_chunks.append(build_chunk(current_segments, chunk_id))
            chunk_id += 1

            overlap_segment = current_segments[-1]
            current_segments = [overlap_segment]
            current_word_count = len(overlap_segment["text"].split())
            just_closed_chunk = True

    if current_segments and not just_closed_chunk:
        all_chunks.append(build_chunk(current_segments, chunk_id))

    output_path = create_output_path(args.transcription)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(all_chunks, file, indent=2, ensure_ascii=False)

    print(f"Created {len(all_chunks)} chunks.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()