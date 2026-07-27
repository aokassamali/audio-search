import json
import pdfplumber
import argparse
import re
import string
import jiwer
from pathlib import Path

from src.config import load_settings


def load_whisper_text(transcript_path):
    with open(transcript_path, "r", encoding="utf-8") as file:
        segments = json.load(file)

    joined_text = " ".join(
        segment["text"].strip()
        for segment in segments
    )

    return joined_text


def clean_reference_text(text):
    cleaned_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        # Remove document structure that was not spoken.
        if re.fullmatch(
            r"P\s+R\s+O\s+C\s+E\s+E\s+D\s+I\s+N\s+G\s+S",
            line,
            flags=re.IGNORECASE,
        ):
            continue

        if re.fullmatch(
            r"\(\d{1,2}:\d{2}\s*[ap]\.m\.\)",
            line,
            flags=re.IGNORECASE,
        ):
            continue

        if line.startswith("ORAL ARGUMENT OF "):
            continue

        if line.startswith("REBUTTAL ARGUMENT OF "):
            continue

        if line.startswith("ON BEHALF OF THE "):
            continue

        if line.startswith("(Whereupon"):
            continue

        # Remove a speaker label but keep anything spoken after the colon.
        line = re.sub(
            r"^(?:"
            r"THE CHIEF JUSTICE|"
            r"CHIEF JUSTICE [A-Z'-]+|"
            r"JUSTICE [A-Z'-]+|"
            r"MR\. [A-Z'-]+|"
            r"MS\. [A-Z'-]+|"
            r"MRS\. [A-Z'-]+"
            r"):\s*",
            "",
            line,
        )

        if line:
            cleaned_lines.append(line)

    return " ".join(cleaned_lines)


def extract_reference_text(pdf_path, start_page=3, end_page=84):
    page_texts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[start_page:end_page]:
            cropped_page = page.crop(
                (
                    page.width * 0.23,   # Remove line-number column
                    page.height * 0.10, # Remove header and page number
                    page.width * 0.90,  # Keep transcript body
                    page.height * 0.94, # Remove footer
                )
            )

            page_text = cropped_page.extract_text(
                x_tolerance=2,
                y_tolerance=3,
            ) or ""

            page_texts.append(page_text)

    return "\n".join(page_texts)


def normalize_text(text):
    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = " ".join(text.split())

    return text


def compute_wer(reference_text, hypothesis_text):
    result = jiwer.process_words(
        reference_text,
        hypothesis_text,
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript")
    parser.add_argument("reference")
    parser.add_argument(
    "--output",
    type=Path,
    default=None,
)
    args = parser.parse_args()

    settings = load_settings()

    output_path = (
        args.output
        or settings.paths.eval_dir
        / "wer_alignment.txt"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    whisper_text = load_whisper_text(args.transcript)
    reference_text = extract_reference_text(args.reference)
    reference_text = clean_reference_text(reference_text)

    normalized_whisper = normalize_text(whisper_text)
    normalized_reference = normalize_text(reference_text)

    result = compute_wer(
        normalized_reference,
        normalized_whisper,
    )

    alignment_text = jiwer.visualize_alignment(
        result,
        show_measures=False,
        skip_correct=False,
        line_width=120,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(alignment_text)

    print(f"Saved alignment to {output_path}")
    print("\nTranscription Evaluation")
    print(f"WER: {result.wer:.3%}")
    print(f"Substitutions: {result.substitutions}")
    print(f"Insertions: {result.insertions}")
    print(f"Deletions: {result.deletions}")
    print(f"Correct words: {result.hits}")

    print(
    "\nMost frequent errors:"
)

    print(
        jiwer.visualize_error_counts(
            result,
            top_k=10,
        )
    )

if __name__ == "__main__":
    main()
    