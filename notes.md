# Project Notes / Learning Log

## 2026-07-16 — Milestone 1: Transcription pipeline

**Shipped:** `src/transcribe.py` — audio file → faster-whisper (GPU, int8) → timestamped segment JSON in `data/processed/`. Validated on a full SCOTUS oral argument (~71 min audio, ~5 min transcription, ~14x realtime on a GTX 1080 Ti).

**Technical learnings:**
- `model.transcribe()` returns a lazy generator — transcription happens as segments are consumed, not at the call site. Explains the silent terminal during processing.
- Pascal GPUs (1080 Ti) have crippled fp16 throughput; fp32 beats fp16 on this card, but int8 (via CTranslate2 / DP4A) is faster still and halves memory. Chose int8 with a `--compute-type` escape hatch.
- `Path.stem` = filename minus final extension, nothing more. Collision-handling (timestamp suffix on existing output) is my code's behavior layered on top — keeping library behavior vs. my behavior straight matters for debugging.
- `mkdir(parents=True, exist_ok=True)` = create intermediate dirs, no error on re-run. Makes the function idempotent.

**Process learnings:**
- AI-generated code shifts work from writing to verifying. Caught a real bug in AI-drafted code: output-writing block dedented to module level → `NameError` on `args` (out of scope). Verification requires the same understanding writing does.
- Design decision I argued for and kept: no silent CUDA→CPU fallback. Silently commandeering the CPU for an hour is worse than failing loudly with a clear `--device cpu` suggestion. Fail loud > fall back quiet when the fallback is expensive.

**Findings from the data:**
- Whisper transcribes phonetically, so proper nouns are the weak point: *Liu v. SEC* rendered as "Liu," "lieu," and "Lue" within one transcript. Downstream impact: keyword search (BM25) will miss inconsistent spellings — motivates the WER eval and eventually vocabulary prompting / normalization.

**TODO:**
- WER eval vs. official SCOTUS transcript (`data/reference/`) using `jiwer`; needs PDF text extraction + normalization strategy for the proper-noun problem.
- Progress bar via `tqdm` (denominator: `info.duration` vs. segment `end` timestamps).
- CUDA loud-failure except block around model load. *(← noting this: it's still not in the committed script)*


## 2026-07-17 — Milestone 2a: Chunking

**Shipped:** `src/chunk.py` — transcript JSON → retrieval-sized chunks (soft target 120 words extended to sentence boundary, hard cap 200, single-segment overlap between consecutive chunks). 98 chunks from the SCOTUS argument, verified by eyeball: sentence-boundary endings, visible overlap, overlapping time ranges as expected.

**Design decisions:**
- Chunk boundaries keyed on terminal punctuation (`.` `?`), not domain keywords ("Your Honor") — pipeline stays corpus-agnostic.
- No silent behavior on punctuation-free transcripts: chunking degrades to pure hard-cap sizing, which is the contract