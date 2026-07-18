# audio-search

Search and Q&A over conversational audio. Ingests audio files, transcribes them into a timestamped corpus (faster-whisper), and will answer natural-language questions about the content via hybrid retrieval + LLM synthesis.

**Pipeline:** audio → Whisper transcription → chunking/embedding → hybrid retrieval (BM25 + dense) → FastAPI endpoint. Orchestration via Dagster.

**v0.1 does not include:** tonal analysis, UI, auth, multi-file scale, speaker diarization, hosting.

**Milestones:** 1) Transcription pipeline ✅ · 2) Retrieval + Q&A via FastAPI — chunking ✅, retrieval next · 3) Dagster orchestration of ingest.

**Eval:** transcription validated against official SCOTUS oral argument transcripts (WER eval in progress).

