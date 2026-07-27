# audio-search

Search and Q&A over conversational audio with **speaker-attributed, citation-grounded answers**.

Ask a natural-language question about an audio file and get an answer that cites specific timestamps, attributed to the speaker who actually said it.

```
                    ┌─> transcript (faster-whisper) ─┐
audio ─> normalize ─┤                                ├─> speaker transcript
                    └─> diarization (pyannote) ──────┘          │
                                                                ▼
                                                         speaker roles (LLM)
                                                                │
                                                                ▼
                                                        chunks ─> embeddings
                                                                │
                    query ──────────────────────────────────────┤
                                                                ▼
                                            hybrid retrieval (BM25 + dense, RRF)
                                                                │
                                                                ▼
                                              grounded answer + timestamp citations
```

Orchestrated as a partitioned Dagster DAG. Served via FastAPI.

---

## Status

**v2.2 — shipped.** Diarization and speaker role attribution.

| | Milestone | |
|---|---|---|
| M1 | Transcription pipeline (faster-whisper, int8) | ✅ |
| M2 | Chunking + hybrid retrieval (BM25 + dense, RRF) | ✅ |
| M3 | Retrieval evaluation (recall@5, MRR) | ✅ |
| M4 | Transcription evaluation (WER + error decomposition) | ✅ |
| M5 | FastAPI search service | ✅ |
| M6 | Dagster orchestration | ✅ |
| v2.1 | RAG answer layer with enforced citations | ✅ |
| v2.2 | Diarization + speaker role attribution | ✅ |
| v2.3 | Prosodic register classification | in design |

Demo corpus is US Supreme Court oral argument. Nothing in the pipeline is court-specific.

---

## Evaluation

Measurements below.

### Transcription — 6.4% WER

Against the official transcript, with case, punctuation, and whitespace normalized on both sides.

```
WER 6.448%   |   Substitutions 202   Insertions 22   Deletions 603   Correct 12,020
```

**Raw WER overstates the substantive error rate.** Word-level alignment:

- 387 deletion runs, 88% of them 1–2 words; only 8 runs exceeded 5 consecutive words.
- Roughly a third of deleted words are adjacent repetitions ("the the", "that that").
- Cause: the court reporter transcribes **verbatim**. Includes false starts, stutters, filler, stage directions. Whisper silently cleans these up.

Most deletions are a convention mismatch, not lost content. Few genuine omissions, including a substantive phrase about jury trials.

**Substitutions are almost entirely one addressable class.** Proper nouns and domain vocabulary: a single case name accounts for ~27% of all substitutions on its own (Liu → lew/lieu/lue, 55 occurrences). Formal legal citation style rendered phonetically (`iii` → `three`).

`initial_prompt` vocabulary biasing seeds the decoder with domain terms and proper nouns would help. No alias-correction step is applied here, because the goal is agnosticism to audiofile. Correcting the homophones would move 6.4% to roughly 5.8%.

### Retrieval — dense led fusion

Hand-labeled query set, recall@5 and MRR:

| Mode | Recall@5 | MRR |
|---|---|---|
| BM25 | 0.375 | 0.524 |
| Dense | **0.417** | **0.650** |
| RRF | 0.358 | 0.543 |

**Caveat :~10 queries.** More hand-labled queries can increase n for confidence, but for this particular section of the project I didn't expect that much value.

 BM25 is weak here because paraphrase-style queries against homophone-riddled ASR output doesn't work well. Lexical matching needs term overlap that the transcript doesn't reliably contain.

Recall is prioritized over precision, because retrieval feeds an LLM. Recall sets the ceiling for the entire pipeline.

### Answer layer — smoke test

System correctly refuses when the topic is absent from the corpus (`out_of_corpus`) and when a question's premise is unsupported (`false_premise`), and answers direct and multi-chunk questions with citations.

---

## Grounding

Citations are enforced in depth rather than requested in the prompt.

1. **Schema constraint.** The answer schema restricts citation IDs to an enum of the chunks actually retrieved for this query.
2. **Prompt rules.** Evidence-only, a citation per claim, refuse when unanswerable, and don't attribute a position to a party.
3. **Post-validation.** Every returned citation is re-checked against the retrieved set. Empty or invalid citations downgrade the response to a refusal.

### Speaker attribution

Role and identity are modeled as **separate claims with separate evidence**. 

Prompt only accepts a neighboring name when the context shows it's being used to *address* the target.

Evidence enforcement is a code-level invariant. Resolution order is manual override → high-confidence identity → high-confidence role → raw speaker ID, and the inferred values are retained alongside the effective label so an override masking a bad inference stays visible.

Rendered output keeps provenance.

### Attribution ambiguity is measured

Two diagnostics per transcript segment:

- **`speaker_overlap_ratio`** = winning speaker's overlap ÷ total diarized speech in the segment. 

- **`diarization_coverage`** = total diarized speech ÷ segment duration

**82 of ~1,650 segments (5%) have an overlap ratio below 0.8** 

---

## Known limitations

- **Faithfulness** The system verifies that a cited chunk is real and was retrieved. It does not verify that the claim is *entailed by* the chunk it cites.
- **Evaluation is small.** ~10 retrieval queries, n=4 answerability.
- **Retrieval finds mentions, not speakers, at the chunk level.** 
- **GPU concurrency is capped.** Materializing multiple Dagster partitions at once puts multiple Whisper models in contention for the same VRAM. Concurrency limits are set so partitions queue rather than collide.
- **Single corpus.** Everything is validated on formal, low-affect speech with an official transcript. Generalization to other registers is untested.

---

## Setup

<!-- TODO: fill in from your actual environment -->

```bash
git clone https://github.com/aokassamali/audio-search
cd audio-search
uv sync
```

Configuration lives in `audio_search.toml`, with environment-variable overrides layered on top.

Requires `ffmpeg` on PATH. Diarization requires a HuggingFace token with access to the pyannote models.

## Usage

<!-- TODO: once v2.3 is completed-->

---

## Roadmap

**v2.3 — prosodic register classification** . Whether prosody adds signal beyond text for classifying speech register: assertion, hypothetical, question, characterization, hyperbole, joke.

Public-domain government audio only.