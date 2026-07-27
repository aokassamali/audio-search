# audio-search

Natural language search and Q&A over audio. Answers cite timestamps and speaker.

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

v2.2

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

---

## Results

### Transcription

Against the official transcript, with case, punctuation, and whitespace normalized on both sides.

```
WER 6.448%   |   Substitutions 202   Insertions 22   Deletions 603   Correct 12,020
```

- 387 deletion runs, 88% of 1-2 words. ~1/3 of deleted tokens are adjacent repetitions.
- Court reporter transcribes verbatim, including false starts and stage directions. WER overstates genuine omissions. Some exist, including one about jury trials.
- Substitutions are almost all proper nouns. Liu (lew/lieu/lue) = 55, ~27% of substitutions.
- `iii` → `three` is a citation convention mismatch, not an error.
- No alias correction applied. With it, 6.4% → ~5.8%.
- Fix for the proper-noun class is `initial_prompt` vocabulary biasing.

### Retrieval

Hand-labeled query set, recall@5 and MRR:

| Mode | Recall@5 | MRR |
|---|---|---|
| BM25 | 0.375 | 0.524 |
| Dense | **0.417** | **0.650** |
| RRF | 0.358 | 0.543 |

n=10. Directional only. Separating dense from RRF needs 30+ queries.

BM25 is weak because lexical matching doesn't work well on this sort of transcript. RRF degrades when one retriever dominates.

Recall prioritized over precision. The LLM tolerates an irrelevant chunk, not a missing one.

### Answer layer

n=4. Refuses when the topic is absent from the corpus (`out_of_corpus`) and when a question's premise is unsupported (`false_premise`). Answers direct and multi-chunk questions with citations. Answerability only.

---

## Grounding

1. **Schema constraint.** The answer schema restricts citation IDs to an enum of the chunks actually retrieved for this query.
2. **Prompt rules.** Evidence-only, a citation per claim, refuse when unanswerable, and don't attribute a position to a party on the basis of another speaker describing it.
3. **Post-validation.** Every returned citation is re-checked against the retrieved set. Empty or invalid citations downgrade the response to a refusal.

Layer 1 binds only if the inference server enforces the supplied grammar.

### Speaker attribution

Role and identity are separate fields with separate confidence and separate evidence.

Speaker samples plus neighboring-turn context are passed to the LLM. Neighboring turns supply names, since speakers do not state their own. A neighboring name is accepted only when context shows it addressing the target.

Role or identity returned without a valid evidence sample is downgraded to `unknown`. Evidence belonging to a different speaker is rejected.

Resolution order is manual override → high-confidence identity → high-confidence role → raw speaker ID.

Inferred values are kept alongside the effective label, so an override masking a bad inference is visible.

Output format is `Geiser [SPEAKER_01]`.

Role prompt contains no domain vocabulary. TOML overrides handle manual correction.

### Attribution ambiguity calcs

- **`speaker_overlap_ratio`** = winning speaker's overlap ÷ total diarized speech in the segment. Near 1.0 = one speaker owns it. Near 0.5 = split.
- **`diarization_coverage`** = total diarized speech ÷ segment duration. Low coverage with high ratio = dominant speaker, non-diarized audio in segment.

**82 of ~1,650 segments (5%) have an overlap ratio below 0.8.** These segments span a speaker change. Speaker changes occur at question-answer transitions, so they cluster there rather than distributing evenly.

---

## Known limitations

- **Faithfulness.** Citations are checked for validity, not entailment. Requires a separate NLI or judge step.
- **Small eval set.** n=10 retrieval, n=4 answerability.
- **Chunk retrieval matches mentions, not speakers.** Rule 8 is a stopgap.
- **Embedding cache key hashes chunk `text` only.** Correct while `text` is the embedded field. Fix is to include the field name in the key.
- **GPU concurrency capped.** Concurrent partitions load multiple Whisper models against shared VRAM.
- **Single corpus.** Formal low-affect speech.

---

## Design decisions

- No silent CUDA → CPU fallback. Fails with a `--device cpu` suggestion.
- No vector DB. NumPy brute force to ~10⁵ vectors.
- int8 over fp16. Pascal fp16 throughput is degraded. `--compute-type` overrides.
- Audio normalized to 16 kHz mono PCM before diarization. Compressed MP3 seeking gave inconsistent sample-length errors.
- Chunk boundaries key on terminal punctuation, not domain keywords.
- Corpus prep amortized in FastAPI `lifespan`. Per request: embed query, rank.

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

<!-- TODO -->

---

## Roadmap

**v2.3 — prosodic register classification.** Whether prosody adds signal beyond text for classifying speech register: assertion, hypothetical, question, characterization, hyperbole, joke.

Three-arm ablation: text-only, prosody-only, text+prosody. eGeMAPS features. Labels from audio. Reported per class.

Label perturbation: strict vs permissive taxonomy variants over the same turns, measuring agreement.

Public-domain government audio only.