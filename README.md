# audio-search

Natural language search and Q&A over audio. Answers cite timestamps and speaker.

**WER 6.4%**  
**Dense retrieval leads BM25 and RRF**  
**Prosody did not improve speech act classification and failed a shuffled-prosody placebo**

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

Orchestrated as a partitioned Dagster DAG. Served through FastAPI.

---

## Status

v2.3

| | Milestone | |
|---|---|---|
| M1 | Transcription pipeline with faster-whisper int8 | ✅ |
| M2 | Chunking and hybrid retrieval with BM25, dense search, and RRF | ✅ |
| M3 | Retrieval evaluation with recall@5 and MRR | ✅ |
| M4 | Transcription evaluation with WER and error decomposition | ✅ |
| M5 | FastAPI search service | ✅ |
| M6 | Dagster orchestration | ✅ |
| v2.1 | RAG answer layer with enforced citations | ✅ |
| v2.2 | Diarization and speaker role attribution | ✅ |
| v2.3 | Speech act and sincerity experiment | ✅ |

---

## Results

### Transcription

Evaluated against the official transcript after normalizing case, punctuation, and whitespace.

```
WER 6.448%   |   Substitutions 202   Insertions 22   Deletions 603   Correct 12,020
```

- 387 deletion runs. 88% contain one or two words. About one third of deleted tokens are adjacent repetitions.
- The court reporter includes false starts and stage directions. WER overstates genuine omissions. Some real omissions remain, including one about jury trials.
- Proper nouns dominate substitutions. `Liu` accounts for 55 substitutions, about 27%.
- `iii` to `three` is a citation convention mismatch.
- No alias correction was applied. Alias correction would reduce WER from about 6.4% to 5.8%.
- The production fix for repeated proper-noun errors is vocabulary biasing through `initial_prompt`.

### Retrieval

Hand-labeled query set.

| Mode | Recall@5 | MRR |
|---|---:|---:|
| BM25 | 0.375 | 0.524 |
| Dense | **0.417** | **0.650** |
| RRF | 0.358 | 0.543 |

n=10. The result is directional. Separating dense retrieval from RRF needs at least 30 queries.

BM25 is weak on this transcript because lexical overlap is often insufficient. RRF degrades when one retriever is substantially stronger.

Recall is prioritized because missing evidence limits the answer layer.

### Answer layer

n=4. The system refuses absent topics with `out_of_corpus` and unsupported premises with `false_premise`. It answers direct and multi-chunk questions with timestamped citations. This evaluation covers answerability only.

### Speech act and sincerity classification

The experiment tested whether prosody adds signal beyond text. The gold set contains 80 speaker turns labeled from audio. Evaluation uses speaker-grouped folds.

| Arm | Accuracy | Macro F1 |
|---|---:|---:|
| Text-only, permissive labels | 0.662 | 0.608 |
| Text-only, strict labels | 0.738 | 0.609 |
| Prosody-only, eGeMAPS, 91 features | 0.225 | 0.175 |
| Prosody-only, compact, 8 features | 0.325 | 0.288 |
| Text-only stacker | 0.688 | 0.629 |
| Text + prosody stacker | 0.613 | 0.522 |
| Majority baseline | 0.475 | |

The null was not rejected.

Strict labeling raised accuracy from 0.662 to 0.738 while macro F1 stayed flat at about 0.609. The gain came from shifting items into the majority class, not from better classification.

Real and shuffled prosody produced the same permissive accuracy at 0.6375. Strict accuracy was 0.750 with real prosody and 0.7375 with shuffled prosody. Adding prosody to the text stacker produced 0 fixes and 6 regressions.

The 91-feature arm performed worse than the 8-feature arm with 80 labeled turns.

On permissive labels, question F1 was 0.857. Hypothetical F1 was 0.417. Characterization F1 was 0.480. Question syntax was easier to detect. The other two labels required more pragmatic inference.

Strict and permissive labels agreed on 92.5% of turns. Six of eight permissive hypothetical labels changed under the strict definition. Every disagreement occurred on a turn marked hard during annotation.

The prompted models predicted zero jokes and zero hyperboles on the 80 sampled turns. The gold set also contained zero positive examples for those classes, so the rare-class controls were not exercised.

This result applies to one SCOTUS argument, this taxonomy, these features, and n=80. SCOTUS speech is formal and low-affect. That may suppress the prosodic variation needed for this task.

---

## Grounding

1. **Schema constraint.** Citation IDs are restricted to the chunks retrieved for the query.
2. **Prompt rules.** Answers use supplied evidence, cite each claim, refuse unsupported questions, and avoid assigning a position from another speaker's description.
3. **Post-validation.** Returned citations are checked against the retrieved set. Empty or invalid citations downgrade the response to a refusal.

The schema constraint depends on the inference server enforcing the supplied grammar.

### Speaker attribution

Role and identity use separate fields, confidence values, and evidence.

Speaker samples include neighboring-turn context. Neighboring turns can supply names because speakers rarely state their own. A neighboring name is accepted only when the context addresses or introduces the target speaker.

A role or identity without valid evidence is downgraded to `unknown`. Evidence from another speaker is rejected.

Resolution order is manual override → high-confidence identity → high-confidence role → raw speaker ID.

Inferred values remain in the artifact beside the effective label. A manual override can therefore be checked against the inference it replaced.

Output uses `Geiser [SPEAKER_01]`.

The role prompt contains no domain vocabulary. TOML overrides support manual correction.

### Attribution ambiguity calculations

- **`speaker_overlap_ratio`** = winning speaker overlap ÷ total diarized speech in the segment. A value near 1.0 indicates one dominant speaker. A value near 0.5 indicates a split.
- **`diarization_coverage`** = total diarized speech ÷ segment duration. Low coverage with a high ratio indicates a dominant speaker plus non-diarized audio.

82 of about 1,650 segments have an overlap ratio below 0.8. These segments usually span question-answer transitions.

---

## Known limitations

- **Faithfulness.** Citation validity is checked. Entailment is not.
- **Small retrieval and answer evaluations.** Retrieval uses n=10. Answerability uses n=4.
- **Speech act gold set.** One annotator labeled 80 turns. There is no inter-annotator agreement.
- **Rare classes.** The gold set contains zero joke and zero hyperbole examples.
- **Confidence.** Self-reported LLM confidence showed almost no variance. Mean confidence was 0.948 on easy turns and 0.947 on hard turns. Logprob-based confidence was not tested.
- **Single corpus.** The current corpus is formal low-affect speech.
- **Chunk retrieval.** Retrieval matches mentions, not speakers. Prompt rules reduce attribution errors but do not solve retrieval-level ambiguity.
- **Embedding cache.** The cache key hashes chunk `text` only. The field name should be included if the embedded field changes.
- **GPU concurrency.** Concurrent partitions can load multiple models onto shared VRAM.

---

## Design decisions

- No silent CUDA to CPU fallback. CPU execution must be requested.
- No vector database. NumPy brute force is sufficient to about 100,000 vectors.
- int8 is used instead of fp16 because Pascal fp16 throughput is weak.
- Audio is normalized to 16 kHz mono PCM before diarization. MP3 seeking produced inconsistent sample lengths.
- Chunk boundaries use terminal punctuation rather than domain keywords.
- Corpus setup runs once during FastAPI lifespan. Each request embeds the query and ranks stored chunks.

---

## Setup

Python 3.13 is required.

```bash
git clone https://github.com/aokassamali/audio-search
cd audio-search
uv sync
```

Install `ffmpeg` on PATH. `ffplay` is only needed for the interactive gold-set annotation script.

Pyannote diarization requires a Hugging Face token.

```bash
export HF_TOKEN="your_token"
```

PowerShell uses the following form.

```powershell
$env:HF_TOKEN = "your_token"
```

Speaker-role inference and `/answer` require an OpenAI-compatible chat server. Configure it in `audio_search.toml`.

```toml
[llm]
base_url = "http://127.0.0.1:8080"
model = "local"
timeout_seconds = 120
```

`AUDIO_SEARCH_CONFIG`, `AUDIO_SEARCH_SOURCE`, `AUDIO_SEARCH_CHUNK_VARIANT`, `LLAMA_CPP_BASE_URL`, `LLAMA_CPP_MODEL`, and `LLAMA_CPP_TIMEOUT_SECONDS` can override the checked-in configuration.

### Add a source

Place the audio file in `data/raw/`. Add a source entry whose key will also be used as the Dagster partition key.

```toml
[sources.sripetch]
audio_filename = "Sripetch_vs_SEC.mp3"

[sources.sripetch.speaker_labels]
SPEAKER_01 = "Geiser"
```

The source ID defaults to the audio filename without its extension. It can be set explicitly.

```toml
[sources.sripetch]
source_id = "Sripetch_vs_SEC"
audio_filename = "Sripetch_vs_SEC.mp3"
```

Set the default source and chunk behavior under `[app]`.

```toml
[app]
default_source_id = "sripetch"
chunk_variant = "prefer_speaker"
```

`plain` uses plain chunks. `prefer_speaker` uses speaker chunks when present and falls back to plain chunks. `require_speaker` requires speaker chunks.

## Usage

### Materialize the Dagster pipeline

Start Dagster.

```bash
uv run dagster dev -f src/dagster_assets.py
```

Open `http://127.0.0.1:3000`.

Add a dynamic partition under `audio_files` using the source key from `audio_search.toml`. For the example source, the partition key is `sripetch`.

Select the assets through `embeddings` and materialize the partition. The full path is shown below.

```
raw_audio
    ↓
normalized_audio
    ├─> transcript ─> chunks
    └─> diarization
             ↓
     speaker_transcript
             ↓
       speaker_roles
             ↓
      speaker_chunks
             ↓
        embeddings
```

The LLM server must be running before materializing `speaker_roles`.

### Start the API

Materialize at least one source, then start FastAPI.

```bash
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Check the loaded corpus.

```bash
curl http://127.0.0.1:8000/health
```

Ask a grounded question.

```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"How do the petitioner and government disagree about the purpose and limits of disgorgement?","source_keys":["sripetch"],"retrieval_mode":"global","top_k":10}'
```

Abridged response.

```json
{
  "answerable": true,
  "answer": "Geiser argues that disgorgement must be a remedial remedy that restores funds to the proper owner or injured parties and cannot serve as punishment or deterrence, while the government argues that disgorgement can focus on depriving wrongdoers and deterring misconduct without requiring restoration to investors.",
  "citations": [
    {
      "citation_id": "Sripetch_vs_SEC:1",
      "source_id": "Sripetch_vs_SEC",
      "chunk_id": 1,
      "start": 45,
      "end": 92
    },
    {
      "citation_id": "Sripetch_vs_SEC:80",
      "source_id": "Sripetch_vs_SEC",
      "chunk_id": 80,
      "start": 3420,
      "end": 3463
    }
  ]
}
```

---

## Roadmap

1. **Learned speech representations.** Compare wav2vec2 and HuBERT against the same gold set. Test frozen representations and a fine-tuned variant.
2. **Technical writeup.** Document the system, ablation, label perturbation result, confidence result, and limitations.

Public artifacts use government audio with clear provenance.
