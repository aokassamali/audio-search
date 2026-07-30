audio-search

Natural language search and grounded question answering over long-form audio. Answers cite timestamps and speakers.

Word error rate 6.4%Dense retrieval led BM25 and RRF on the current evaluation setHandcrafted prosody did not improve speech act classificationCanonical speaker grouping eliminated the initial WavLM advantage over shuffled embeddings

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

The production pipeline is orchestrated as a partitioned Dagster DAG and served through FastAPI. The repository also contains controlled speech act experiments with handcrafted acoustic features and frozen WavLM representations.

Status

v2.4

Version

Milestone

Status

M1

Transcription pipeline with faster-whisper int8

Complete

M2

Chunking and hybrid retrieval with BM25, dense search, and RRF

Complete

M3

Retrieval evaluation with recall@5 and MRR

Complete

M4

Transcription evaluation with WER and error decomposition

Complete

M5

FastAPI search service

Complete

M6

Dagster orchestration

Complete

v2.1

RAG answer layer with enforced citations

Complete

v2.2

Diarization and speaker role attribution

Complete

v2.3

Handcrafted prosody experiment

Complete

v2.4

Learned speech representations and cross-case speaker audit

Complete

Results

Transcription

Evaluation used the official transcript after normalizing case, punctuation, and whitespace.

WER 6.448%   |   Substitutions 202   Insertions 22   Deletions 603   Correct 12,020

The evaluation found 387 deletion runs. Eighty-eight percent contain one or two words. About one third of deleted tokens are adjacent repetitions.

The court reporter includes false starts and stage directions. WER therefore overstates genuine omissions. Some real omissions remain, including one about jury trials.

Proper nouns dominate substitutions. Liu accounts for 55 substitutions, about 27 percent.

iii to three is a citation convention mismatch.

No alias correction was applied. Alias correction would reduce WER from about 6.4 percent to 5.8 percent.

Repeated proper-noun errors can be reduced through vocabulary biasing with initial_prompt.

Retrieval

The retrieval evaluation uses 10 hand-labeled queries.

Mode

Recall@5

MRR

BM25

0.375

0.524

Dense

0.417

0.650

RRF

0.358

0.543

The result is directional because the evaluation set is small. Dense retrieval led both metrics. RRF underperformed dense retrieval because the weaker lexical ranking still affected the fused order.

Recall is prioritized because missing evidence limits the answer layer.

Answer layer

The answerability evaluation uses four questions. The system refuses absent topics with out_of_corpus and unsupported premises with false_premise. It answers direct and multi-chunk questions with timestamped citations. The evaluation covers answerability and citation validity. It does not measure entailment.

Handcrafted prosody

The v2.3 experiment tested whether handcrafted prosodic features add speech act signal beyond text. The gold set contains 80 speaker turns from one Supreme Court argument. One annotator labeled each turn from audio. Evaluation used speaker-grouped folds.

Arm

Accuracy

Macro F1

Text-only, permissive labels

0.662

0.608

Text-only, strict labels

0.738

0.609

Prosody-only, eGeMAPS, 91 features

0.225

0.175

Prosody-only, compact, 8 features

0.325

0.288

Text-only stacker

0.688

0.629

Text plus prosody stacker

0.613

0.522

Majority baseline

0.475



The experiment did not find evidence that the tested prosodic features improved classification.

Strict labeling raised accuracy from 0.662 to 0.738 while macro F1 remained near 0.609. The accuracy gain came from moving items into common classes.

Real and shuffled prosody produced the same permissive accuracy at 0.6375. Strict accuracy was 0.750 with real prosody and 0.7375 with shuffled prosody. Adding prosody to the text stacker produced no fixes and six regressions.

The 91-feature model performed worse than the compact 8-feature model with 80 labeled turns. Self-reported LLM confidence also failed to distinguish easy turns from hard turns.

Strict and permissive labels agreed on 92.5 percent of turns. Six of eight permissive hypothetical labels changed under the strict definition. Every disagreement occurred on a turn marked hard.

The sample contained no joke or hyperbole examples. Those classes were excluded from the primary v2.4 task.

Learned speech representations

The v2.4 experiment expanded the speech act gold set to 400 turns from four Supreme Court arguments. Each case contributed 100 randomly sampled turns. The original 80 labels were frozen and 320 additional turns were annotated.

Permissive labels were used for primary modeling because the strict taxonomy contained only four hypothetical examples.

Label

Count

Assertion

227

Question

140

Characterization

20

Hypothetical

13

The set contains 287 easy turns and 113 hard turns. Strict and permissive labels agree on 97.5 percent of turns.

WavLM Base+ produced one 768-dimensional mean-pooled representation for the encoder input and each of 12 transformer layers. Evaluation used four-fold stratified group cross-validation. Logistic regression regularization was selected inside each training fold.

The first split treated each case-specific diarization ID as a separate speaker. That evaluation produced a best macro F1 of 0.457 at layer 11. A within-speaker shuffled control produced an empirical p-value of 0.038.

Recurring speakers appeared in several cases under different diarization IDs. A cross-case review confirmed 23 same-voice pair decisions and produced eight merged identities. The final mapping contains 28 canonical speaker groups. It has no same-case collisions and no conflicts with different-voice decisions.

The canonical split changed the result.

Representation

Accuracy

Macro F1

Majority baseline

0.568



MiniLM text embeddings

0.487

0.253

Word and character TF-IDF

0.588

0.300

WavLM best layer with provisional speaker groups

0.740

0.457

WavLM best layer with canonical speaker groups

0.700

0.393

TF-IDF plus WavLM layer 10

0.690

0.366

The canonical WavLM sweep selected layer 10 with macro F1 0.393. The mean best-layer score under 100 within-speaker shuffled permutations was 0.395. The empirical p-value was 0.505. A comparison restricted to the selected layer produced a shuffled mean of 0.360 and an empirical p-value of 0.079.

The best-layer control accounts for searching across 13 representations. Under that comparison, the canonical WavLM result was consistent with the shuffled control.

Fusion improved the observed TF-IDF score from 0.300 to 0.366 macro F1. Accuracy increased from 0.588 to 0.690. The fusion model made 90 fixes, 49 regressions, and 18 changes between incorrect labels. Performance on hard turns increased from 0.203 to 0.312 macro F1.

The matched-audio fusion score was 0.366. The mean score after shuffling WavLM turns within canonical speakers was 0.349 with a standard deviation of 0.016. The empirical p-value was 0.158. The residual benefit from correct audio-text alignment was therefore inconclusive.

The fusion model learned the two common classes.

Label

F1

Support

Assertion

0.766

227

Question

0.696

140

Characterization

0.000

20

Hypothetical

0.000

13

The final result does not support a reliable turn-specific benefit from the tested frozen audio representations. The initial positive WavLM result depended in part on recurring speaker identity. The speaker audit was required to expose that leakage.

Evaluation design

Gold labels

Each turn has a permissive label, a strict label, a difficulty label, and optional annotation notes. The primary v2.4 task uses the dominant function of the whole turn.

Conditional language is necessary but insufficient for a hypothetical label. A conditional setup followed by a substantive request is generally labeled as a question. Ambiguous turns are marked hard.

The notes support review and error analysis. They are not model inputs.

Speaker grouping

Diarization IDs are local to each audio file. The same justice can therefore receive a different raw ID in every case.

Cross-case candidates were ranked from early-layer WavLM speaker centroids. High-similarity pairs were reviewed by listening to representative clips. Confirmed matches were collapsed into transitive canonical groups. Model evaluation uses those canonical groups.

A false merge can place unrelated voices in one group. A missed merge can leave the same person in both training and test data. The review favored precision when the identity was uncertain.

Placebo controls

The WavLM placebo shuffles full turn embeddings within canonical speakers. This preserves speaker identity and each speaker's embedding distribution while breaking the match between a turn and its audio representation.

The fusion placebo keeps text fixed and shuffles WavLM turns within canonical speakers. This tests whether any fusion gain depends on the correct audio-text pairing.

Grounding

Schema constraint. Citation IDs are restricted to chunks retrieved for the query.

Prompt rules. Answers use supplied evidence, cite each claim, refuse unsupported questions, and avoid assigning a position from another speaker's description.

Post-validation. Returned citations are checked against the retrieved set. Empty or invalid citations downgrade the response to a refusal.

The schema constraint depends on the inference server enforcing the supplied grammar.

Speaker attribution

Role and identity use separate fields, confidence values, and evidence.

Speaker samples include neighboring-turn context. A neighboring name is accepted only when the context addresses or introduces the target speaker.

A role or identity without valid evidence is downgraded to unknown. Evidence from another speaker is rejected.

Resolution follows manual override, high-confidence identity, high-confidence role, and raw speaker ID in that order.

Inferred values remain in the artifact beside the effective label. A manual override can therefore be checked against the inference it replaced.

Output uses Geiser [SPEAKER_01].

The role prompt contains no domain vocabulary. TOML overrides support manual correction.

Attribution ambiguity calculations

speaker_overlap_ratio equals winning speaker overlap divided by total diarized speech in the segment. A value near 1.0 indicates one dominant speaker. A value near 0.5 indicates a split.

diarization_coverage equals total diarized speech divided by segment duration. Low coverage with a high ratio indicates a dominant speaker plus non-diarized audio.

Eighty-two of about 1,650 segments have an overlap ratio below 0.8. These segments usually span question-answer transitions.

Known limitations

Faithfulness. Citation validity is checked. Entailment is not.

Small retrieval and answer evaluations. Retrieval uses 10 queries. Answerability uses four questions.

Single annotator. One annotator labeled all 400 speech act turns. There is no inter-annotator agreement estimate.

Class imbalance. The primary set contains 20 characterizations and 13 hypotheticals. Every final text and fusion model produced zero F1 for both classes.

Dominant-function labels. Many turns contain several pragmatic functions. A single label discards that internal structure.

Restricted domain. The speech act corpus contains four Supreme Court arguments. Courtroom speech is formal and often low-affect.

Speaker review coverage. The cross-case audit reviewed high-similarity candidate pairs. Lower-similarity recurring speakers may remain unmerged.

Layer selection. The best WavLM layer was selected from the same four outer folds used for descriptive reporting. The best-layer shuffled control reduces this bias. There is no independent held-out test set.

Frozen representations. WavLM was not fine-tuned. Mean pooling may discard local timing information.

Confidence. Self-reported LLM confidence showed almost no variance in v2.3. Logprob-based confidence was not tested.

Chunk retrieval. Retrieval matches mentions rather than speakers. Prompt rules reduce attribution errors but do not solve retrieval-level ambiguity.

Embedding cache. The production cache key hashes chunk text only. The embedded field name should be included if that field changes.

GPU concurrency. Concurrent partitions can load several models onto shared VRAM.

Design decisions

CPU execution must be requested explicitly. The pipeline does not silently fall back from CUDA.

NumPy brute-force search is sufficient for the current corpus and remains practical to about 100,000 vectors.

int8 transcription is used because Pascal GPUs have weak fp16 throughput.

Audio is normalized to 16 kHz mono PCM before diarization. MP3 seeking produced inconsistent sample lengths.

Chunk boundaries use terminal punctuation rather than domain keywords.

Corpus setup runs once during FastAPI lifespan. Each request embeds the query and ranks stored chunks.

Four-fold grouped cross-validation was used because a stable fixed test set would contain very few hypothetical examples.

Permissive labels were primary because the strict taxonomy reduced the hypothetical class to four examples.

Cross-case speaker identities were reviewed before final model comparison.

Shuffled controls preserve speaker identity while breaking turn-level alignment.

Setup

Python 3.13 is required.

git clone https://github.com/aokassamali/audio-search
cd audio-search
uv sync

Install ffmpeg on PATH. ffplay is needed only for interactive annotation and speaker review.

Pyannote diarization requires a Hugging Face token.

export HF_TOKEN="your_token"

PowerShell uses the following form.

$env:HF_TOKEN = "your_token"

Speaker-role inference and /answer require an OpenAI-compatible chat server. Configure it in audio_search.toml.

[llm]
base_url = "http://127.0.0.1:8080"
model = "local"
timeout_seconds = 120

AUDIO_SEARCH_CONFIG, AUDIO_SEARCH_SOURCE, AUDIO_SEARCH_CHUNK_VARIANT, LLAMA_CPP_BASE_URL, LLAMA_CPP_MODEL, and LLAMA_CPP_TIMEOUT_SECONDS can override the checked-in configuration.

Add a source

Place the audio file in data/raw/. Add a source entry. The source key also serves as the Dagster partition key.

[sources.sripetch]
audio_filename = "Sripetch_vs_SEC.mp3"

[sources.sripetch.speaker_labels]
SPEAKER_01 = "Geiser"

The source ID defaults to the audio filename without its extension. It can be set explicitly.

[sources.sripetch]
source_id = "Sripetch_vs_SEC"
audio_filename = "Sripetch_vs_SEC.mp3"

Set the default source and chunk behavior under [app].

[app]
default_source_id = "sripetch"
chunk_variant = "prefer_speaker"

plain uses plain chunks. prefer_speaker uses speaker chunks when present and falls back to plain chunks. require_speaker requires speaker chunks.

Usage

Materialize the Dagster pipeline

Start Dagster.

uv run dagster dev -f src/dagster_assets.py

Open http://127.0.0.1:3000.

Add a dynamic partition under audio_files using the source key from audio_search.toml. For the example source, the partition key is sripetch.

Select the assets through embeddings and materialize the partition.

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

The LLM server must be running before materializing speaker_roles.

Start the API

Materialize at least one source, then start FastAPI.

uv run uvicorn src.api:app --host 127.0.0.1 --port 8000

Check the loaded corpus.

curl http://127.0.0.1:8000/health

Ask a grounded question.

curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"How do the petitioner and government disagree about the purpose and limits of disgorgement?","source_keys":["sripetch"],"retrieval_mode":"global","top_k":10}'

Abridged response.

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