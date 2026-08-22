# Audio Search

Natural language search and grounded question answering over long-form audio. Answers cite timestamps and speakers.

> Demo frontend work lives on the `demo-frontend` branch. The browser can submit raw audio as a real Dagster materialization, inspect live job state, import existing transcripts, query the corpus, and audit cited chunks.

## Demo startup

Start the API and frontend:

```bash
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` for the product UI and `http://127.0.0.1:8000/docs` for FastAPI.

Browser-triggered raw-audio jobs use a shared local Dagster instance under `data/dagster_home` unless `DAGSTER_HOME` is already set. To inspect those same runs in Dagster UI, point the shell at that directory before starting Dagster.

PowerShell:

```powershell
$env:DAGSTER_HOME = "$PWD\data\dagster_home"
uv run dagster dev -f src/dagster_assets.py
```

Then open `http://127.0.0.1:3000`.

Raw-audio ingestion defaults to one concurrent top-level job on a machine. Override after benchmarking with:

```powershell
$env:AUDIO_SEARCH_INGEST_WORKERS = "2"
```

Each audio file is still a separate Dagster partition/run. Increasing this value can increase throughput on suitable hardware, but it also creates concurrent GPU model workloads.

---

## Architecture

```text
audio
  ↓
normalize
  ├──> transcript ──> chunks
  └──> diarization
          ↓
    speaker transcript
          ↓
      speaker roles
          ↓
     speaker chunks
          ↓
       embeddings
          ↓
 hybrid retrieval
          ↓
 grounded answer + timestamp citations
```

The production pipeline is a partitioned Dagster asset graph served through FastAPI. Existing transcript files can bypass speech recognition and enter at the text/chunking path in the demo frontend.

---

## Grounding

1. Citation IDs are schema-restricted to chunks retrieved for the query.
2. Answers use supplied evidence, cite claims, refuse unsupported questions, and avoid speaker misattribution.
3. Returned citations are post-validated; empty or invalid citations downgrade to a refusal.

---

## Setup

Python 3.13 is required.

```bash
git clone https://github.com/aokassamali/audio-search
cd audio-search
uv sync
```

Install `ffmpeg` on PATH. Pyannote diarization requires a Hugging Face token.

PowerShell:

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

### Add a configured source

Place the audio file in `data/raw/` and add a source entry:

```toml
[sources.sripetch]
audio_filename = "Sripetch_vs_SEC.mp3"

[sources.sripetch.speaker_labels]
SPEAKER_01 = "Geiser"
```

Set the default source and chunk behavior under `[app]`.

```toml
[app]
default_source_id = "sripetch"
chunk_variant = "prefer_speaker"
```

`plain` uses plain chunks. `prefer_speaker` uses speaker chunks when present and falls back to plain chunks. `require_speaker` requires speaker chunks.

---

## Manual Dagster materialization

```bash
uv run dagster dev -f src/dagster_assets.py
```

Open `http://127.0.0.1:3000`, add the source key as a dynamic partition under `audio_files`, select the assets through `embeddings`, and materialize the partition.

---

## API

```bash
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Check the loaded corpus:

```bash
curl http://127.0.0.1:8000/health
```

Ask a grounded question:

```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"How do the petitioner and government disagree about the purpose and limits of disgorgement?","source_keys":["sripetch"],"retrieval_mode":"global","top_k":10}'
```

Public artifacts use United States government audio with clear provenance.
