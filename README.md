# Ask-a-Podcast

A retrieval-augmented question answering system for podcasts. Subscribe to any show's RSS feed, and it transcribes new episodes, indexes them for semantic search, and answers questions with citations that link to the exact moment something was said, with an in-page audio player that seeks straight to that timestamp.

**Live:** [ask-a-podcast-ui.vercel.app](https://ask-a-podcast-ui.vercel.app) · Backend: [ask-a-podcast-production.up.railway.app](https://ask-a-podcast-production.up.railway.app)

## What it does

- Subscribe to any podcast by pasting its RSS feed URL. New episodes are downloaded, transcribed, and indexed automatically.
- Ask natural language questions across all subscribed shows, or scope a question to one specific podcast.
- Every answer comes with citations. Click one and the audio player jumps to that exact second.
- A brand new subscription automatically backfills its full history in the background. Existing subscriptions get checked daily for new episodes, and the search index rebuilds hourly.

## How it works

```
RSS feed -> download -> transcribe (faster-whisper, word-level timestamps)
    -> chunk into overlapping, token-aware windows
    -> embed (sentence-transformers) -> store in pgvector

Question -> LangGraph pipeline:
    decompose (split if needed) -> retrieve (vector + BM25 search, fused with RRF)
    -> grade (is the context good enough?) -> rewrite and retry, or generate answer
    -> stream answer + citations to the frontend
```

## Stack

Backend: FastAPI, PostgreSQL with pgvector, faster-whisper, sentence-transformers, rank-bm25, LangGraph, Claude (Haiku), APScheduler, OpenTelemetry, RAGAS, slowapi for rate limiting.

Frontend: Next.js (App Router), TypeScript, Tailwind.

Containerized with Docker. Deployed on Railway (backend + Postgres) and Vercel (frontend).

## Running it locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Postgres with pgvector, via Docker:

```bash
docker run -d --name podcast-pg \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=podcasts \
  -p 5432:5432 ankane/pgvector
docker exec -it podcast-pg psql -U postgres -d podcasts -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Copy `.env.example` to `.env` and fill in your database credentials and Anthropic API key.

```bash
python3 setup_db.py
uvicorn main:app --reload
```

In a separate terminal:

```bash
cd ask-a-podcast-ui
npm run dev
```

Backend on `http://localhost:8000`, frontend on `http://localhost:3000`.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/ask` | POST | Answer a question. Streams NDJSON: one citations event, then a sequence of token events. Rate limited, 10/minute. |
| `/podcasts` | GET | List subscribed podcasts with episode counts. |
| `/podcasts` | POST | Subscribe to a feed. Streams sync progress. If brand new, automatically backfills the full history in the background afterward. Rate limited, 5/hour. |
| `/podcasts` | DELETE | Unsubscribe. Episodes stay in the database and remain searchable, this is a soft delete. Rate limited, 10/hour. |
| `/podcasts/backfill` | POST | Manually trigger a full backfill, in case you want it outside the automatic flow above. |

## Engineering notes

A few real problems came up during development that are worth documenting, since they shaped design decisions that aren't obvious from the code alone.

**Reciprocal Rank Fusion was being computed correctly, then thrown away.** Fused chunk IDs were collected into a Python set, which has no ordering, then queried back from Postgres with a plain `WHERE id IN (...)`, which also returns rows in no guaranteed order. The ranking that RRF worked out never actually reached the language model. Fixed by tracking order explicitly through the whole pipeline. On the evaluation suite, this raised average context precision from 0.52 to 0.72, and fixed one specific query that had gone from 0.25 to 1.0.

**A crash partway through ingestion could permanently disable an episode.** An episode's row was written to the database before its transcript chunks were guaranteed to exist. If the process crashed in between, which happened, the episode ended up with a row but no chunks: unsearchable, and never retried, because the deduplication logic saw the row and assumed the episode was already handled. Fixed by committing the episode's row and its chunks together, as one atomic unit, per episode.

**Running Whisper and the embedding model in the same process caused hard crashes.** CTranslate2 (used by faster-whisper) and torch's OpenMP thread pool collided, producing a segfault. Fixed by running transcription in its own subprocess. A second, similar looking crash turned out to have a different cause: a transitively installed newer NumPy silently broke torch's pinned binary compatibility. Fixed by pinning `numpy<2` explicitly in `requirements.txt`.

**Subscribing to a large podcast tried to process its entire back catalog at once.** One show had over 1,700 episodes in its feed. Fixed with a per-sync cap of 3 new episodes per call, and an automatic background backfill triggered for any brand new subscription, so the full history still arrives without manual intervention, it just takes real time in the background rather than blocking the first request.

**A production database starts genuinely empty, in ways local development never tests.** Two separate bugs only surfaced on first deploy: BM25 crashed with a division by zero when the `chunks` table had zero rows, since it had never been tested against a truly empty index locally. Separately, downloading an episode failed outright because the `episodes/` and `transcripts/` directories didn't exist at all in a fresh container, they'd simply always existed locally since early in development and nothing ever had to create them. Both fixed by handling the empty/missing case explicitly rather than assuming prior state.

## Evaluation

A 20 question RAGAS suite covers three different podcast transcripts, including questions with no answer in the corpus, to check that the system refuses rather than guesses, and one question that names a real public figure who was not actually the one discussed on the show, to check that the model does not substitute a familiar name for the correct one.

Results on answerable questions: faithfulness 0.95, answer relevancy 0.91, context precision 0.73, context recall 0.95. Unanswerable questions correctly refuse to answer about 95% of the time.

One limitation surfaced by the eval suite and left unresolved on purpose: the retrieval graph's rewrite step, which reformulates a question when the first attempt at context looks insufficient, can shift what the question is actually asking. One test question, "she's about to turn 60," was rewritten into a search for her birth year, which does not match how the fact was actually phrased in the source audio. A partial fix, keeping the original question as a fallback search alongside the rewritten one, measurably improved recall but did not fully fix precision. This is documented as an open problem rather than hidden.

## Known limitations

- No authentication. Rate limiting and an Anthropic spending cap are the current mitigations against abuse of a public URL, not a hard guarantee.
- A very large backlog (well over 1,700 episodes) still takes real wall-clock time to fully backfill, since transcription is genuinely CPU-bound work, not something the automatic backfill makes instant.
- Speaker diarization is a possible addition.