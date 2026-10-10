# DARPAN — A Digital Mentor for Students

A warm, context-aware student mentor built with FastAPI, LangGraph, Groq (`groq` SDK), ChromaDB, BGE embeddings, PyMuPDF, and a responsive HTML/CSS/JavaScript frontend.

## Run from a fresh terminal

Use Python 3.10 or newer. The current machine's default Python 3.9 can import and run the local endpoints, but it is end-of-life and emits dependency warnings; use Python 3.10+ for a fresh setup.

```bash
cd /path/to/mentor
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Configure Groq. Preserve an existing `.env`; copy the template only if you do not have one yet:

```bash
if [ ! -f .env ]; then cp .env.example .env; fi
```

Set `GROQ_API_KEY` in `.env`. If a key has been shared in chat, revoke it and use a newly rotated key. `SARVAM_API_KEY` enables question audio and speech transcription; keep both keys private and never commit `.env`. You can change `GROQ_MODEL` there to another active Groq model.

## Build the source index

Put searchable PDFs in `data/pdfs/`. The indexer accepts only filenames matching these titles:

- `Complete Works Vol. 1.pdf` through `Complete Works Vol. 9.pdf` (or a combined Complete Works PDF)
- `Inspired Talks.pdf`
- `Reminiscences of Swami Vivekananda.pdf`
- `Vivekananda A Biography.pdf`

Then run the idempotent ingestion command:

```bash
python -m backend.rag_ingest
```

The matching extracted pages and page-linked chunks are stored under `data/processed/` and are used only while their source PDF checksum matches. New PDFs are extracted with PyMuPDF. Retrieval combines BGE semantic search with BM25 keyword matching, preserving student concern words such as fear, failure, focus, and purpose. Chunks retain source, volume/heading when available, and PDF page. Re-ingestion replaces the indexed chunks. The combined `data/pdfs/Complete Works of Swami Vivekananda.pdf` is already in this project. Image-only scanned PDFs need OCR before ingestion. On first use, BGE downloads its model from Hugging Face; it is cached locally afterward.

## Start the app

```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000>. Health is available at <http://localhost:8000/api/health>; interactive API docs are at <http://localhost:8000/docs>. Tailwind and web fonts load from CDNs.

The browser sends the conversation and session memory with every turn. Viveka listens through six questions, then uses retrieved evidence for its reflection and next-step guidance. Saving the final reflection writes the established conversation summary to `data/sessions/<uuid>.json`. Unknown summary fields remain empty/null. Groq and RAG errors are reported without exposing internal paths or API keys.

At the end of a conversation, the student can save a reflection flower in **My Reflections**. Opening a flower shows the student's reflection, next step, source passage and saved conversation. The app uses `yt_suggester.py` to search for explicit, solution-oriented Shorts for a normalized topic and plays them in embedded players; availability depends on YouTube search and embed permissions. Sarvam voice support records a short WAV clip in the browser, transcribes it on the server, and speaks each mentor question. Microphone access works on localhost and HTTPS. Speech input is capped at 12 seconds for responsive transcription.

## Environment variables

- `GROQ_API_KEY` — required for chat; keep it private in `.env`
- `GROQ_MODEL` — model name; default `openai/gpt-oss-120b`
- `DATA_PATH` — default `data`
- `PDF_PATH` — default `data/pdfs`
- `CHROMA_PATH` — default `data/chroma`
- `SESSION_PATH` — default `data/sessions`
- `CORS_ORIGINS` — comma-separated allowed browser origins; defaults to localhost:8000 and 127.0.0.1:8000

Relative paths resolve from the project root. `.env`, PDFs, Chroma data, and saved sessions are git-ignored.

## API

- `GET /api/health` — liveness check
- `GET /api/status` — Groq configuration and indexed-chunk count
- `POST /api/chat` — a turn with session ID, conversation history, and context
- `POST /api/session/end` — save the session JSON
- `POST /api/knowledge/ingest` — ingest matching PDFs
