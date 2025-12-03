# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REST API for classifying English text as "правда" (truth), "неправда" (falsehood), or "нейтрально" (neutral) using NLI (Natural Language Inference) and Wikipedia evidence verification.

## Core Architecture

The system is a **stateless FastAPI application** with the following pipeline:

1. **Claim Extraction** (`app/services/claim_extractor.py`) - Breaks input text into factual statements
2. **Evidence Retrieval** (`app/services/evidence_retriever.py`) - FAISS vector search against Wikipedia KB
3. **NLI Verification** (`app/services/nli_verifier.py`) - roberta-large-mnli scores claim-evidence entailment
4. **Classification** (`app/services/classifier.py`) - Aggregates scores and applies thresholds

### Key Components

- **ModelManager** (`app/core/models.py`) - Singleton pattern for model lifecycle management
  - Loads models once at startup, reuses across requests
  - Manages: SentenceTransformer (embeddings), HuggingFace pipeline (NLI), FAISS index, KB snippets
  - CRITICAL: Models must be loaded before first request or endpoints will fail

- **Configuration** (`app/core/config.py`) - pydantic-settings with .env support
  - Thresholds: `TRUTH_THRESHOLD` (default 0.85), `FALSEHOOD_THRESHOLD` (default 0.4)
  - Retrieval: `TOP_K_PROOFS` (default 6), `MAX_CLAIMS` (default 8)

- **Classification Logic** (`app/services/classifier.py:51-126`)
  - Per-claim: support >= 0.85 → "правда", < 0.4 → "неправда", else "нейтрально"
  - Overall: ANY "неправда" → overall "неправда" (pessimistic aggregation)

## Common Commands

### First-time Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Build Wikipedia Knowledge Base (required before first run)
python scripts/build_kb.py
```

### Running the API
```bash
# Quick start (handles setup + run)
./run.sh

# Manual start
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Classify text
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Albert Einstein was born in 1879."}'
```

## Important Details

### Knowledge Base
- Located at `data/faiss_index/wikipedia.index` + `data/kb_snippets.json`
- Built by `scripts/build_kb.py` - scrapes ~18 Wikipedia topics, ~265 snippets
- Uses FAISS IndexFlatL2 for L2 distance search
- Must exist before starting server (run.sh auto-builds if missing)

### Model Loading
- All models loaded synchronously at startup via `@app.on_event("startup")` in `app/main.py:22-34`
- First request after startup will be slow (~5-10s) while models initialize
- Subsequent requests fast (models cached in memory)

### Project Structure
```
app/
├── main.py              # FastAPI app, startup/shutdown hooks
├── api/
│   ├── routes.py        # /classify and /health endpoints
│   └── schemas.py       # Pydantic request/response models
├── core/
│   ├── config.py        # Settings (thresholds, paths, model names)
│   └── models.py        # ModelManager singleton
├── services/            # Business logic (claim extraction, retrieval, NLI, classification)
└── utils/
    └── wikipedia_kb.py  # Wikipedia scraping utilities
```

## Historical Context

This is a university project for "Технологии проектирования и сопровождения информационных систем" (Information Systems Design and Maintenance Technologies).

**Migration Note**: The project was migrated from a PostgreSQL-based architecture (see `models.psql` in git history) to a stateless API. The database schema (`users`, `requests`, `responses`, `claims`, `proofs`, `tickets`) is no longer active but remains in version control for reference.
