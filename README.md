# Lead Discovery Agent (MVP)

An automated pipeline that discovers, filters, and stores publicly available leads for the Saudi kitchen market. The system focuses on B2B leads (showrooms, interior studios, fit‑out contractors, developers, architects) and can be extended for B2C signals later.

This README is intentionally focused on the **single production entrypoint**: `scripts/run_pipeline.py`. Other scripts are development helpers only.

## Requirements Coverage (from the technical task)

Functional requirements:
- Discover leads from public sources: **Google Search (CSE)** queries.
- Extract and normalize lead info: contacts, category, type, score, source metadata.
- Classify leads: B2B/B2C + category.
- Deduplicate across runs: domain‑keyed candidate store and lead store upsert.
- Persist leads: JSON index + CSV export.
- Re‑runnable without duplicates: idempotent storage with stable lead IDs.

Non‑functional notes:
- Uses public data only (no logins).
- File‑based storage (no DB required).
- Designed for readability and extensibility.

## How to Run (single entrypoint)

1) Create `.env` with Google CSE credentials:

```bash
GOOGLE_CSE_API_KEY=...
GOOGLE_CSE_CX=...
```

2) Run discovery + evaluation:

```bash
python -m scripts.run_pipeline --run-google --max-queries 10 --pages 1
```

3) Outputs:
- JSON leads index: `data/leads/leads_index.json`
- CSV export: `data/leads/leads.csv`

## HTTP Service (n8n Runner MVP)

This repo also includes a FastAPI service to run long discovery jobs asynchronously and page results.

Required env:

```bash
API_TOKEN=your-secret-token
GOOGLE_CSE_API_KEY=...
GOOGLE_CSE_CX=...
```

Optional:

```bash
DATABASE_URL=sqlite:///./.data/lead_discovery.db
DEBUG=false
PORT=8000
EVALUATOR_MODE=precision
```

Run locally:

```bash
uvicorn service.main:app --host 0.0.0.0 --port 8000
```

DigitalOcean App Platform command:

```bash
uvicorn service.main:app --host 0.0.0.0 --port $PORT
```

Endpoints:
- `GET /health`
- `POST /runs`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/results`
- `POST /runs/{run_id}/cancel`

`POST /runs` supports an evaluator mode override in the request body:

```json
{
  "sources": ["google"],
  "queries": ["kitchen showroom riyadh", "modular kitchen jeddah"],
  "evaluator_mode": "recall"
}
```

Notes:
- `evaluator_mode` accepts `precision` or `recall`.
- If omitted, the service falls back to `EVALUATOR_MODE`, then defaults to `precision`.

### Optional: evaluate specific URLs

```bash
python -m scripts.run_pipeline --urls https://example.com https://example2.com --max-domains 2
```

Or from a file:

```bash
python -m scripts.run_pipeline --urls-file urls.txt --max-domains 50
```

## Sources Used

- **Google Search (CSE)**: query‑driven discovery of public websites.
- Other sources can be added under `discovery/` (e.g., directories or Maps later).

## Deduplication Logic

- **Discovery dedup**: candidate domains are normalized and stored once in the `CandidateStore`.
- **Evaluation dedup**: `LeadsStore` upserts by domain‑based lead ID.
- Re‑running the pipeline updates existing leads instead of duplicating them.

## Data Schema

Each lead record includes:
- `lead_id`
- `lead_type` (B2B/B2C)
- `category` (showroom/fit‑out/designer/architect/unknown)
- `name` (currently domain; can be improved later)
- `country`, `city`
- `email`, `phone`, `website`
- `source`
- `relevance_score`, `relevant`
- `discovered_at`

## Example Output

Example JSON record (shortened):

```json
{
  "lead_id": "nadco.com.sa",
  "lead_type": "B2B",
  "category": "showroom",
  "name": "nadco.com.sa",
  "country": "Saudi Arabia",
  "city": "riyadh",
  "email": "info@example.com",
  "phone": "+9665XXXXXXX",
  "website": "nadco.com.sa",
  "source": "google_cse|query=...|rank=...|url=...",
  "relevance_score": 82,
  "relevant": true,
  "discovered_at": "2025-01-01T12:00:00Z"
}
```

## Assumptions

- MVP focuses on KSA‑targeted leads; KSA signals are required to pass gates.
- B2B is the primary target; B2C classifieds are not prioritized.
- Current focus is on businesses with their own websites.
- Google CSE results are treated as public sources.
- The current pipeline can store some non-target domains (banks, gov services, marketplaces) because filtering is heuristic and records may still be saved when they meet `min_score` even if `relevant=False`.
- `BLOCKED_DOMAINS` only blocks exact domains and does not cover subdomains (for example, `sa.linkedin.com`).
- Social-only businesses (LinkedIn/Facebook-only presence) are intentionally out of scope for now; they would need a separate evaluator because those platforms have different page structures.

## Project Layout (high level)

- `discovery/` – discovery sources (Google Search now; more later)
- `crawling/` – fetch + crawl + aggregated text
- `evaluation/` – scoring, classification, and reasons/signals
- `storage/` – caches and lead persistence (JSON/CSV)
- `scripts/` – CLI entrypoints (production uses `run_pipeline.py`)
- `data/` – output artifacts
