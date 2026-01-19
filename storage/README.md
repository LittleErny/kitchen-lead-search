# Storage (Caches + Persistence)

This package contains persistence utilities used by discovery and pipeline stages. It is intentionally simple and file-based (JSON + CSV).

## Purpose
- Cache external API responses (Google CSE).
- Store candidate discovery hits keyed by domain.
- Store evaluated leads and export them to CSV.

## Components
- `cse_cache.py`: disk cache for Google CSE requests/responses.
- `candidate_store.py`: domain-centric store for discovery hits.
- `leads_store.py`: JSON-based lead store with upsert semantics.
- `csv_export.py`: exports lead records to CSV.

## Usage in pipeline
- Discovery writes candidates to `.cache/google_cse/` via `CandidateStore`.
- Evaluation writes leads to `data/leads/` via `LeadsStore`.
- CSV export is written to `data/leads/leads.csv`.

## Data locations (default)
- CSE cache: `.cache/google_cse/`
- Candidates index: `.cache/google_cse/candidates_index.json`
- Leads index: `data/leads/leads_index.json`
- Leads CSV: `data/leads/leads.csv`

## Notes
- Storage modules are shared by multiple packages and should stay dependency-light.
- The storage layer is not responsible for crawling or evaluation logic.

