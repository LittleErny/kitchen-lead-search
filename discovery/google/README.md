# Discovery: Google Search (CSE)

This module performs **Google Search** discovery using the Google Custom Search API (CSE). It is a search-engine based source, not a listings database.

## What it does
- Runs a set of search queries (e.g., "kitchen showroom Riyadh").
- Fetches paginated results (10 results per page).
- Normalizes and filters domains to keep only likely company sites.
- Stores candidate domains with hit metadata (query, rank, title, snippet).

## Key components
- `GoogleCSEClient`: API client wrapper (`discovery/google/google_cse_client.py`).
- `GoogleCSEDiscovery`: orchestrates queries, pagination, caching, and candidate collection (`discovery/google/google_cse.py`).
- `QueryGenerator`: builds default query variants (`discovery/google/query_generator.py`).

## Storage integration (explicit)
- `CSECache` lives in `storage/cse_cache.py` (not in this module).
- `CandidateStore` lives in `storage/candidate_store.py` (not in this module).

## How pagination works
- `DiscoveryConfig.pages_per_query` controls how many result pages are fetched per query.
- Each page is 10 results, so `pages_per_query=3` yields up to ~30 results per query.
- The pipeline exposes this via `--pages` in `scripts/run_pipeline.py`.

## Output
Candidates are stored by domain in `.cache/google_cse/` by `storage/CandidateStore` and later consumed by the pipeline.

## Note on future sources
This is **Google Search** only. A separate **Google Maps** discovery module is (expected?) later, but is not implemented yet.
