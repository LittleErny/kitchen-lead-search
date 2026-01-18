# Lead Discovery Agent (MVP)

A minimal pipeline that discovers candidate companies via Google Custom Search, crawls their sites, evaluates relevance, and stores leads in JSON/CSV.

## Structure

- `app/` – application settings and shared configuration
- `discovery/` – Google CSE discovery client and query generation
- `crawling/` – site fetching and HTML crawling
- `evaluation/` – scoring and categorization logic
- `storage/` – caches, stores, and CSV export
- `scripts/` – CLI entrypoints and debug scripts
- `data/` – output artifacts (leads index and CSV)

## Quick Start

1) Create `.env` with:

```
GOOGLE_CSE_API_KEY=...
GOOGLE_CSE_CX=...
```

2) Run discovery + evaluation:

```
python scripts/run_pipeline.py --run-google
```

3) Exported leads:

- `data/leads/leads_index.json`
- `data/leads/leads.csv`

## Common Scripts

- `python scripts/run_google_discovery.py` – run discovery only
- `python scripts/run_pipeline.py` – full pipeline
- `python scripts/debug_one_query.py` – debug one CSE query

## Notes

- All docs and code comments are in English.
- This is an MVP structure optimized for clarity and fast iteration.
