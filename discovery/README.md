# Discovery (Lead Sources)

This package contains all discovery methods that generate candidate websites for evaluation. Each method is responsible for producing a list of domain candidates with basic metadata (query, rank, source, etc.).

## Purpose
- Provide multiple discovery channels for candidate sites.
- Keep discovery logic separate from crawling, evaluation, and storage.
- Allow new sources to be added without touching the scoring pipeline.

## Current sources
- `google/`: Google Search Engine (CSE) discovery.

## Storage boundary (important)
- This package does **not** implement persistence itself.
- Discovery uses storage helpers from `storage/` (e.g., `storage/cse_cache.py`, `storage/candidate_store.py`).

## Future sources (planned)
- Additional search engines or directories.
- Google Maps (business listings) as a separate source (not implemented yet).
- Catalogues (such as arablocal.com or sa.muqawlat.com)
