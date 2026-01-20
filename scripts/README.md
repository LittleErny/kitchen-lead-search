# `run_pipeline.py` Pipeline Overview (Step-by-Step)

This doc explains what the pipeline does **from a Google search result to a saved lead**,
in plain, step-by-step terms. It focuses on what happens **before evaluation**;
the evaluator is only mentioned briefly.

## 1) Google discovery returns raw results
- Entry point: `scripts/run_pipeline.py` (with `--run-google`) calls Google CSE.
- The API returns items with `link`, `title`, `snippet` (JSON).
- We take each `link` (URL) as a candidate.

## 2) Normalize domain + filter obvious non-company sites
- Code: `storage/candidate_store.py`
- The URL is normalized to a domain (strip `www`, lowercase).
- Domains like Facebook/LinkedIn/etc. are filtered out.

## 3) Store candidate hits on disk (JSON)
- Code: `storage/candidate_store.py`
- We persist candidates as:
  - `data/candidates/candidates.jsonl` (append-only)
  - `data/candidates/candidates_index.json` (compact index)
- Each candidate stores `domain`, `example_url`, and `hits[]`
  (query, rank, url, title, snippet).

## 4) Pick a candidate domain to crawl
- Code: `scripts/run_pipeline.py`
- For each domain we choose an `example_url` (typically the Google link).

## 5) Fetch the homepage HTML
- Code: `crawling/fetcher.py`, `crawling/crawler.py`
- The crawler downloads the homepage and keeps the **raw HTML** as `home_html`.
- If the homepage fails, the domain is skipped.

## 6) Extract internal links from the homepage
- Code: `crawling/crawler.py`
- A lightweight HTML parser extracts `<a>` links and their anchor text.
- Only internal links (same domain) are kept.

## 7) Pick a few priority pages to crawl
- Code: `crawling/crawler.py`
- Links are bucketed by keywords into:
  `about`, `contact`, `services`, `projects`.
- We pick up to one best link per bucket.

## 8) Fetch those pages and build aggregated text
- Code: `crawling/crawler.py`
- Each selected page is downloaded as HTML.
- HTML is converted to **plain text** (tags/scripts/styles removed).
- All page texts are concatenated into a single `aggregated_text` string.

## 9) Pass HTML + text to the evaluator
- Code: `scripts/run_pipeline.py`
- We call the evaluator with:
  - `html = home_html` (homepage HTML for name/title/JSON-LD extraction)
  - `text = aggregated_text` (multi-page text for scoring)
- The evaluator returns classification + score + extracted fields.

## 10) Save the lead to CSV/JSON
- Code: `storage/leads_store.py`, `storage/csv_export.py`
- The final record is saved with fields like:
  `name`, `domain`, `website`, `email`, `phone`, `score`, `relevant`, `confidence`.
- Outputs are stored in `data/leads/` (CSV + JSON index).

