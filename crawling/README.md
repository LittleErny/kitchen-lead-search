# Crawling (Site Fetch + Aggregation)

This module is responsible for fetching website content and aggregating text across a small set of pages. It is intentionally lightweight and avoids heavy browser automation.

## Purpose
- Fetch HTML for a given URL with caching and polite delays.
- Extract links and crawl a small number of pages per domain.
- Aggregate plain text into one blob for downstream evaluation.

## Key components
- `CachedFetcher`: HTTP client with disk cache and rate limiting.
- `SiteCrawler`: single-domain crawler that collects up to `max_pages` and returns:
  - `aggregated_text`
  - `home_html`
  - `visited_urls`
  - `failed_urls`

## How it is used in the pipeline
- `scripts/run_pipeline.py` calls `SiteCrawler.collect(url)`.
- The crawler returns an aggregate text blob.
- The evaluator receives `evaluate(url, html=home_html, text=aggregated_text)`.

## Behavior notes
- URL paths are not included in `aggregated_text` to avoid false keyword triggers.
- The crawler skips some binary file types (e.g., PDF) when collecting pages.

## Output fields (from `SiteCrawler.collect`)
- `aggregated_text: str`
- `home_html: str`
- `visited_urls: List[str]`
- `failed_urls: List[Tuple[str, str]]` (url, reason)

## Debugging tips
- If `aggregated_text` is empty, the evaluator will mark the site as `empty_content`.
- Check `failed_urls` for 404/blocked pages.
- Large or JS-heavy sites may return minimal text without a browser.
