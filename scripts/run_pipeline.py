from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.settings import Settings

from discovery.google.google_cse_client import GoogleCSEClient
from storage.cse_cache import CSECache
from storage.candidate_store import CandidateStore
from discovery.google.google_cse import DiscoveryConfig
from discovery.google.google_cse import GoogleCSEDiscovery
from discovery.google.query_generator import QueryGenerator, QuerySpec

from crawling.fetcher import CachedFetcher
from crawling.crawler import SiteCrawler
from evaluation.evaluator import SiteEvaluator

from storage.leads_store import LeadsStore, LeadRecord, utc_now_iso, lead_id_from_domain
from storage.csv_export import export_leads_csv

# Exclude obviously non-business-website domains
BLOCKED_DOMAINS = {
    "facebook.com", "www.facebook.com",
    "instagram.com", "www.instagram.com",
    "linkedin.com", "www.linkedin.com",
    "youtube.com", "www.youtube.com",
    "maps.google.com",
}


def ensure_http(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return "https://" + url


def load_queries_from_file(path: str) -> Optional[List[QuerySpec]]:
    p = Path(path)
    if not p.exists():
        return None
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        return None
    return [QuerySpec(q=ln, tag="file", city="") for ln in lines]

def load_urls_from_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    return lines

def pick_source(candidate: Dict[str, Any]) -> str:
    """
    Required field: Source (where the lead was discovered).
    We'll store a compact text like:
      google_cse|query=...|rank=...|url=...
    """
    hits = candidate.get("hits") or []
    if not hits:
        return "google_cse"
    h0 = hits[0]
    q = (h0.get("query") or "").replace("\n", " ")
    rank = h0.get("rank")
    url = h0.get("url")
    return f"google_cse|query={q}|rank={rank}|url={url}"


def infer_country_city(text: str, evaluator: SiteEvaluator) -> Tuple[str, str]:
    """
    MVP heuristic: if KSA signal or any known KSA city appears -> Saudi Arabia + first matched city.
    """
    norm = evaluator._normalize(text or "")
    city = "unknown"
    for c in evaluator.cities_ksa:
        if evaluator._normalize(c) in norm:
            city = c
            break
    country = "Saudi Arabia" if (
                "saudi" in norm or "ksa" in norm or "السعود" in norm or city != "unknown") else "unknown"
    return country, city


async def evaluate_one(
        domain: str,
        candidate: Dict[str, Any],
        crawler: SiteCrawler,
        evaluator: SiteEvaluator,
        leads: LeadsStore,
        min_score: int,
        sem: asyncio.Semaphore,
) -> None:
    async with sem:
        if domain in BLOCKED_DOMAINS:
            return

        example_url = ensure_http(candidate.get("example_url") or domain)
        if not example_url:
            return

        # Run sync crawler/evaluator in a thread (async MVP)
        agg = await asyncio.to_thread(crawler.collect, example_url)
        text = getattr(agg, "aggregated_text", "") or ""
        home_html = getattr(agg, "home_html", "") or ""
        ev = await asyncio.to_thread(evaluator.evaluate, example_url, home_html, text)

        # if ev.relevance_score < min_score:
        #     return

        country, city = infer_country_city(text, evaluator)

        emails = (ev.contacts or {}).get("emails") or []
        phones = (ev.contacts or {}).get("phones") or []
        whatsapp = (ev.contacts or {}).get("whatsapp") or []

        email = emails[0] if emails else ""
        phone = phones[0] if phones else ""

        discovered_at = candidate.get("first_seen_at") or utc_now_iso()

        rec = LeadRecord(
            lead_id=lead_id_from_domain(domain),
            lead_type=ev.lead_type,
            category=ev.category,
            name=ev.company_name or domain,
            country=country,
            city=city,
            email=email,
            phone=phone,
            website=domain,
            source=pick_source(candidate),
            relevance_score=ev.relevance_score,
            relevant=ev.relevant,
            discovered_at=discovered_at,
            domain=domain,
            confidence=ev.confidence,
        )
        # print("Evaluation object:", ev)
        # print("Reasons:", ev.reasons)
        # print("Signals:", ev.signals)
        # print()

        leads.upsert(rec)


async def run_async_pipeline(
        candidates: Dict[str, Any],
        max_domains: int,
        concurrency: int,
        min_score: int,
        leads_store: LeadsStore,
) -> None:
    # Website evaluation components
    fetcher = CachedFetcher(cache_dir=".cache/http", min_delay=0.3, max_delay=1.0)
    crawler = SiteCrawler(fetcher=fetcher, max_pages=5)
    evaluator = SiteEvaluator()

    sem = asyncio.Semaphore(concurrency)

    tasks: List[asyncio.Task] = []
    count = 0

    for domain, cand in candidates.items():
        if count >= max_domains:
            break
        count += 1
        tasks.append(asyncio.create_task(
            evaluate_one(domain, cand, crawler, evaluator, leads_store, min_score, sem)
        ))

    await asyncio.gather(*tasks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-google", action="store_true", help="Run Google discovery before evaluation")
    ap.add_argument("--max-queries", type=int, default=10)
    ap.add_argument("--pages", type=int, default=1)

    ap.add_argument("--queries-file", type=str, default="google_queries.txt",
                    help="If exists, use these queries (one per line) instead of generator")

    ap.add_argument("--candidates-index", type=str, default=".cache/google_cse/candidates_index.json")
    ap.add_argument("--leads-index", type=str, default="data/leads/leads_index.json")
    ap.add_argument("--csv-out", type=str, default="data/leads/leads.csv")

    ap.add_argument("--max-domains", type=int, default=150)
    ap.add_argument("--min-score", type=int, default=45)

    ap.add_argument("--concurrency", type=int, default=6,
                    help="How many sites to evaluate in parallel (threaded via asyncio.to_thread)")

    ap.add_argument(
        "--urls",
        nargs="*",
        default=None,
        help="Manual URLs to evaluate (skip Google + skip candidates index). Example: --urls https://a.com https://b.com"
    )
    ap.add_argument(
        "--urls-file",
        type=str,
        default=None,
        help="Path to a text file with URLs (one per line). Alternative to --urls."
    )


    args = ap.parse_args()

    settings = Settings.from_env(".env")

    # 1) Run Google discovery (optional)
    if args.run_google:
        # Build components used by your existing discovery code
        client = GoogleCSEClient(settings=settings)
        cache = CSECache(cache_dir=Path(".cache/google_cse"))
        store = CandidateStore(out_dir=Path(".cache/google_cse"))

        discovery = GoogleCSEDiscovery(client=client, cache=cache, store=store)

        # Queries: prefer file if exists, else generator
        qs = load_queries_from_file(args.queries_file)
        if qs is None:
            gen = QueryGenerator()
            qs = gen.generate()

        qs = qs[: args.max_queries]

        # Run discovery (uses cache => rerun should cost ~0)
        cfg = DiscoveryConfig(
            pages_per_query=args.pages,
            num_per_page=10,
            max_queries=args.max_queries,
        )
        discovery.run(qs, cfg)
        store.save()

    # 2) Load candidates
    candidates: Dict[str, Any] = {}

    # Manual URLs mode: skip Google discovery + skip candidates index
    manual_urls: List[str] = []
    if args.urls:
        manual_urls.extend(args.urls)

    if args.urls_file:
        manual_urls.extend(load_urls_from_file(args.urls_file))

    # Normalize + deduplicate while preserving order
    if manual_urls:
        seen = set()
        norm_urls: List[str] = []
        for u in manual_urls:
            u2 = ensure_http(u)
            if not u2:
                continue
            if u2 in seen:
                continue
            seen.add(u2)
            norm_urls.append(u2)

        # Build candidates dict compatible with evaluate_one()
        now = utc_now_iso()
        for i, url in enumerate(norm_urls):
            # domain key should match how the rest of the pipeline expects it
            # (candidate store uses domain as key)
            domain = url.replace("https://", "").replace("http://", "").split("/")[0].lower()
            if domain.startswith("www."):
                domain = domain[4:]
            candidates[domain] = {
                "example_url": url,
                "hits": [{"query": "manual_urls", "rank": i + 1, "url": url}],
                "first_seen_at": now,
            }

        # Optional: override max-domains so you don't accidentally truncate the manual list
        args.max_domains = min(args.max_domains, len(candidates))

    else:
        # Default mode: load candidates from index (produced by discovery)
        candidates_path = Path(args.candidates_index)
        if not candidates_path.exists():
            raise SystemExit(
                f"Candidates index not found: {args.candidates_index}. "
                f"Run with --run-google first OR pass --urls / --urls-file."
            )
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))

    # 3) Evaluate sites concurrently, store leads
    leads = LeadsStore(args.leads_index)
    asyncio.run(run_async_pipeline(
        candidates=candidates,
        max_domains=args.max_domains,
        concurrency=args.concurrency,
        min_score=args.min_score,
        leads_store=leads,
    ))


    # 4) Persist + CSV export
    leads.save()
    export_leads_csv(leads.by_domain, args.csv_out)

    print(f"[OK] Leads stored: {len(leads.by_domain)}")
    print(f"[OK] JSON index: {args.leads_index}")
    print(f"[OK] CSV: {args.csv_out}")


if __name__ == "__main__":
    # PLEASE READ README FOR INSTRUCTIONS TO RUN
    main()
