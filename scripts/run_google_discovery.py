from __future__ import annotations

import argparse
from pathlib import Path

from app.settings import Settings
from discovery.google.google_cse_client import GoogleCSEClient
from storage.cse_cache import CSECache
from storage.candidate_store import CandidateStore
from discovery.google.query_generator import QueryGenerator
from discovery.google.google_cse import GoogleCSEDiscovery, DiscoveryConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dotenv", default=".env")
    ap.add_argument("--dry-run", action="store_true", help="Only generate queries to test in browser")
    ap.add_argument("--export-queries", default="", help="Write generated queries to txt file")
    ap.add_argument("--max-queries", type=int, default=30)
    ap.add_argument("--pages", type=int, default=1, help="Pages per query (each page up to 10 results)")
    args = ap.parse_args()

    settings = Settings.from_env(args.dotenv)

    gen = QueryGenerator()
    queries = gen.generate(include_tags=[
        "kitchen_en", "kitchen_ar",
        "fitout_en", "fitout_ar",
        "architect_en", "architect_ar",
    ])

    if args.export_queries:
        QueryGenerator.export_to_txt(queries, path=args.export_queries)
        print(f"Wrote {len(queries)} queries to {args.export_queries}")

    if args.dry_run:
        # Print a sample to test manually in browser
        for i, qs in enumerate(queries[: args.max_queries], start=1):
            print(f"{i:03d} | {qs.tag:12s} | {qs.city:8s} | {qs.q}")
        return

    client = GoogleCSEClient(settings=settings)
    cache = CSECache(settings.cache_dir)
    store = CandidateStore(settings.cache_dir)

    discovery = GoogleCSEDiscovery(client=client, cache=cache, store=store)
    discovery.run(
        queries=queries,
        cfg=DiscoveryConfig(pages_per_query=args.pages, num_per_page=10, max_queries=args.max_queries),
    )

    print(f"Done. Candidate domains: {len(store.all_domains())}")
    print(f"Cache dir: {settings.cache_dir}")


if __name__ == "__main__":
    main()
