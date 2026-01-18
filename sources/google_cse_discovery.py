from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from clients.google_cse_client import GoogleCSEClient
from storage.cse_cache import CSECache, CacheEntry, utc_now_iso
from storage.candidate_store import (
    CandidateHit, CandidateStore, normalize_domain, looks_like_company_site_domain
)
from sources.query_generator import QuerySpec


@dataclass
class DiscoveryConfig:
    # How many result pages per query (1 page = up to 10 results)
    pages_per_query: int = 1
    num_per_page: int = 10  # max 10 :contentReference[oaicite:6]{index=6}
    # Stop if too many queries
    max_queries: Optional[int] = None


class GoogleCSEDiscovery:
    def __init__(self, client: GoogleCSEClient, cache: CSECache, store: CandidateStore):
        self.client = client
        self.cache = cache
        self.store = store

    def run(self, queries: Iterable[QuerySpec], cfg: DiscoveryConfig) -> None:
        count = 0
        for qs in queries:
            count += 1
            if cfg.max_queries is not None and count > cfg.max_queries:
                break
            self._run_single_query(qs, cfg)

        self.store.save()

    def _run_single_query(self, qs: QuerySpec, cfg: DiscoveryConfig) -> None:
        for page_idx in range(cfg.pages_per_query):
            start = 1 + page_idx * cfg.num_per_page
            request = {
                "key": self.client.settings.google_api_key,   # will be stripped in cache
                "cx": self.client.settings.google_cx,
                "q": qs.q,
                "start": start,
                "num": cfg.num_per_page,
                "hl": self.client.settings.hl,
                "gl": self.client.settings.gl,
                "cr": self.client.settings.cr,
                "safe": self.client.settings.safe,
            }

            cached = self.cache.get(request)
            if cached is not None:
                if cached.ok and cached.response:
                    self._consume_response(qs.q, cached.fetched_at, cached.response)
                continue

            http_res = self.client.search(q=qs.q, start=start, num=cfg.num_per_page)
            entry = CacheEntry(
                ok=http_res.ok,
                status_code=http_res.status_code,
                fetched_at=utc_now_iso(),
                request=request,
                response=http_res.data,
                error=http_res.error,
                from_cache=False,
            )
            self.cache.set(entry)

            if http_res.ok and http_res.data:
                self._consume_response(qs.q, entry.fetched_at, http_res.data)

    def _consume_response(self, query: str, fetched_at: str, data: Dict[str, Any]) -> None:
        items = data.get("items") or []
        rank = 0
        for it in items:
            rank += 1
            url = (it.get("link") or "").strip()
            if not url:
                continue
            domain = normalize_domain(url)
            if not looks_like_company_site_domain(domain):
                continue

            hit = CandidateHit(
                discovered_at=fetched_at,
                query=query,
                rank=rank,
                url=url,
                title=(it.get("title") or "")[:300],
                snippet=(it.get("snippet") or "")[:800],
            )
            self.store.add_hit(domain=domain, example_url=url, hit=hit)
