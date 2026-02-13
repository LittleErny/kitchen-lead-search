from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from discovery.google.google_cse_client import GoogleCSEClient
from storage.cse_cache import CSECache, CacheEntry, utc_now_iso
from storage.candidate_store import (
    CandidateHit, CandidateStore, normalize_domain, looks_like_company_site_domain
)
from discovery.google.query_generator import QuerySpec


_REQUEST_LOCKS_GUARD = threading.Lock()
_REQUEST_LOCKS: Dict[str, threading.Lock] = {}


def _request_lock_for(key: str) -> threading.Lock:
    # Single-flight lock by cache key to prevent duplicate concurrent HTTP calls
    # for the same Google request when multiple runs execute in parallel.
    with _REQUEST_LOCKS_GUARD:
        lock = _REQUEST_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _REQUEST_LOCKS[key] = lock
        return lock


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
        self._cache_hits = 0
        self._http_requests = 0
        self._quota_exhausted = False

    def run(self, queries: Iterable[QuerySpec], cfg: DiscoveryConfig) -> None:
        count = 0
        for qs in queries:
            count += 1
            if cfg.max_queries is not None and count > cfg.max_queries:
                break
            self._run_single_query(qs, cfg)

        self.store.save()
        print(f"[Google CSE] Cache hits: {self._cache_hits}")
        print(f"[Google CSE] HTTP requests: {self._http_requests}")

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

            key = self.cache._key_for(request)
            lock = _request_lock_for(key)

            with lock:
                cached = self.cache.get(request)
                if cached is not None:
                    self._cache_hits += 1
                    print(f"[Google CSE] Cache hit: q={qs.q} start={start}")
                    if cached.ok and cached.response:
                        self._consume_response(qs.q, cached.fetched_at, cached.response)
                    continue

                # When daily quota is exhausted we can still serve cache hits,
                # but we should skip any new HTTP misses in the same process run.
                if self._quota_exhausted:
                    print(f"[Google CSE] Skip HTTP (quota exhausted): q={qs.q} start={start}")
                    continue

                self._http_requests += 1
                print(f"[Google CSE] HTTP request: q={qs.q} start={start}")
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

                if http_res.status_code == 429 and self.client.is_quota_exhausted_error(http_res.error, http_res.data):
                    self._quota_exhausted = True

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
