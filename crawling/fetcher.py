# fetcher.py
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx


@dataclass
class FetchResult:
    url: str
    final_url: str
    ok: bool
    status_code: int
    content_type: str
    text: str
    error: str = ""
    from_cache: bool = False


class CachedFetcher:
    """
    Polite HTTP fetcher with:
      - disk cache (URL -> response text + metadata)
      - per-domain rate limiting with jitter (micro-delays)
      - retries + exponential backoff for common transient errors (429/5xx/timeouts)

    Notes:
      - This fetcher intentionally does NOT try to bypass CAPTCHAs or WAFs.
      - It will cache results to reduce repeated hits (good for avoiding blocks).
    """

    def __init__(
        self,
        cache_dir: str = ".cache/http",
        timeout: float = 20.0,
        max_retries: int = 3,
        base_backoff: float = 1.0,
        min_delay: float = 0.4,
        max_delay: float = 1.6,
        user_agent: Optional[str] = None,
        accept_language: str = "en-US,en;q=0.9,ar;q=0.8",
        cache_ttl_seconds: Optional[int] = None,
        max_response_bytes: int = 3_000_000,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.timeout = timeout
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_response_bytes = max_response_bytes

        self.user_agent = user_agent or (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )
        self.accept_language = accept_language

        self._last_request_ts: Dict[str, float] = {}
        self._lock = threading.Lock()

        self._client = httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(self.timeout),
            headers={
                "User-Agent": self.user_agent,
                "Accept-Language": self.accept_language,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )

    # ---------- Public API ----------

    def close(self) -> None:
        self._client.close()

    def fetch_text(self, url: str, force_refresh: bool = False) -> FetchResult:
        """
        Fetch URL, returning decoded text.
        Uses cache unless force_refresh is True.
        """
        url = self._normalize_url(url)
        cached = None if force_refresh else self._load_cache(url)

        if cached is not None:
            return FetchResult(
                url=url,
                final_url=cached.get("final_url", url),
                ok=cached.get("ok", True),
                status_code=int(cached.get("status_code", 200)),
                content_type=cached.get("content_type", ""),
                text=cached.get("text", ""),
                error=cached.get("error", ""),
                from_cache=True,
            )

        # Polite per-domain delay
        self._polite_delay(url)

        # Retry loop
        last_err = ""
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.get(url)
                status = r.status_code
                ctype = r.headers.get("content-type", "")

                # Avoid giant payloads
                raw = r.content
                if len(raw) > self.max_response_bytes:
                    last_err = f"Response too large: {len(raw)} bytes"
                    self._save_cache(url, ok=False, status_code=status, content_type=ctype, text="", error=last_err, final_url=str(r.url))
                    return FetchResult(url=url, final_url=str(r.url), ok=False, status_code=status, content_type=ctype, text="", error=last_err, from_cache=False)

                text = r.text if raw else ""

                # Consider 2xx/3xx as ok; httpx follows redirects anyway
                ok = 200 <= status < 400

                # If throttled or transient server errors: retry
                if status in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {status}"
                    if attempt < self.max_retries:
                        self._backoff(attempt, hint=status)
                        continue

                # Cache and return
                self._save_cache(
                    url,
                    ok=ok,
                    status_code=status,
                    content_type=ctype,
                    text=text,
                    error="" if ok else f"HTTP {status}",
                    final_url=str(r.url),
                )
                return FetchResult(
                    url=url,
                    final_url=str(r.url),
                    ok=ok,
                    status_code=status,
                    content_type=ctype,
                    text=text,
                    error="" if ok else f"HTTP {status}",
                    from_cache=False,
                )

            except (httpx.TimeoutException, httpx.ReadError, httpx.ConnectError) as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    self._backoff(attempt, hint="timeout/net")
                    continue
                self._save_cache(url, ok=False, status_code=0, content_type="", text="", error=last_err, final_url=url)
                return FetchResult(url=url, final_url=url, ok=False, status_code=0, content_type="", text="", error=last_err, from_cache=False)

            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                self._save_cache(url, ok=False, status_code=0, content_type="", text="", error=last_err, final_url=url)
                return FetchResult(url=url, final_url=url, ok=False, status_code=0, content_type="", text="", error=last_err, from_cache=False)

        # Should not reach
        self._save_cache(url, ok=False, status_code=0, content_type="", text="", error=last_err, final_url=url)
        return FetchResult(url=url, final_url=url, ok=False, status_code=0, content_type="", text="", error=last_err, from_cache=False)

    # ---------- Internals: polite delay / backoff ----------

    def _polite_delay(self, url: str) -> None:
        domain = self._domain(url)
        delay = random.uniform(self.min_delay, self.max_delay)

        with self._lock:
            last = self._last_request_ts.get(domain, 0.0)
            now = time.time()
            wait = (last + delay) - now
            if wait > 0:
                time.sleep(wait)
            self._last_request_ts[domain] = time.time()

    def _backoff(self, attempt: int, hint: object = None) -> None:
        # exponential backoff + jitter
        base = self.base_backoff * (2 ** attempt)
        jitter = random.uniform(0.0, 0.4 * base)
        time.sleep(base + jitter)

    # ---------- Internals: cache ----------

    def _cache_paths(self, url: str) -> Tuple[Path, Path]:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        meta_path = self.cache_dir / f"{key}.json"
        body_path = self.cache_dir / f"{key}.txt"
        return meta_path, body_path

    def _load_cache(self, url: str) -> Optional[dict]:
        meta_path, body_path = self._cache_paths(url)
        if not meta_path.exists() or not body_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if self.cache_ttl_seconds is not None:
                ts = float(meta.get("cached_at", 0))
                if ts and (time.time() - ts) > self.cache_ttl_seconds:
                    return None
            meta["text"] = body_path.read_text(encoding="utf-8", errors="replace")
            return meta
        except Exception:
            return None

    def _save_cache(
        self,
        url: str,
        ok: bool,
        status_code: int,
        content_type: str,
        text: str,
        error: str,
        final_url: str,
    ) -> None:
        meta_path, body_path = self._cache_paths(url)
        meta = {
            "url": url,
            "final_url": final_url,
            "ok": bool(ok),
            "status_code": int(status_code),
            "content_type": content_type,
            "error": error,
            "cached_at": time.time(),
        }
        try:
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            body_path.write_text(text or "", encoding="utf-8", errors="replace")
        except Exception:
            # cache failures shouldn't kill the pipeline
            pass

    # ---------- Internals: URL helpers ----------

    def _normalize_url(self, url: str) -> str:
        url = (url or "").strip()
        if not url:
            return url
        # If scheme missing, default to https
        p = urlparse(url)
        if not p.scheme:
            return "https://" + url
        return url

    def _domain(self, url: str) -> str:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
