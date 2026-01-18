from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx

from app.settings import Settings


@dataclass
class GoogleCSEHttpResult:
    ok: bool
    status_code: int
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GoogleCSEClient:
    """
    Thin client for Google Custom Search JSON API.

    Endpoint base URL documented as customsearch.googleapis.com. :contentReference[oaicite:3]{index=3}
    The list method takes key, cx, q, start, num, etc. :contentReference[oaicite:4]{index=4}
    """
    BASE_URL = "https://customsearch.googleapis.com/customsearch/v1"

    def __init__(self, settings: Settings, timeout: float = 20.0, max_retries: int = 4):
        self.settings = settings
        self.timeout = timeout
        self.max_retries = max_retries

    def search(
        self,
        q: str,
        start: int = 1,
        num: int = 10,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        cr: Optional[str] = None,
        safe: Optional[str] = None,
    ) -> GoogleCSEHttpResult:
        """
        Performs a single API call.
        Notes:
          - num max is 10, and start+num cannot exceed 100 across pages. :contentReference[oaicite:5]{index=5}
        """
        params: Dict[str, Any] = {
            "key": self.settings.google_api_key,
            "cx": self.settings.google_cx,
            "q": q,
            "start": int(start),
            "num": int(num),
            "hl": hl or self.settings.hl,
            "gl": gl or self.settings.gl,
            "cr": cr or self.settings.cr,
            "safe": safe or self.settings.safe,
        }

        headers = {"User-Agent": self.settings.user_agent}

        # Retry with exponential backoff for 429/5xx
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout, follow_redirects=True, headers=headers) as client:
                    r = client.get(self.BASE_URL, params=params)
                if r.status_code == 200:
                    return GoogleCSEHttpResult(ok=True, status_code=200, data=r.json())
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    self._sleep_backoff(attempt)
                    continue
                return GoogleCSEHttpResult(ok=False, status_code=r.status_code, error=r.text[:500])
            except Exception as e:
                last_err = repr(e)
                self._sleep_backoff(attempt)

        return GoogleCSEHttpResult(ok=False, status_code=0, error=last_err or "unknown error")

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        # 0.6, 1.2, 2.4, 4.8 ... + jitter
        base = 0.6 * (2 ** attempt)
        jitter = random.uniform(0.0, 0.4)
        time.sleep(min(10.0, base + jitter))
