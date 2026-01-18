from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CacheEntry:
    ok: bool
    fetched_at: str
    request: Dict[str, Any]
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: int = 0
    from_cache: bool = False


class CSECache:
    """
    Simple file cache:
      .cache/google_cse/responses/<sha1>.json
    """
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.responses_dir = cache_dir / "responses"
        self.responses_dir.mkdir(parents=True, exist_ok=True)

    def _key_for(self, request: Dict[str, Any]) -> str:
        # Don't hash the API key to avoid accidentally persisting it in clear anywhere.
        safe_req = dict(request)
        safe_req.pop("key", None)

        raw = json.dumps(safe_req, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def get(self, request: Dict[str, Any]) -> Optional[CacheEntry]:
        key = self._key_for(request)
        path = self.responses_dir / f"{key}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return CacheEntry(
            ok=bool(data.get("ok", False)),
            fetched_at=str(data.get("fetched_at", "")),
            request=data.get("request", {}),
            response=data.get("response"),
            error=data.get("error"),
            status_code=int(data.get("status_code", 0)),
            from_cache=True,
        )

    def set(self, entry: CacheEntry) -> None:
        key = self._key_for(entry.request)
        path = self.responses_dir / f"{key}.json"
        payload = {
            "ok": entry.ok,
            "fetched_at": entry.fetched_at or utc_now_iso(),
            "status_code": entry.status_code,
            "request": self._strip_key(entry.request),
            "response": entry.response,
            "error": entry.error,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _strip_key(req: Dict[str, Any]) -> Dict[str, Any]:
        r = dict(req)
        r.pop("key", None)
        return r
