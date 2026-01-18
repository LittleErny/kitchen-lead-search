from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_domain(url: str) -> str:
    p = urlparse(url)
    host = (p.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def looks_like_company_site_domain(domain: str) -> bool:
    # Filter obvious “non-company” platforms. Extend as needed.
    # ToDo: consider option of going through these as well, as some leads may not have site but just facebook page
    blocked = {
        "facebook.com", "instagram.com", "linkedin.com", "twitter.com", "x.com",
        "youtube.com", "tiktok.com",
        "maps.google.com", "google.com",
        "pinterest.com",
    }
    if not domain:
        return False
    if domain in blocked:
        return False
    return True


@dataclass
class CandidateHit:
    discovered_at: str
    query: str
    rank: int
    url: str
    title: str = ""
    snippet: str = ""


@dataclass
class CandidateCompany:
    domain: str
    first_seen_at: str
    last_seen_at: str
    example_url: str
    hits: List[CandidateHit]


class CandidateStore:
    """
    Stores deduped candidate companies in JSONL with timestamps.

    Files:
      - candidates.jsonl  (append-only; latest snapshot rebuilt on save)
      - candidates_index.json (compact index for fast load)
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / "candidates.jsonl"
        self.index_path = self.out_dir / "candidates_index.json"

        self._by_domain: Dict[str, CandidateCompany] = {}
        self._load()

    def _load(self) -> None:
        if self.index_path.exists():
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            for domain, rec in data.items():
                hits = [CandidateHit(**h) for h in rec.get("hits", [])]
                self._by_domain[domain] = CandidateCompany(
                    domain=domain,
                    first_seen_at=rec["first_seen_at"],
                    last_seen_at=rec["last_seen_at"],
                    example_url=rec["example_url"],
                    hits=hits,
                )
            return

        # Fallback: read JSONL if index missing
        if self.jsonl_path.exists():
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                hits = [CandidateHit(**h) for h in rec.get("hits", [])]
                self._by_domain[rec["domain"]] = CandidateCompany(
                    domain=rec["domain"],
                    first_seen_at=rec["first_seen_at"],
                    last_seen_at=rec["last_seen_at"],
                    example_url=rec["example_url"],
                    hits=hits,
                )

    def add_hit(self, domain: str, example_url: str, hit: CandidateHit, max_hits_per_domain: int = 20) -> None:
        now = hit.discovered_at or utc_now_iso()

        if domain not in self._by_domain:
            self._by_domain[domain] = CandidateCompany(
                domain=domain,
                first_seen_at=now,
                last_seen_at=now,
                example_url=example_url,
                hits=[hit],
            )
        else:
            c = self._by_domain[domain]
            c.last_seen_at = now
            # Append hit if it’s not duplicate of same query+url
            key = (hit.query, hit.url)
            existing = {(h.query, h.url) for h in c.hits}
            if key not in existing:
                c.hits.append(hit)
                if len(c.hits) > max_hits_per_domain:
                    c.hits = c.hits[-max_hits_per_domain:]

    def save(self) -> None:
        # Write compact index
        idx = {}
        for domain, c in self._by_domain.items():
            idx[domain] = {
                "domain": c.domain,
                "first_seen_at": c.first_seen_at,
                "last_seen_at": c.last_seen_at,
                "example_url": c.example_url,
                "hits": [asdict(h) for h in c.hits],
            }
        self.index_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")

        # Write JSONL snapshot
        lines = []
        for domain, c in sorted(self._by_domain.items()):
            lines.append(json.dumps({
                "domain": c.domain,
                "first_seen_at": c.first_seen_at,
                "last_seen_at": c.last_seen_at,
                "example_url": c.example_url,
                "hits": [asdict(h) for h in c.hits],
            }, ensure_ascii=False))
        self.jsonl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def all_domains(self) -> List[str]:
        return sorted(self._by_domain.keys())
