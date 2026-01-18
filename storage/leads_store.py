# storage/leads_store.py
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def lead_id_from_domain(domain: str) -> str:
    return hashlib.sha1(domain.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """
    MVP-safe write: write to tmp then replace.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class LeadRecord:
    lead_id: str
    lead_type: str
    category: str
    name: str
    country: str
    city: str
    email: str
    phone: str
    website: str
    source: str
    relevance_score: int
    relevant: bool
    discovered_at: str

    # debug/optional
    domain: str = ""
    confidence: float = 0.0


class LeadsStore:
    """
    Dedup across runs:
      - key = domain
      - upsert preserves discovered_at (first time) and updates info on reruns
    """
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.by_domain: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.by_domain = {}
            return
        try:
            self.by_domain = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            # MVP: if corrupted, start empty (or keep a .bak if you want later)
            self.by_domain = {}

    def upsert(self, rec: LeadRecord) -> None:
        existing = self.by_domain.get(rec.domain)
        if existing is None:
            self.by_domain[rec.domain] = asdict(rec)
            return

        # Preserve first discovery timestamp and lead_id
        rec_dict = asdict(rec)
        rec_dict["discovered_at"] = existing.get("discovered_at") or rec.discovered_at
        rec_dict["lead_id"] = existing.get("lead_id") or rec.lead_id

        # Keep best score
        if int(existing.get("relevance_score", 0)) > rec.relevance_score:
            rec_dict["relevance_score"] = int(existing.get("relevance_score", 0))
            rec_dict["relevant"] = bool(existing.get("relevant", False))
            rec_dict["confidence"] = float(existing.get("confidence", 0.0))

        # Prefer non-empty contacts
        for k in ["email", "phone"]:
            if not rec_dict.get(k) and existing.get(k):
                rec_dict[k] = existing.get(k)

        self.by_domain[rec.domain] = rec_dict

    def save(self) -> None:
        atomic_write_text(self.path, json.dumps(self.by_domain, ensure_ascii=False, indent=2))
