from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LimitsModel(BaseModel):
    max_leads: int = 2000
    max_pages: int = 20
    per_domain_concurrency: int = 1
    global_concurrency: int = 6
    request_timeout_s: int = 20

    @validator("*")
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("limits values must be >= 0")
        return v


class DedupeModel(BaseModel):
    enabled: bool = True
    strategy: str = "domain_email_phone"


class OutputModel(BaseModel):
    include_raw: bool = False


class GeoModel(BaseModel):
    country: str = "SA"
    cities: List[str] = Field(default_factory=list)


class RunCreateRequest(BaseModel):
    run_name: Optional[str] = None
    mode: str = "refresh"
    evaluator_mode: Optional[str] = None
    sources: List[str] = Field(default_factory=lambda: ["google"])
    queries: List[str] = Field(default_factory=list)
    geo: GeoModel = Field(default_factory=GeoModel)
    limits: LimitsModel = Field(default_factory=LimitsModel)
    dedupe: DedupeModel = Field(default_factory=DedupeModel)
    output: OutputModel = Field(default_factory=OutputModel)
    callback_url: Optional[str] = None

    @validator("evaluator_mode")
    def normalize_evaluator_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        mode = str(v).strip().lower()
        if not mode:
            return None
        if mode not in ("precision", "recall"):
            raise ValueError("evaluator_mode must be 'precision' or 'recall'")
        return mode


class RunCreateResponse(BaseModel):
    run_id: str
    status: str
    created_at: str
    links: Dict[str, str]


class ProgressModel(BaseModel):
    stage: str
    percent: float
    message: str
    started_at: Optional[str]
    updated_at: str


class MetricsModel(BaseModel):
    requests_made: int = 0
    leads_found_total: int = 0
    leads_new: int = 0
    leads_duplicates: int = 0
    errors: int = 0


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    progress: ProgressModel
    metrics: MetricsModel
    error: Optional[Dict[str, Any]] = None


class LeadItem(BaseModel):
    lead_id: str
    name: str
    domain: str
    website: str
    country: str
    city: str
    emails: List[str]
    phones: List[str]
    category: str
    source: str
    relevance_score: int
    confidence: float
    discovered_at: str
    empty_content: bool = False
    decision: str = "reject"


class ResultsResponse(BaseModel):
    run_id: str
    items: List[LeadItem]
    next_cursor: Optional[str]
    has_more: bool
