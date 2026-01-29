from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.settings import Settings
from crawling.crawler import SiteCrawler
from crawling.fetcher import CachedFetcher
from discovery.google.google_cse import DiscoveryConfig, GoogleCSEDiscovery
from discovery.google.google_cse_client import GoogleCSEClient
from discovery.google.query_generator import QueryGenerator, QuerySpec
from evaluation.evaluator import SiteEvaluator
from storage.candidate_store import CandidateStore
from storage.cse_cache import CSECache
from storage.leads_store import lead_id_from_domain, utc_now_iso

from service import db
from service.models import RunCreateRequest


logger = logging.getLogger("lead_discovery_runner")


_cancel_events: Dict[str, threading.Event] = {}


def _ensure_cancel_event(run_id: str) -> threading.Event:
    ev = _cancel_events.get(run_id)
    if ev is None:
        ev = threading.Event()
        _cancel_events[run_id] = ev
    return ev


def request_cancel(run_id: str) -> None:
    ev = _ensure_cancel_event(run_id)
    ev.set()


def _is_cancel_requested(run_id: str) -> bool:
    ev = _cancel_events.get(run_id)
    if ev and ev.is_set():
        return True
    row = db.fetch_one("SELECT cancel_requested FROM runs WHERE run_id = ?", (run_id,))
    return bool(row["cancel_requested"]) if row else False


def _update_run(
    run_id: str,
    *,
    status: Optional[str] = None,
    progress: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> None:
    row = db.fetch_one(
        "SELECT status, progress_json, metrics_json, error_json, updated_at FROM runs WHERE run_id = ?",
        (run_id,),
    )
    if not row:
        return
    new_status = status or row["status"]
    new_progress = progress or json.loads(row["progress_json"])
    new_metrics = metrics or json.loads(row["metrics_json"])
    new_error = error if error is not None else (json.loads(row["error_json"]) if row["error_json"] else None)
    now = utc_now_iso()
    db.execute(
        """
        UPDATE runs
        SET status = ?,
            progress_json = ?,
            metrics_json = ?,
            error_json = ?,
            updated_at = ?
        WHERE run_id = ?
        """,
        (
            new_status,
            json.dumps(new_progress, ensure_ascii=False),
            json.dumps(new_metrics, ensure_ascii=False),
            json.dumps(new_error, ensure_ascii=False) if new_error else None,
            now,
            run_id,
        ),
    )


def _pick_source(candidate: Dict[str, Any]) -> str:
    hits = candidate.get("hits") or []
    if not hits:
        return "google_cse"
    h0 = hits[0]
    q = (h0.get("query") or "").replace("\n", " ")
    rank = h0.get("rank")
    url = h0.get("url")
    return f"google_cse|query={q}|rank={rank}|url={url}"


def _store_run_result(run_id: str, lead_id: str, lead_json: Dict[str, Any]) -> bool:
    try:
        db.execute(
            """
            INSERT INTO run_results (run_id, lead_id, lead_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, lead_id, json.dumps(lead_json, ensure_ascii=False), utc_now_iso()),
        )
        return True
    except Exception:
        return False


def _store_master_lead(lead_id: str, lead_json: Dict[str, Any]) -> bool:
    row = db.fetch_one("SELECT lead_id FROM leads WHERE lead_id = ?", (lead_id,))
    now = utc_now_iso()
    if row:
        db.execute(
            "UPDATE leads SET lead_json = ?, last_seen_at = ? WHERE lead_id = ?",
            (json.dumps(lead_json, ensure_ascii=False), now, lead_id),
        )
        return False
    db.execute(
        "INSERT INTO leads (lead_id, lead_json, first_seen_at, last_seen_at) VALUES (?, ?, ?, ?)",
        (lead_id, json.dumps(lead_json, ensure_ascii=False), now, now),
    )
    return True


def _hash_payload(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def create_run(params: RunCreateRequest, idempotency_key: Optional[str]) -> Dict[str, Any]:
    now = utc_now_iso()
    payload = params.dict()
    payload_hash = _hash_payload(payload)

    if idempotency_key:
        row = db.fetch_one(
            "SELECT run_id FROM idempotency WHERE idem_key = ? AND payload_hash = ?",
            (idempotency_key, payload_hash),
        )
        if row:
            run_id = row["run_id"]
            run_row = db.fetch_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            return dict(run_row) if run_row else {"run_id": run_id}

    run_id = _new_run_id()
    progress = {
        "stage": "queued",
        "percent": 0.0,
        "message": "Queued",
        "started_at": None,
        "updated_at": now,
    }
    metrics = {
        "requests_made": 0,
        "leads_found_total": 0,
        "leads_new": 0,
        "leads_duplicates": 0,
        "errors": 0,
    }
    db.execute(
        """
        INSERT INTO runs (
            run_id, status, created_at, started_at, updated_at,
            params_json, progress_json, metrics_json, error_json,
            idempotency_key, payload_hash, cancel_requested
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            "queued",
            now,
            None,
            now,
            json.dumps(payload, ensure_ascii=False),
            json.dumps(progress, ensure_ascii=False),
            json.dumps(metrics, ensure_ascii=False),
            None,
            idempotency_key,
            payload_hash,
            0,
        ),
    )
    if idempotency_key:
        db.execute(
            """
            INSERT OR IGNORE INTO idempotency (idem_key, payload_hash, run_id, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (idempotency_key, payload_hash, run_id, now),
        )
    run_row = db.fetch_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    return dict(run_row) if run_row else {"run_id": run_id, "status": "queued", "created_at": now}


def _new_run_id() -> str:
    import uuid

    return str(uuid.uuid4())


def start_run_thread(run_id: str) -> None:
    t = threading.Thread(target=_run_job, args=(run_id,), daemon=True)
    t.start()


def _run_job(run_id: str) -> None:
    row = db.fetch_one("SELECT params_json FROM runs WHERE run_id = ?", (run_id,))
    if not row:
        return
    params = RunCreateRequest.parse_obj(json.loads(row["params_json"]))
    cancel_event = _ensure_cancel_event(run_id)

    progress = {
        "stage": "running",
        "percent": 0.0,
        "message": "Starting",
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    _update_run(run_id, status="running", progress=progress)

    try:
        _execute_run(run_id, params, cancel_event)
        if _is_cancel_requested(run_id):
            progress.update(
                {"stage": "cancelled", "percent": 100.0, "message": "Cancelled", "updated_at": utc_now_iso()}
            )
            _update_run(run_id, status="cancelled", progress=progress)
        else:
            progress.update(
                {"stage": "done", "percent": 100.0, "message": "Finished", "updated_at": utc_now_iso()}
            )
            _update_run(run_id, status="finished", progress=progress)
    except Exception as exc:
        trace_id = f"err-{run_id[:8]}"
        logger.exception("Run failed: run_id=%s trace_id=%s", run_id, trace_id)
        error = {"type": type(exc).__name__, "message": str(exc), "trace_id": trace_id}
        progress.update({"stage": "failed", "message": "Failed", "updated_at": utc_now_iso()})
        _update_run(run_id, status="failed", progress=progress, error=error)


def _execute_run(run_id: str, params: RunCreateRequest, cancel_event: threading.Event) -> None:
    metrics = {
        "requests_made": 0,
        "leads_found_total": 0,
        "leads_new": 0,
        "leads_duplicates": 0,
        "errors": 0,
    }

    if "google" in [s.lower() for s in params.sources]:
        _update_progress(run_id, "searching", 5.0, "Running Google CSE discovery", metrics)
        queries = _build_queries(params)
        if not queries:
            _update_progress(run_id, "searching", 10.0, "No queries provided", metrics)
            candidates = {}
        else:
            candidates = _run_google_discovery(run_id, queries, params, metrics)
    else:
        candidates = {}

    if _is_cancel_requested(run_id):
        return

    _update_progress(run_id, "scraping", 20.0, "Evaluating candidate sites", metrics)
    _run_evaluation(run_id, candidates, params, metrics, cancel_event)

    _update_progress(run_id, "saving", 95.0, "Persisting results", metrics)


def _build_queries(params: RunCreateRequest) -> List[QuerySpec]:
    if params.queries:
        return [QuerySpec(q=q, tag="manual", city="") for q in params.queries]
    cities = params.geo.cities if params.geo and params.geo.cities else None
    gen = QueryGenerator(cities=cities)
    return gen.generate()


def _run_google_discovery(
    run_id: str,
    queries: List[QuerySpec],
    params: RunCreateRequest,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    settings = Settings.from_env(".env")
    cache_dir = Path(".cache/google_cse")
    run_dir = Path(".cache/google_cse") / "runs" / run_id
    cache = CSECache(cache_dir=cache_dir)
    store = CandidateStore(out_dir=run_dir)
    client = GoogleCSEClient(settings=settings, timeout=params.limits.request_timeout_s)
    discovery = GoogleCSEDiscovery(client=client, cache=cache, store=store)
    cfg = DiscoveryConfig(
        pages_per_query=params.limits.max_pages if params.limits.max_pages > 0 else 1,
        num_per_page=10,
        max_queries=len(queries),
    )
    discovery.run(queries=queries, cfg=cfg)
    metrics["requests_made"] += discovery._http_requests  # best-effort
    store.save()
    return json.loads((run_dir / "candidates_index.json").read_text(encoding="utf-8")) if store.all_domains() else {}


def _update_progress(
    run_id: str,
    stage: str,
    percent: float,
    message: str,
    metrics: Dict[str, Any],
) -> None:
    progress = {
        "stage": stage,
        "percent": float(percent),
        "message": message,
        "started_at": None,
        "updated_at": utc_now_iso(),
    }
    row = db.fetch_one("SELECT progress_json FROM runs WHERE run_id = ?", (run_id,))
    if row:
        cur = json.loads(row["progress_json"])
        progress["started_at"] = cur.get("started_at")
    _update_run(run_id, progress=progress, metrics=metrics)


def _run_evaluation(
    run_id: str,
    candidates: Dict[str, Any],
    params: RunCreateRequest,
    metrics: Dict[str, Any],
    cancel_event: threading.Event,
) -> None:
    fetcher = CachedFetcher(
        cache_dir=".cache/http",
        timeout=float(params.limits.request_timeout_s),
        min_delay=0.3,
        max_delay=1.0,
    )
    crawler = SiteCrawler(fetcher=fetcher, max_pages=5)
    evaluator = SiteEvaluator()

    domains = list(candidates.keys())
    if params.limits.max_leads > 0:
        domains = domains[: params.limits.max_leads]

    total = len(domains) if domains else 0
    completed = 0

    async def worker(domain: str, cand: Dict[str, Any]) -> None:
        nonlocal completed
        try:
            if _is_cancel_requested(run_id):
                return
            example_url = cand.get("example_url") or domain
            if not example_url:
                return
            if not example_url.startswith("http"):
                example_url = "https://" + example_url
            agg = await asyncio.to_thread(crawler.collect, example_url)
            text = getattr(agg, "aggregated_text", "") or ""
            home_html = getattr(agg, "home_html", "") or ""
            ev = await asyncio.to_thread(evaluator.evaluate, example_url, home_html, text)

            emails = (ev.contacts or {}).get("emails") or []
            phones = (ev.contacts or {}).get("phones") or []

            lead_id = lead_id_from_domain(ev.domain or domain)
            lead = {
                "lead_id": lead_id,
                "name": ev.company_name or (ev.domain or domain),
                "domain": ev.domain or domain,
                "website": example_url,
                "country": ev.country or "unknown",
                "city": ev.city or "unknown",
                "emails": emails,
                "phones": phones,
                "category": ev.category,
                "source": _pick_source(cand),
                "relevance_score": int(ev.relevance_score),
                "confidence": float(ev.confidence),
                "discovered_at": utc_now_iso(),
            }

            if params.dedupe.enabled:
                is_new = _store_master_lead(lead_id, lead)
                if not is_new:
                    metrics["leads_duplicates"] += 1
                    return
                metrics["leads_new"] += 1
            else:
                metrics["leads_new"] += 1

            if _store_run_result(run_id, lead_id, lead):
                metrics["leads_found_total"] += 1
        except Exception:
            metrics["errors"] += 1
        finally:
            completed += 1
            if total > 0:
                pct = 20.0 + (70.0 * (completed / total))
                _update_progress(run_id, "scraping", pct, f"Processed {completed}/{total}", metrics)

    async def run_all() -> None:
        sem = asyncio.Semaphore(max(1, params.limits.global_concurrency))
        tasks = []
        for domain in domains:
            cand = candidates.get(domain, {})
            async def _wrap(d: str, c: Dict[str, Any]) -> None:
                async with sem:
                    await worker(d, c)
            tasks.append(asyncio.create_task(_wrap(domain, cand)))
        if tasks:
            await asyncio.gather(*tasks)

    asyncio.run(run_all())
