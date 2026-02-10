from __future__ import annotations

import hmac
import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.settings import load_dotenv
from service import db
from service.models import (
    ResultsResponse,
    RunCreateRequest,
    RunCreateResponse,
    RunStatusResponse,
)
from service.runner import create_run, request_cancel, start_run_thread


logger = logging.getLogger("lead_discovery_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


SERVICE_VERSION = "0.1.1"
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", "1048576"))  # 1MB default
DEBUG = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")


def _get_api_token() -> str:
    token = os.getenv("API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("API_TOKEN is required")
    return token


def _auth_required(authorization: Optional[str] = Header(None)) -> None:
    token = _get_api_token()
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    provided = authorization.split(" ", 1)[1].strip()
    if not hmac.compare_digest(provided, token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


app = FastAPI(title="Lead Discovery Runner", version=SERVICE_VERSION)


@app.on_event("startup")
def _startup() -> None:
    load_dotenv(".env")
    db.init_db()
    db.mark_incomplete_runs_failed(now_iso=_utc_now_iso())
    _get_api_token()


@app.middleware("http")
async def _body_limit_middleware(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
            if size > MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"error": {"type": "RequestTooLarge", "message": "Request body too large"}},
                )
        except ValueError:
            pass
    return await call_next(request)


@app.exception_handler(RequestValidationError)
def _validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "type": "ValidationError",
                "message": "Invalid request body",
                "details": exc.errors(),
            }
        },
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "lead-discovery-runner", "version": SERVICE_VERSION}


@app.post("/runs", status_code=202, response_model=RunCreateResponse)
def start_run(
    req: RunCreateRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    _auth: Any = Depends(_auth_required),
) -> RunCreateResponse:
    row = create_run(req, idempotency_key)
    run_id = row.get("run_id")
    if not run_id:
        raise HTTPException(status_code=500, detail="Failed to create run")
    if row.get("status", "queued") == "queued":
        start_run_thread(run_id)
    return RunCreateResponse(
        run_id=run_id,
        status=row.get("status", "queued"),
        created_at=row.get("created_at", _utc_now_iso()),
        links={"status": f"/runs/{run_id}", "results": f"/runs/{run_id}/results"},
    )


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
def get_run(run_id: str, _auth: Any = Depends(_auth_required)) -> RunStatusResponse:
    row = db.fetch_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    progress = json.loads(row["progress_json"])
    metrics = json.loads(row["metrics_json"])
    error = json.loads(row["error_json"]) if row["error_json"] else None
    if error and not DEBUG:
        error = {k: error.get(k) for k in ("type", "message", "trace_id")}
    return RunStatusResponse(
        run_id=row["run_id"],
        status=row["status"],
        progress=progress,
        metrics=metrics,
        error=error,
    )


@app.get("/runs/{run_id}/results", response_model=ResultsResponse)
def get_results(
    run_id: str,
    cursor: Optional[str] = None,
    limit: int = 200,
    _auth: Any = Depends(_auth_required),
) -> ResultsResponse:
    run_row = db.fetch_one("SELECT run_id FROM runs WHERE run_id = ?", (run_id,))
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be > 0")
    limit = min(limit, 1000)
    last_id = 0
    if cursor:
        try:
            last_id = int(cursor)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid cursor")

    rows = db.fetch_all(
        "SELECT id, lead_json FROM run_results WHERE run_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
        (run_id, last_id, limit + 1),
    )
    items = []
    next_cursor = None
    for row in rows[:limit]:
        items.append(json.loads(row["lead_json"]))
        next_cursor = str(row["id"])

    has_more = len(rows) > limit
    if not has_more:
        next_cursor = None

    return ResultsResponse(run_id=run_id, items=items, next_cursor=next_cursor, has_more=has_more)


@app.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, _auth: Any = Depends(_auth_required)) -> Dict[str, Any]:
    row = db.fetch_one("SELECT status, progress_json, metrics_json FROM runs WHERE run_id = ?", (run_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    db.execute("UPDATE runs SET cancel_requested = 1 WHERE run_id = ?", (run_id,))
    request_cancel(run_id)
    if row["status"] == "queued":
        progress = json.loads(row["progress_json"])
        progress.update({"stage": "cancelled", "percent": 100.0, "message": "Cancelled"})
        db.execute(
            "UPDATE runs SET status = ?, progress_json = ?, updated_at = ? WHERE run_id = ?",
            ("cancelled", json.dumps(progress, ensure_ascii=False), _utc_now_iso(), run_id),
        )
        return {"run_id": run_id, "status": "cancelled"}
    return {"run_id": run_id, "status": row["status"]}


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
