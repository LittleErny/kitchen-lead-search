from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple


DEFAULT_DB_PATH = ".data/lead_discovery.db"


def _db_path_from_env() -> Path:
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        return Path(DEFAULT_DB_PATH)
    if db_url.startswith("sqlite:///"):
        return Path(db_url.replace("sqlite:///", "", 1))
    if db_url.startswith("sqlite://"):
        return Path(db_url.replace("sqlite://", "", 1))
    # MVP: only sqlite is supported
    raise RuntimeError("DATABASE_URL must be sqlite:// or sqlite:/// for MVP")


def init_db() -> None:
    path = _db_path_from_env()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                updated_at TEXT NOT NULL,
                params_json TEXT NOT NULL,
                progress_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                error_json TEXT,
                idempotency_key TEXT,
                payload_hash TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                lead_id TEXT NOT NULL,
                lead_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(run_id, lead_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                lead_id TEXT PRIMARY KEY,
                lead_json TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency (
                idem_key TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                run_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(idem_key, payload_hash)
            )
            """
        )
        conn.commit()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = _db_path_from_env()
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


def fetch_one(query: str, params: Tuple[Any, ...]) -> Optional[sqlite3.Row]:
    with _connect() as conn:
        cur = conn.execute(query, params)
        row = cur.fetchone()
        return row


def fetch_all(query: str, params: Tuple[Any, ...]) -> list[sqlite3.Row]:
    with _connect() as conn:
        cur = conn.execute(query, params)
        rows = cur.fetchall()
        return list(rows)


def execute(query: str, params: Tuple[Any, ...]) -> None:
    with _connect() as conn:
        conn.execute(query, params)
        conn.commit()


def execute_returning_id(query: str, params: Tuple[Any, ...]) -> int:
    with _connect() as conn:
        cur = conn.execute(query, params)
        conn.commit()
        return int(cur.lastrowid)


def mark_incomplete_runs_failed(now_iso: str) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE runs
            SET status = 'failed',
                updated_at = ?,
                error_json = ?
            WHERE status IN ('queued', 'running')
            """,
            (
                now_iso,
                json.dumps(
                    {
                        "type": "ServiceRestart",
                        "message": "Server restarted while run was in progress",
                        "trace_id": "restart",
                    }
                ),
            ),
        )
        conn.commit()

