# storage/csv_export.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


BASE_COLUMNS = [
    # Required by the technical assignment
    "lead_id",
    "lead_type",
    "category",
    "name",
    "country",
    "city",
    "email",
    "phone",
    "website",
    "source",
    "relevance_score",
    "decision",
    "discovered_at",
    # extra but useful
    "domain",
    "confidence",
    "empty_content",
]


def _flatten_signals(records: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for rec in records:
        sig = rec.get("signals") or {}
        if isinstance(sig, dict):
            for k in sig.keys():
                keys.add(f"signal_{k}")
    return sorted(keys)


def export_leads_csv_records(records: List[Dict[str, Any]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    signal_columns = _flatten_signals(records)
    columns = BASE_COLUMNS + signal_columns + ["signals_json"]

    rows: List[Dict[str, Any]] = []
    for rec in records:
        row = {k: rec.get(k, "") for k in BASE_COLUMNS}
        sig = rec.get("signals") or {}
        for sk in signal_columns:
            key = sk.replace("signal_", "", 1)
            row[sk] = sig.get(key, "")
        row["signals_json"] = json.dumps(sig, ensure_ascii=False)
        rows.append(row)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def export_leads_csv(leads_by_domain: Dict[str, Dict[str, Any]], out_path: str) -> None:
    records = list(leads_by_domain.values())
    export_leads_csv_records(records, out_path)
