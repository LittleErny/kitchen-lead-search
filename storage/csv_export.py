# storage/csv_export.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


CSV_COLUMNS = [
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
    "relevant",
    "discovered_at",
    # extra but useful
    "domain",
    "confidence",
]


def export_leads_csv(leads_by_domain: Dict[str, Dict[str, Any]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for _domain, rec in leads_by_domain.items():
        row = {k: rec.get(k, "") for k in CSV_COLUMNS}
        rows.append(row)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
