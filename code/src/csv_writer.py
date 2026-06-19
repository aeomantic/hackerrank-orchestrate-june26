"""Writes OutputRow objects to a CSV with the exact column order required
by the problem statement."""

from __future__ import annotations

import csv
from pathlib import Path

from src.schemas import OutputRow, OUTPUT_COLUMN_ORDER


def write_output_csv(rows: list[OutputRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMN_ORDER)
        writer.writeheader()
        for row in rows:
            d = row.model_dump()
            # bools -> lowercase "true"/"false" strings to match the
            # problem statement's example output formatting exactly.
            d["evidence_standard_met"] = str(d["evidence_standard_met"]).lower()
            d["valid_image"] = str(d["valid_image"]).lower()
            writer.writerow({k: d[k] for k in OUTPUT_COLUMN_ORDER})
