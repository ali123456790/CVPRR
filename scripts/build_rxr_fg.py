#!/usr/bin/env python
"""Deterministically curate RxR-FG (fine-grained) split.
This is a placeholder that writes a fixed JSON schema with filter criteria
so results are reproducible when you implement the real curation.
"""
import json, hashlib, pathlib, sys

criteria = {
  "relations": ["between", "next to", "on", "under", "left of", "right of"],
  "attributes": ["ceramic", "wooden", "smallest", "largest", "red", "blue"]
}
payload = {
  "version": "0.1",
  "seed": 17,
  "criteria": criteria,
  "note": "Implement actual filtering over RxR annotations; keep schema stable."
}
out = pathlib.Path("data/processed/rxr_fg_criteria.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2))
print(f"Wrote {out}")
