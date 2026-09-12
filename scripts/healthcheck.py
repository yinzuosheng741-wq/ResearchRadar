"""Lightweight, offline project health check for CI and demos."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def run(root: Path) -> dict[str, object]:
    checks = {
        "seed_config": (root / "config" / "seed_queries.yml").is_file(),
        "rag_config": (root / "config" / "rag.yml").is_file(),
        "evaluation_dataset": (root / "data" / "evaluation" / "questions-annotated.jsonl").is_file(),
        "web_entrypoint": (root / "web" / "app.py").is_file(),
    }
    return {"ok": all(checks.values()), "checks": checks}


if __name__ == "__main__":
    payload = run(Path(__file__).resolve().parents[1])
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    sys.exit(0 if payload["ok"] else 1)
