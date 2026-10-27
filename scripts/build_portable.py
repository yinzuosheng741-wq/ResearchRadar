"""Package a clean application snapshot with an optional prepared embedded runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def selected_files(root: Path):
    visible_files = None
    if (root / ".git").exists():
        try:
            output = subprocess.check_output(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError("git_file_inventory_required") from exc
        visible_files = {name.decode("utf-8") for name in output.split(b"\0") if name}

    def eligible(path):
        return visible_files is None or path.relative_to(root).as_posix() in visible_files

    common = {"desktop.py", "Start.cmd", "README.md", ".env.example", "requirements.txt",
              "requirements-portable.txt", "portable-lock.txt", "app.py", "pyproject.toml"}
    directories = (["stylemate", "assets/demo", "data/knowledge", "docs", ".streamlit"]
                   if (root / "stylemate").is_dir() else
                   ["agent", "config", "domain", "evaluation", "ingestion", "model", "prompts", "providers",
                    "rag", "retrieval", "storage", "tools", "utils", "web", "workflows", "data/evaluation", "docs"])
    if (root / "stylemate").is_dir():
        directories.append("evaluation")
    for name in sorted(common):
        if (root / name).is_file() and eligible(root / name):
            yield root / name
    for directory in directories:
        for path in sorted((root / directory).rglob("*")):
            if not path.is_file() or path.is_symlink() or not eligible(path):
                continue
            parts = path.relative_to(root).parts
            if any(part.startswith(".") and part != ".streamlit" for part in parts):
                continue
            if "__pycache__" in parts or path.suffix in {".pyc", ".db", ".sqlite3", ".log"}:
                continue
            if path.name.startswith(".env") or path.name == "secrets.toml":
                continue
            yield path
    reports = {"evaluation.json", "agent_evaluation.json", "agent_evaluation_20261022_round2.json"}
    manifest_path = root / "artifacts/evaluation_manifest.json"
    if manifest_path.is_file():
        try:
            active = json.loads(manifest_path.read_text("utf-8"))["report"]
            if isinstance(active, str) and Path(active).name == active and active.endswith(".json"):
                reports.update({"evaluation_manifest.json", active})
        except (OSError, ValueError, KeyError, TypeError):
            pass
    for name in sorted(reports):
        path = root / "artifacts" / name
        if path.is_file() and eligible(path):
            yield path


def build(output: Path, runtime: Path | None = None) -> dict:
    name = "StyleMate" if (ROOT / "stylemate").is_dir() else "ResearchRadar"
    if runtime is not None and not (runtime / "python.exe").is_file():
        raise ValueError("prepared_embedded_python_required")
    manifest = {"application": name, "runtime_included": runtime is not None, "files": {}}
    files = [(path, Path(name) / path.relative_to(ROOT)) for path in selected_files(ROOT)]
    if runtime is not None:
        files.extend((path, Path(name) / "runtime" / path.relative_to(runtime))
                     for path in runtime.rglob("*") if path.is_file() and not path.is_symlink()
                     and "__pycache__" not in path.parts and path.suffix != ".pyc")
    if output.resolve() in {path.resolve() for path, _ in files}:
        raise ValueError("output_must_not_replace_source")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for path, target in files:
            data = path.read_bytes()
            manifest["files"][target.as_posix()] = hashlib.sha256(data).hexdigest()
            archive.writestr(target.as_posix(), data)
        archive.writestr(f"{name}/manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"application": name, "file_count": len(files), "runtime_included": runtime is not None,
            "archive": str(output), "bytes": output.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.runtime)))
