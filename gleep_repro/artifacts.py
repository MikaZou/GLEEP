"""Integrity checks and optional downloads for Git-ignored experiment artifacts."""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from .io_utils import sha256


def _load_index(workspace: Path) -> tuple[Path, dict[str, Any]]:
    index_path = workspace.resolve() / "artifacts" / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"artifact index is missing: {index_path}")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("entries"), list):
        raise ValueError("unsupported artifacts/index.json schema")
    return index_path, data


def _safe_path(workspace: Path, relative: str) -> Path:
    root = workspace.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"artifact path escapes repository: {relative}")
    return path


def audit_artifacts(workspace: Path) -> dict[str, Any]:
    _, index = _load_index(workspace)
    rows = []
    for entry in index["entries"]:
        path = _safe_path(workspace, entry["path"])
        exists = path.is_file()
        size_ok = exists and path.stat().st_size == entry["bytes"]
        digest = sha256(path) if size_ok else None
        rows.append({
            "path": entry["path"],
            "profiles": entry["profiles"],
            "exists": exists,
            "size_ok": size_ok,
            "sha256_ok": digest == entry["sha256"] if digest else False,
            "storage_status": entry.get("storage_status", "unknown"),
            "remote_configured": bool(entry.get("remote_uri")),
        })
    present = sum(row["sha256_ok"] for row in rows)
    return {
        "index": "artifacts/index.json",
        "entry_count": len(rows),
        "verified": present,
        "missing_or_invalid": len(rows) - present,
        "complete": present == len(rows),
        "entries": rows,
    }


def fetch_artifacts(workspace: Path, profile: str) -> dict[str, Any]:
    _, index = _load_index(workspace)
    selected = [
        entry for entry in index["entries"]
        if profile == "all" or profile in entry["profiles"]
    ]
    downloaded, reused, unavailable = [], [], []
    for entry in selected:
        path = _safe_path(workspace, entry["path"])
        if path.is_file() and path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"]:
            reused.append(entry["path"])
            continue
        uri = entry.get("remote_uri")
        if not uri:
            unavailable.append(entry["path"])
            continue
        if not uri.startswith(("https://", "http://")):
            raise ValueError(f"unsupported artifact URI for {entry['path']}: {uri}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        try:
            with urllib.request.urlopen(uri) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if temporary.stat().st_size != entry["bytes"] or sha256(temporary) != entry["sha256"]:
                raise ValueError(f"downloaded artifact failed integrity check: {entry['path']}")
            os.replace(temporary, path)
            downloaded.append(entry["path"])
        finally:
            if temporary.exists():
                temporary.unlink()
    if unavailable:
        raise FileNotFoundError(
            "remote_uri is not configured for: " + ", ".join(unavailable[:5])
            + (" ..." if len(unavailable) > 5 else "")
            + ". Upload artifacts to Hugging Face or cloud storage, then update artifacts/index.json."
        )
    return {"profile": profile, "selected": len(selected), "downloaded": downloaded, "reused": reused}
