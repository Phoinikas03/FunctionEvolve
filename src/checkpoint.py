"""Checkpoint helpers for resumable symbolic-regression runs."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix fallback
    fcntl = None


def utc_timestamp() -> str:
    """Return a compact UTC timestamp for checkpoint metadata."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_checkpoint(path: str | Path) -> Optional[dict[str, Any]]:
    """Load a checkpoint JSON file if it exists."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonable(value: Any) -> Any:
    """Convert common scientific-Python scalar/container values to JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return value


def write_checkpoint_atomic(path: str | Path, state: dict[str, Any]) -> None:
    """Atomically write checkpoint state.

    The temporary file lives in the same directory so ``os.replace`` is atomic
    on the target filesystem.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(
        f".{checkpoint_path.name}.{os.getpid()}.tmp"
    )
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(state),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, checkpoint_path)


def update_manifest(
    checkpoint_dir: str | Path,
    equation_tag: str,
    entry: dict[str, Any],
) -> None:
    """Best-effort manifest update for quick run-level status inspection."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ckpt_dir / "manifest.json"
    lock_path = ckpt_dir / ".manifest.lock"
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            manifest = load_checkpoint(manifest_path) or {
                "schema_version": 1,
                "equations": {},
            }
            manifest["updated_at"] = utc_timestamp()
            manifest.setdefault("equations", {})[equation_tag] = entry
            write_checkpoint_atomic(manifest_path, manifest)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
