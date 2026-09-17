"""Pending ntfy claims under AGENT_STATE_DIR."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CURSOR_NAME = "ntfy-cursor.json"
COMMAND_CURSOR_NAME = "ntfy-command-cursor.json"


def state_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def claim_path(root: Path, claim_id: str) -> Path:
    return state_dir(root) / f"{claim_id}.json"


def load_claim(root: Path, claim_id: str) -> dict[str, Any] | None:
    path = claim_path(root, claim_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_claim(root: Path, claim: dict[str, Any]) -> Path:
    claim = dict(claim)
    claim["updated"] = datetime.now(timezone.utc).isoformat()
    path = claim_path(root, str(claim["id"]))
    path.write_text(json.dumps(claim, indent=2, default=str), encoding="utf-8")
    return path


def iter_claims(root: Path) -> list[dict[str, Any]]:
    folder = state_dir(root)
    found: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.json")):
        if path.name in {CURSOR_NAME, COMMAND_CURSOR_NAME, "prepaid.json", "xai-credit-alert.json"}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("id"):
            found.append(data)
    return found


def awaiting(root: Path) -> list[dict[str, Any]]:
    found = [c for c in iter_claims(root) if c.get("status") == "awaiting_reply"]
    found.sort(key=lambda c: str(c.get("updated") or ""))
    return found


def known_message_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for claim in iter_claims(root):
        mid = claim.get("message_id")
        if isinstance(mid, str) and mid.strip():
            ids.add(mid.strip())
    return ids


def known_ntfy_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for claim in iter_claims(root):
        for item in claim.get("ntfy_ids") or []:
            if item:
                ids.add(str(item))
    return ids


def load_cursor(root: Path, name: str = CURSOR_NAME) -> str | None:
    path = state_dir(root) / name
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    since = data.get("since") if isinstance(data, dict) else None
    return str(since) if since else None


def save_cursor(root: Path, since: str, name: str = CURSOR_NAME) -> None:
    path = state_dir(root) / name
    path.write_text(json.dumps({"since": since}), encoding="utf-8")


def delete_claim(root: Path, claim_id: str) -> None:
    path = claim_path(root, claim_id)
    if path.is_file():
        path.unlink()


def prune(
    root: Path,
    *,
    retention_days: int,
    delete_completed: bool,
) -> list[str]:
    """Drop finished claims, and anything older than retention_days."""
    removed: list[str] = []
    now = datetime.now(timezone.utc)
    keep_days = max(retention_days, 0)
    for claim in iter_claims(root):
        cid = str(claim.get("id") or "")
        if not cid:
            continue
        status = claim.get("status")
        if delete_completed and status in ("logged", "skipped") and claim.get("done_sent"):
            delete_claim(root, cid)
            removed.append(cid)
            continue
        updated = _parse_updated(claim.get("updated"))
        if updated is not None and (now - updated).days >= keep_days:
            delete_claim(root, cid)
            removed.append(cid)
    return removed


def _parse_updated(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
