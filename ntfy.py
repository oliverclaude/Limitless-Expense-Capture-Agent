"""Publish and poll home ntfy. Host/topic come from env, never from git."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

LECA_PREFIX = "[leca "


def topic_url(base: str, topic: str) -> str:
    return f"{base.rstrip('/')}/{urllib.parse.quote(topic, safe='')}"


def publish(
    base: str,
    topic: str,
    *,
    title: str,
    message: str,
    image_path: Path | None = None,
) -> dict[str, Any]:
    url = topic_url(base, topic)
    headers = {
        "Title": _header_value(title),
        "Priority": "4",
        "Accept": "application/json",
    }
    if image_path is not None and image_path.is_file():
        headers["Filename"] = _header_value(image_path.name)
        headers["Message"] = _header_value(message)
        body: bytes = image_path.read_bytes()
        method = "PUT"
    else:
        headers["Content-Type"] = "text/plain; charset=utf-8"
        body = message.encode("utf-8")
        method = "POST"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"ntfy HTTP {exc.code}: {detail}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"id": "", "raw": raw}
    if not isinstance(parsed, dict):
        return {"id": ""}
    return parsed


def poll(base: str, topic: str, since: str | None) -> list[dict[str, Any]]:
    query = {"poll": "1"}
    if since:
        query["since"] = since
    url = topic_url(base, topic) + "/json?" + urllib.parse.urlencode(query)
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"ntfy HTTP {exc.code}: {detail}") from exc
    messages: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("event") == "message":
            messages.append(item)
    return messages


def is_our_message(item: dict[str, Any], known_ids: set[str]) -> bool:
    msg_id = str(item.get("id") or "")
    if msg_id and msg_id in known_ids:
        return True
    text = f"{item.get('title') or ''} {item.get('message') or ''}"
    return LECA_PREFIX in text


def _header_value(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ").strip()
