#!/usr/bin/env python3
"""Slice 1: connect to IMAP, print one CLAIM: message, exit.

Leaves the message unread. Does not write files, ntfy, or reply.
"""

from __future__ import annotations

import os
import sys
from email import policy
from email.header import decode_header, make_header
from email.message import Message
from email.parser import BytesParser

import imaplib

REQUIRED_ENV = ("IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD")
PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".gif", ".tif", ".tiff")


def load_env_file(path: str) -> None:
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)


def load_config() -> dict[str, str | int]:
    env_file = os.environ.get("ENV_FILE")
    if env_file:
        load_env_file(env_file)
    elif os.path.isfile(".env"):
        load_env_file(".env")

    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print(f"missing env: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)

    port_raw = os.environ.get("IMAP_PORT", "993").strip() or "993"
    try:
        port = int(port_raw)
    except ValueError:
        print("IMAP_PORT must be an integer", file=sys.stderr)
        sys.exit(2)

    return {
        "host": os.environ["IMAP_HOST"].strip(),
        "port": port,
        "user": os.environ["IMAP_USER"].strip(),
        "password": os.environ["IMAP_PASSWORD"],
    }


def decode_mime_header(value: str | None) -> str:
    if not value:
        return ""
    return str(make_header(decode_header(value)))


def is_claim_subject(subject: str) -> bool:
    return subject.strip().lower().startswith("claim:")


def fetch_part(imap: imaplib.IMAP4, seq: bytes, spec: str) -> bytes | None:
    typ, data = imap.fetch(seq, spec)
    if typ != "OK" or not data:
        return None
    for item in data:
        if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], (bytes, bytearray)):
            return bytes(item[1])
    return None


def parse_headers(raw: bytes) -> Message:
    return BytesParser(policy=policy.default).parsebytes(raw)


def attachment_entries(msg: Message) -> list[tuple[str, bool]]:
    entries: list[tuple[str, bool]] = []
    parts: list[Message]
    if msg.is_multipart():
        parts = [p for p in msg.walk() if p.get_content_maintype() != "multipart"]
    else:
        parts = [msg]

    for part in parts:
        filename = part.get_filename()
        name = decode_mime_header(filename) if filename else ""
        ctype = (part.get_content_type() or "").lower()
        disposition = (part.get_content_disposition() or "").lower()
        is_image = ctype.startswith("image/") or name.lower().endswith(PHOTO_SUFFIXES)
        is_file = bool(name) or disposition in ("attachment", "inline") or is_image
        if not is_file:
            continue
        if not name:
            name = f"(unnamed {ctype or 'part'})"
        entries.append((name, is_image))
    return entries


def find_latest_claim(imap: imaplib.IMAP4) -> tuple[bytes, Message] | None:
    typ, data = imap.search(None, "ALL")
    if typ != "OK" or not data or not data[0]:
        return None
    seqs = data[0].split()
    for seq in reversed(seqs):
        raw_headers = fetch_part(imap, seq, "(BODY.PEEK[HEADER.FIELDS (SUBJECT DATE)])")
        if not raw_headers:
            continue
        headers = parse_headers(raw_headers)
        subject = decode_mime_header(headers.get("Subject"))
        if not is_claim_subject(subject):
            continue
        raw_full = fetch_part(imap, seq, "(BODY.PEEK[])")
        if not raw_full:
            return None
        return seq, parse_headers(raw_full)
    return None


def report(msg: Message) -> None:
    subject = decode_mime_header(msg.get("Subject"))
    date = decode_mime_header(msg.get("Date"))
    attachments = attachment_entries(msg)
    names = ", ".join(name for name, _ in attachments) if attachments else "(none)"
    photo = "yes" if any(is_photo for _, is_photo in attachments) else "no"
    print("CLAIM message found")
    print(f"subject: {subject}")
    print(f"date: {date}")
    print(f"attachments: {names}")
    print(f"photo: {photo}")


def main() -> None:
    cfg = load_config()
    imap: imaplib.IMAP4 | None = None
    try:
        imap = imaplib.IMAP4_SSL(str(cfg["host"]), int(cfg["port"]))
        imap.login(str(cfg["user"]), str(cfg["password"]))
        typ, _ = imap.select("INBOX", readonly=True)
        if typ != "OK":
            print("failed to select INBOX", file=sys.stderr)
            sys.exit(1)
        found = find_latest_claim(imap)
        if found is None:
            print("No CLAIM: message found in INBOX")
            return
        _, msg = found
        report(msg)
    except imaplib.IMAP4.error as exc:
        print(f"IMAP error: {exc}", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(f"connection error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if imap is not None:
            try:
                imap.logout()
            except Exception:
                pass


if __name__ == "__main__":
    main()
