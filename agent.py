#!/usr/bin/env python3
"""Slice 2: poll one CLAIM: message and save the original proof.

Leaves the message unread. Does not crop, extract fields, ntfy, or reply.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from email import policy
from email.header import decode_header, make_header
from email.message import Message
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path

import imaplib

REQUIRED_ENV = ("IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD")
DEFAULT_EXPENSES_ROOT = "/g-drive"
PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".gif", ".tif", ".tiff")
PDF_SUFFIXES = (".pdf",)
CONTENT_TYPE_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/tiff": ".tiff",
    "application/pdf": ".pdf",
}


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


def load_config() -> dict[str, str | int | Path]:
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

    root = os.environ.get("EXPENSES_ROOT", "").strip() or DEFAULT_EXPENSES_ROOT
    return {
        "host": os.environ["IMAP_HOST"].strip(),
        "port": port,
        "user": os.environ["IMAP_USER"].strip(),
        "password": os.environ["IMAP_PASSWORD"],
        "expenses_root": Path(root),
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


def iter_file_parts(msg: Message) -> list[Message]:
    if msg.is_multipart():
        return [p for p in msg.walk() if p.get_content_maintype() != "multipart"]
    return [msg]


def part_filename(part: Message) -> str:
    filename = part.get_filename()
    return decode_mime_header(filename) if filename else ""


def is_photo_part(part: Message) -> bool:
    name = part_filename(part).lower()
    ctype = (part.get_content_type() or "").lower()
    return ctype.startswith("image/") or name.endswith(PHOTO_SUFFIXES)


def is_pdf_part(part: Message) -> bool:
    name = part_filename(part).lower()
    ctype = (part.get_content_type() or "").lower()
    return ctype == "application/pdf" or name.endswith(PDF_SUFFIXES)


def is_file_part(part: Message) -> bool:
    name = part_filename(part)
    disposition = (part.get_content_disposition() or "").lower()
    return bool(name) or disposition in ("attachment", "inline") or is_photo_part(part) or is_pdf_part(part)


def attachment_entries(msg: Message) -> list[tuple[str, bool]]:
    entries: list[tuple[str, bool]] = []
    for part in iter_file_parts(msg):
        if not is_file_part(part):
            continue
        name = part_filename(part) or f"(unnamed {(part.get_content_type() or 'part').lower()})"
        entries.append((name, is_photo_part(part)))
    return entries


def proof_extension(part: Message) -> str:
    name = part_filename(part).lower()
    suffix = Path(name).suffix if name else ""
    if suffix in PHOTO_SUFFIXES or suffix in PDF_SUFFIXES:
        if suffix == ".jpeg":
            return ".jpg"
        return suffix
    ctype = (part.get_content_type() or "").lower()
    return CONTENT_TYPE_EXT.get(ctype, "")


def extract_proofs(msg: Message) -> list[tuple[str, bytes]]:
    photos: list[tuple[str, bytes]] = []
    pdfs: list[tuple[str, bytes]] = []
    for part in iter_file_parts(msg):
        if is_photo_part(part):
            bucket = photos
        elif is_pdf_part(part):
            bucket = pdfs
        else:
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        ext = proof_extension(part)
        if not ext:
            continue
        bucket.append((ext, bytes(payload)))
    return photos or pdfs


def claim_date(msg: Message) -> datetime | None:
    raw = msg.get("Date")
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        return None


def proof_dest(root: Path, when: datetime, ext: str, index: int) -> Path:
    month_dir = root / when.strftime("%Y-%m") / "proofs"
    stamp = when.strftime("%Y-%m-%d")
    suffix = "" if index == 1 else f"-{index}"
    return month_dir / f"{stamp}_pending_orig{suffix}{ext}"


def atomic_write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".part", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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


def save_originals(msg: Message, root: Path) -> list[Path]:
    when = claim_date(msg)
    if when is None:
        print("no usable Date header; not writing files", file=sys.stderr)
        sys.exit(1)
    proofs = extract_proofs(msg)
    if not proofs:
        print("no photo or PDF proof on this message; not writing files", file=sys.stderr)
        sys.exit(1)
    written: list[Path] = []
    for index, (ext, payload) in enumerate(proofs, start=1):
        dest = proof_dest(root, when, ext, index)
        atomic_write(dest, payload)
        written.append(dest)
    return written


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
        for path in save_originals(msg, Path(str(cfg["expenses_root"]))):
            print(f"wrote: {path}")
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
