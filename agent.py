#!/usr/bin/env python3
"""Slice 9: Done/Skipped email to the configured work address only."""

from __future__ import annotations

import hashlib
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
DEFAULT_STATE_DIR = "/g-drive/agent-state"
DEFAULT_NTFY_TOPIC = "Limitless-Expense-Capture-Agent"
FOLDER_PROCESSED = "processed"
FOLDER_REVIEW = "needs_review"
FOLDER_FAILED = "failed"
MAIL_FOLDERS = (FOLDER_PROCESSED, FOLDER_REVIEW, FOLDER_FAILED)
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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"{name} must be a number", file=sys.stderr)
        sys.exit(2)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"{name} must be an integer", file=sys.stderr)
        sys.exit(2)
    return value


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
    state = os.environ.get("AGENT_STATE_DIR", "").strip() or DEFAULT_STATE_DIR
    smtp_port_raw = os.environ.get("SMTP_PORT", "").strip() or "465"
    try:
        smtp_port = int(smtp_port_raw)
    except ValueError:
        print("SMTP_PORT must be an integer", file=sys.stderr)
        sys.exit(2)
    imap_host = os.environ["IMAP_HOST"].strip()
    imap_user = os.environ["IMAP_USER"].strip()
    return {
        "host": imap_host,
        "port": port,
        "user": imap_user,
        "password": os.environ["IMAP_PASSWORD"],
        "smtp_host": os.environ.get("SMTP_HOST", "").strip() or imap_host,
        "smtp_port": smtp_port,
        "work_email": os.environ.get("WORK_EMAIL", "").strip(),
        "expenses_root": Path(root),
        "state_dir": Path(state),
        "ntfy_url": os.environ.get("NTFY_URL", "").strip(),
        "ntfy_topic": os.environ.get("NTFY_TOPIC", "").strip() or DEFAULT_NTFY_TOPIC,
        "poll_interval_minutes": _env_int("POLL_INTERVAL_MINUTES", 30),
        "state_retention_days": _env_int("STATE_RETENTION_DAYS", 60),
        "delete_completed_state": os.environ.get("DELETE_COMPLETED_STATE", "1").strip()
        not in ("0", "false", "no", "off"),
        "xai_credit_alert_usd": _env_float("XAI_CREDIT_ALERT_USD", 0.05),
        "xai_credit_alert_hours": _env_int("XAI_CREDIT_ALERT_COOLDOWN_HOURS", 6),
    }


def decode_mime_header(value: str | None) -> str:
    if not value:
        return ""
    return str(make_header(decode_header(value)))


def is_claim_subject(subject: str) -> bool:
    return subject.strip().lower().startswith("claim:")


def fetch_part(imap: imaplib.IMAP4, seq: bytes, spec: str) -> bytes | None:
    typ, data = imap.uid("FETCH", seq, spec)
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


def message_key(msg: Message) -> str:
    mid = decode_mime_header(msg.get("Message-ID")).strip()
    if mid:
        return mid
    return f"{decode_mime_header(msg.get('Subject'))}|{decode_mime_header(msg.get('Date'))}"


def claim_id_for(message_id: str) -> str:
    return hashlib.sha256(message_id.encode("utf-8")).hexdigest()[:8]


def ensure_mail_folders(imap: imaplib.IMAP4) -> None:
    for name in MAIL_FOLDERS:
        typ, data = imap.create(name)
        if typ == "OK":
            continue
        detail = b" ".join(item for item in (data or []) if isinstance(item, bytes)).decode(
            "utf-8", errors="replace"
        ).lower()
        if "exists" not in detail and "already" not in detail:
            print(f"folder {name}: {typ} {detail}", file=sys.stderr)


def _imap_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _uid_search(imap: imaplib.IMAP4, *criteria: str) -> list[bytes]:
    typ, data = imap.uid("SEARCH", None, *criteria)
    if typ != "OK" or not data or not data[0]:
        return []
    return data[0].split()


def find_uid_by_message_id(imap: imaplib.IMAP4, mailbox: str, message_id: str) -> bytes | None:
    typ, _ = imap.select(mailbox)
    if typ != "OK":
        return None
    for candidate in (message_id, message_id.strip("<>")):
        uids = _uid_search(imap, "HEADER", "Message-ID", _imap_quote(candidate))
        if uids:
            return uids[-1]
    uids = _uid_search(imap, "ALL")
    for uid in reversed(uids):
        raw = fetch_part(imap, uid, "(BODY.PEEK[HEADER.FIELDS (MESSAGE-ID SUBJECT DATE)])")
        if not raw:
            continue
        if message_key(parse_headers(raw)) == message_id:
            return uid
    return None


def imap_move(imap: imaplib.IMAP4, uid: bytes, dest: str) -> None:
    typ, _ = imap.uid("MOVE", uid, dest)
    if typ == "OK":
        return
    typ, _ = imap.uid("COPY", uid, dest)
    if typ != "OK":
        raise imaplib.IMAP4.error(f"copy to {dest} failed")
    imap.uid("STORE", uid, "+FLAGS", r"(\Deleted)")
    imap.expunge()


def file_mail(imap: imaplib.IMAP4, message_id: str, dest: str) -> None:
    sources = ("INBOX", FOLDER_REVIEW, FOLDER_FAILED, FOLDER_PROCESSED)
    for box in sources:
        uid = find_uid_by_message_id(imap, box, message_id)
        if uid is None:
            continue
        if box == dest:
            print(f"mail: already in {dest}")
            return
        imap.select(box)
        imap_move(imap, uid, dest)
        print(f"mail: {box} -> {dest}")
        return
    print(f"mail: not found for {dest}", file=sys.stderr)


def file_known_claims(imap: imaplib.IMAP4, root: Path, cfg: dict) -> None:
    import state as claim_state

    for claim in claim_state.iter_claims(root):
        mid = claim.get("message_id")
        status = claim.get("status")
        if not isinstance(mid, str) or not mid.strip():
            continue
        if status in ("logged", "skipped"):
            dest = FOLDER_PROCESSED
            if not claim.get("done_sent"):
                try:
                    send_close_mail(
                        cfg,
                        dict(claim.get("fields") or {}),
                        subject=str(claim.get("subject") or "CLAIM:"),
                        message_id=mid.strip(),
                        skipped=(status == "skipped"),
                    )
                    claim["done_sent"] = True
                    claim_state.save_claim(root, claim)
                except Exception as exc:
                    print(f"done mail failed: {exc}", file=sys.stderr)
        elif status == "awaiting_reply":
            dest = FOLDER_REVIEW
        else:
            continue
        try:
            file_mail(imap, mid.strip(), dest)
        except imaplib.IMAP4.error as exc:
            print(f"mail file failed: {exc}", file=sys.stderr)


def find_latest_claim(imap: imaplib.IMAP4, skip_ids: set[str] | None = None) -> tuple[bytes, Message] | None:
    skip_ids = skip_ids or set()
    typ, data = imap.uid("SEARCH", None, "ALL")
    if typ != "OK" or not data or not data[0]:
        return None
    seqs = data[0].split()
    for seq in reversed(seqs):
        raw_headers = fetch_part(
            imap, seq, "(BODY.PEEK[HEADER.FIELDS (SUBJECT DATE MESSAGE-ID)])"
        )
        if not raw_headers:
            continue
        headers = parse_headers(raw_headers)
        subject = decode_mime_header(headers.get("Subject"))
        if not is_claim_subject(subject):
            continue
        raw_full = fetch_part(imap, seq, "(BODY.PEEK[])")
        if not raw_full:
            continue
        msg = parse_headers(raw_full)
        if message_key(msg) in skip_ids:
            continue
        return seq, msg
    return None


def text_body(msg: Message) -> str:
    chunks: list[str] = []
    for part in iter_file_parts(msg):
        if part.get_content_type() != "text/plain":
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        charset = part.get_content_charset() or "utf-8"
        chunks.append(bytes(payload).decode(charset, errors="replace"))
    if chunks:
        return "\n".join(chunks).strip()
    if not msg.is_multipart():
        payload = msg.get_payload(decode=True)
        if payload and (msg.get_content_type() or "").startswith("text/"):
            charset = msg.get_content_charset() or "utf-8"
            return bytes(payload).decode(charset, errors="replace").strip()
    return ""


def print_extraction(fields: dict) -> None:
    from extract import format_money

    print("claim date:", fields.get("claim_date") or "(missing)")
    personal = fields.get("personal_amount")
    company = fields.get("company_amount")
    vat = fields.get("vat")
    print(f"personal: {personal if personal is not None else '(missing)'}")
    print(f"company: {company if company is not None else '(missing)'}")
    print(f"vat: {vat if vat is not None else '(missing)'}")
    print("comment:", fields.get("comment") or "(none)")
    print("journal:", fields.get("journal") or "(none)")
    print("proof:", fields.get("proof_file") or "(none)")
    print("confidence:", fields.get("confidence"))
    status = "needs_review" if fields.get("needs_review") else "logged"
    print("status:", status)
    reasons = fields.get("review_reasons") or []
    if reasons and fields.get("needs_review"):
        print("review:", "; ".join(str(r) for r in reasons))
    print("credits used:", format_money(fields.get("credits_used_usd")))
    print("credits left:", format_money(fields.get("credits_left_usd")))


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


def scan_path_for(orig: Path) -> Path:
    stem = orig.stem
    if "_orig-" in stem:
        stem = stem.replace("_orig-", "_scan-", 1)
    elif stem.endswith("_orig"):
        stem = stem[: -len("_orig")] + "_scan"
    else:
        stem = f"{stem}_scan"
    return orig.with_name(stem + ".jpg")


def save_scan(orig: Path) -> tuple[Path | None, str]:
    if orig.suffix.lower() in PDF_SUFFIXES:
        return None, "skipped (pdf)"
    if orig.suffix.lower() not in PHOTO_SUFFIXES:
        return None, "skipped"
    try:
        from crop import straighten_slip_hybrid
    except ImportError:
        return None, "failed (install: pip install -r requirements.txt)"
    jpeg, status = straighten_slip_hybrid(orig)
    if jpeg is None:
        return None, status
    dest = scan_path_for(orig)
    atomic_write(dest, jpeg)
    return dest, status


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


def finish_logged(fields: dict, orig: Path) -> None:
    from sheet import month_sheet_path, write_claim_row

    sheet = write_claim_row(month_sheet_path(orig.parent.parent), fields)
    print(f"sheet: {sheet}")


def send_close_mail(cfg: dict, fields: dict, *, subject: str, message_id: str, skipped: bool) -> None:
    from done_mail import send_done_mail

    send_done_mail(
        host=str(cfg["smtp_host"]),
        port=int(cfg["smtp_port"]),
        user=str(cfg["user"]),
        password=str(cfg["password"]),
        work_email=str(cfg["work_email"]),
        subject=subject,
        fields=fields,
        skipped=skipped,
        in_reply_to=message_id,
    )
    print("mail: Done" if not skipped else "mail: Skipped")


def notify_xai_credits(cfg: dict, *, remaining: float | None, detail: str) -> None:
    """Ping the main ntfy topic when prepaid xAI credit is gone or below the alert floor."""
    import json
    from datetime import datetime, timedelta, timezone

    from ntfy import publish

    url = str(cfg["ntfy_url"])
    if not url:
        print("xAI credit alert skipped: NTFY_URL empty", file=sys.stderr)
        return
    stamp_path = Path(str(cfg["state_dir"])) / "xai-credit-alert.json"
    cooldown = timedelta(hours=max(1, int(cfg["xai_credit_alert_hours"])))
    now = datetime.now(timezone.utc)
    if stamp_path.is_file():
        try:
            last = json.loads(stamp_path.read_text(encoding="utf-8")).get("sent")
            sent_at = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
            if sent_at.tzinfo is None:
                sent_at = sent_at.replace(tzinfo=timezone.utc)
            if now - sent_at < cooldown:
                print("xAI credit alert: still in cooldown")
                return
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            pass
    left = "unknown" if remaining is None else f"${remaining:.4f}"
    message = (
        "xAI prepaid credits are exhausted or too low to process claims. "
        f"Remaining about {left}. Top up at https://console.x.ai then the next loop will retry. "
        f"{detail[:200]}"
    )
    try:
        publish(
            url,
            str(cfg["ntfy_topic"]),
            title="xAI credits low",
            message=message,
        )
        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_path.write_text(json.dumps({"sent": now.isoformat()}), encoding="utf-8")
        print("ntfy: xAI credits low")
    except Exception as exc:
        print(f"xAI credit alert failed: {exc}", file=sys.stderr)


def maybe_alert_low_credits(cfg: dict, remaining: float | None) -> None:
    if remaining is None:
        return
    if remaining > float(cfg["xai_credit_alert_usd"]):
        return
    notify_xai_credits(cfg, remaining=remaining, detail="balance at or below alert floor")


def send_review_ping(cfg: dict, claim: dict, image: Path | None) -> None:
    from ntfy import LECA_PREFIX, publish

    url = str(cfg["ntfy_url"])
    if not url:
        print("ntfy skipped: NTFY_URL is empty", file=sys.stderr)
        sys.exit(1)
    claim_id = str(claim["id"])
    reasons = "; ".join(claim.get("reasons") or []) or "needs review"
    fields = claim.get("fields") or {}
    message = (
        f"{LECA_PREFIX}{claim_id}]\n"
        f"{claim.get('subject')}\n"
        f"{reasons}\n"
        f"personal {fields.get('personal_amount')}  company {fields.get('company_amount')}  "
        f"vat {fields.get('vat')}\n"
        "Reply in English (ok, skip, merchant Woolies, that was my half)."
    )
    result = publish(
        url,
        str(cfg["ntfy_topic"]),
        title="CLAIM needs review",
        message=message,
        image_path=image if image is not None and image.is_file() else None,
    )
    ntfy_id = str(result.get("id") or "")
    ids = list(claim.get("ntfy_ids") or [])
    if ntfy_id:
        ids.append(ntfy_id)
    claim["ntfy_ids"] = ids
    print(f"ntfy: sent claim {claim_id}")


def process_ntfy_replies(cfg: dict, imap: imaplib.IMAP4 | None = None) -> None:
    from extract import CreditExhaustedError, interpret_reply, journals_from_env
    from ntfy import is_our_message, poll
    import state as claim_state

    url = str(cfg["ntfy_url"])
    if not url:
        return
    root = Path(str(cfg["state_dir"]))
    pending = claim_state.awaiting(root)
    if not pending:
        return
    since = claim_state.load_cursor(root)
    try:
        messages = poll(url, str(cfg["ntfy_topic"]), since)
    except RuntimeError as exc:
        print(f"ntfy poll failed: {exc}", file=sys.stderr)
        return
    known = claim_state.known_ntfy_ids(root)
    last_id = since
    journals = journals_from_env()
    for item in messages:
        last_id = str(item.get("id") or last_id or "")
        if is_our_message(item, known):
            continue
        text = str(item.get("message") or "").strip()
        if not text:
            continue
        target = _match_pending(text, pending)
        if target is None:
            print(f"ntfy reply unmatched ({len(pending)} pending): {text[:80]}")
            continue
        print(f"ntfy reply for {target['id']}: {text}")
        try:
            result = interpret_reply(text, dict(target.get("fields") or {}), journals)
        except CreditExhaustedError as exc:
            print(f"ntfy reply parse failed (credits): {exc}", file=sys.stderr)
            notify_xai_credits(cfg, remaining=0.0, detail=str(exc))
            continue
        except RuntimeError as exc:
            print(f"ntfy reply parse failed: {exc}", file=sys.stderr)
            continue
        action = result["action"]
        fields = result["fields"]
        target["fields"] = fields
        mid = str(target.get("message_id") or "")
        if action == "skip":
            target["status"] = "skipped"
            try:
                send_close_mail(
                    cfg,
                    fields,
                    subject=str(target.get("subject") or "CLAIM:"),
                    message_id=mid,
                    skipped=True,
                )
                target["done_sent"] = True
            except Exception as exc:
                print(f"done mail failed: {exc}", file=sys.stderr)
            claim_state.save_claim(root, target)
            print(f"claim {target['id']}: skipped")
            if imap is not None and mid:
                file_mail(imap, mid, FOLDER_PROCESSED)
            pending = [c for c in pending if c["id"] != target["id"]]
            continue
        orig = Path(str(target.get("orig_path") or ""))
        if action == "update" and fields.get("needs_review"):
            claim_state.save_claim(root, target)
            print(f"claim {target['id']}: still needs review")
            print_extraction(fields)
            if imap is not None and mid:
                file_mail(imap, mid, FOLDER_REVIEW)
            continue
        if orig.is_file():
            finish_logged(fields, orig)
        try:
            send_close_mail(
                cfg,
                fields,
                subject=str(target.get("subject") or "CLAIM:"),
                message_id=mid,
                skipped=False,
            )
            target["done_sent"] = True
        except Exception as exc:
            print(f"done mail failed: {exc}", file=sys.stderr)
        target["status"] = "logged"
        claim_state.save_claim(root, target)
        print_extraction(fields)
        print(f"claim {target['id']}: logged")
        if imap is not None and mid:
            file_mail(imap, mid, FOLDER_PROCESSED)
        pending = [c for c in pending if c["id"] != target["id"]]
    if last_id:
        claim_state.save_cursor(root, last_id)


def _match_pending(reply: str, pending: list[dict]) -> dict | None:
    lower = reply.lower()
    for claim in pending:
        token = str(claim.get("id") or "")
        if token and token.lower() in lower:
            return claim
    if len(pending) == 1:
        return pending[0]
    return None


def _prune_state(cfg: dict) -> None:
    import state as claim_state

    removed = claim_state.prune(
        Path(str(cfg["state_dir"])),
        retention_days=int(cfg["state_retention_days"]),
        delete_completed=bool(cfg["delete_completed_state"]),
    )
    for cid in removed:
        print(f"state: removed {cid}")


def git_pull_home() -> None:
    import subprocess

    home = os.environ.get("AGENT_HOME", "").strip() or os.getcwd()
    print(f"command: git pull in {home}")
    result = subprocess.run(
        ["git", "-C", home, "pull"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    out = (result.stdout or "") + (result.stderr or "")
    print(out.strip() or f"git pull exit {result.returncode}")
    if result.returncode != 0:
        raise RuntimeError(f"git pull failed: {out.strip()[:400]}")


def restart_service() -> None:
    import shutil
    import subprocess

    unit = "Limitless-Expense-Capture-Agent.service"
    systemctl = shutil.which("systemctl")
    if not systemctl:
        raise RuntimeError("systemctl not found")
    sudo = shutil.which("sudo")
    commands = []
    if sudo:
        commands.append([sudo, "-n", systemctl, "restart", unit])
    commands.append([systemctl, "restart", unit])
    last = ""
    for cmd in commands:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode == 0:
            print("command: service restart requested")
            return
        last = (proc.stderr or proc.stdout or "").strip()
    raise RuntimeError(
        "restart failed (allow sudoers NOPASSWD systemctl restart "
        f"{unit}): {last[:300]}"
    )


def handle_control_commands(cfg: dict) -> None:
    from ntfy import command_topic, parse_command, poll, publish
    import state as claim_state

    url = str(cfg["ntfy_url"])
    if not url:
        return
    topic = command_topic(str(cfg["ntfy_topic"]))
    root = Path(str(cfg["state_dir"]))
    since = claim_state.load_cursor(root, claim_state.COMMAND_CURSOR_NAME)
    try:
        messages = poll(url, topic, since)
    except RuntimeError as exc:
        print(f"command topic poll failed: {exc}", file=sys.stderr)
        return
    last_id = since
    action = None
    for item in messages:
        last_id = str(item.get("id") or last_id or "")
        text = str(item.get("message") or "")
        parsed = parse_command(text)
        if parsed:
            action = parsed
            print(f"command topic: {parsed}")
    if last_id:
        claim_state.save_cursor(root, last_id, claim_state.COMMAND_CURSOR_NAME)
    if action == "git-pull":
        git_pull_home()
        restart_service()
    elif action == "restart":
        restart_service()


def announce_command_topic(cfg: dict) -> None:
    from ntfy import command_topic, publish
    import state as claim_state

    url = str(cfg["ntfy_url"])
    if not url:
        print("command topic: NTFY_URL empty, not announced", file=sys.stderr)
        return
    topic = command_topic(str(cfg["ntfy_topic"]))
    result = publish(
        url,
        topic,
        title="Limitless Expense Capture Agent",
        message="commands here",
    )
    ntfy_id = str(result.get("id") or "")
    if ntfy_id:
        claim_state.save_cursor(
            Path(str(cfg["state_dir"])), ntfy_id, claim_state.COMMAND_CURSOR_NAME
        )
    print(f"command topic: {topic}")


def process_inbox_claim(
    cfg: dict, imap: imaplib.IMAP4, msg: Message, state_root: Path
) -> str:
    import state as claim_state
    from extract import CreditExhaustedError, extract_claim, journals_from_env, notify_reasons
    from sheet import confirm_proofs

    report(msg)
    mid = message_key(msg)
    subject = decode_mime_header(msg.get("Subject"))
    body = text_body(msg)
    try:
        orig_paths = save_originals(msg, Path(str(cfg["expenses_root"])))
    except SystemExit:
        file_mail(imap, mid, FOLDER_FAILED)
        return "failed"
    for path in orig_paths:
        print(f"wrote: {path}")
        pdf_text = ""
        try:
            if path.suffix.lower() in PDF_SUFFIXES:
                from pdf_proof import prepare_pdf

                pdf_text, scan, crop_status = prepare_pdf(path)
            else:
                scan, crop_status = save_scan(path)
        except CreditExhaustedError as exc:
            print(f"crop xAI credits: {exc}", file=sys.stderr)
            notify_xai_credits(cfg, remaining=0.0, detail=str(exc))
            file_mail(imap, mid, FOLDER_FAILED)
            raise
        print(f"crop: {crop_status}")
        if scan is not None:
            print(f"wrote: {scan}")
        if scan is not None:
            image: Path | None = scan
        elif path.suffix.lower() in PDF_SUFFIXES:
            image = None
        else:
            image = path
        try:
            fields = extract_claim(
                subject=subject,
                body=body,
                image_path=image,
                proof_name=(scan or path).name,
                journals=journals_from_env(),
                pdf_text=pdf_text,
            )
        except CreditExhaustedError as exc:
            print(f"extract failed (credits): {exc}", file=sys.stderr)
            notify_xai_credits(cfg, remaining=0.0, detail=str(exc))
            file_mail(imap, mid, FOLDER_FAILED)
            raise
        except RuntimeError as exc:
            print(f"extract failed: {exc}", file=sys.stderr)
            file_mail(imap, mid, FOLDER_FAILED)
            return "failed"

        path, scan, proof_name = confirm_proofs(
            path, scan, fields.get("claim_date"), fields.get("merchant")
        )
        fields["proof_file"] = proof_name
        print_extraction(fields)
        maybe_alert_low_credits(cfg, fields.get("credits_left_usd"))
        reasons = notify_reasons(fields, crop_status)
        if not reasons:
            finish_logged(fields, path)
            done_sent = False
            try:
                send_close_mail(cfg, fields, subject=subject, message_id=mid, skipped=False)
                done_sent = True
            except Exception as exc:
                print(f"done mail failed: {exc}", file=sys.stderr)
            claim_state.save_claim(
                state_root,
                {
                    "id": claim_id_for(mid),
                    "status": "logged",
                    "message_id": mid,
                    "subject": subject,
                    "fields": fields,
                    "orig_path": str(path),
                    "scan_path": str(scan) if scan is not None else "",
                    "done_sent": done_sent,
                },
            )
            file_mail(imap, mid, FOLDER_PROCESSED)
            continue
        claim = {
            "id": claim_id_for(mid),
            "status": "awaiting_reply",
            "message_id": mid,
            "subject": subject,
            "reasons": reasons,
            "fields": fields,
            "orig_path": str(path),
            "scan_path": str(scan) if scan is not None else "",
            "ntfy_ids": [],
        }
        ping_image = scan if scan is not None else path
        send_review_ping(cfg, claim, ping_image)
        saved = claim_state.save_claim(state_root, claim)
        print(f"state: {saved}")
        print("sheet: (held until ntfy reply)")
        file_mail(imap, mid, FOLDER_REVIEW)
        return "review"
    return "logged"


def run_once() -> None:
    cfg = load_config()
    print("pass: checking mail")
    imap: imaplib.IMAP4 | None = None
    try:
        imap = imaplib.IMAP4_SSL(str(cfg["host"]), int(cfg["port"]))
        imap.login(str(cfg["user"]), str(cfg["password"]))
        ensure_mail_folders(imap)
        import state as claim_state
        from extract import CreditExhaustedError, extract_claim, journals_from_env, notify_reasons
        from sheet import confirm_proofs

        state_root = Path(str(cfg["state_dir"]))
        file_known_claims(imap, state_root, cfg)
        process_ntfy_replies(cfg, imap)
        waiting = claim_state.awaiting(state_root)
        if waiting:
            ids = ", ".join(str(c.get("id") or "?") for c in waiting)
            print(f"waiting for ntfy reply on claim {ids}; not starting a new claim")
            _prune_state(cfg)
            return
        done = 0
        while True:
            typ, _ = imap.select("INBOX")
            if typ != "OK":
                print("failed to select INBOX", file=sys.stderr)
                sys.exit(1)
            skip_ids = claim_state.known_message_ids(state_root)
            found = find_latest_claim(imap, skip_ids)
            if found is None:
                if done == 0:
                    print("No new CLAIM: message found in INBOX")
                else:
                    print(f"pass: finished {done} claim(s)")
                break
            _, msg = found
            try:
                outcome = process_inbox_claim(cfg, imap, msg, state_root)
            except CreditExhaustedError:
                break
            done += 1
            if outcome == "review":
                print("pass: paused for ntfy; will not take another claim until this one is done or skipped")
                break
        _prune_state(cfg)
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


def main() -> None:
    import argparse
    import time

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Limitless Expense Capture Agent")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="run forever, sleeping POLL_INTERVAL_MINUTES between passes",
    )
    args = parser.parse_args()
    if not args.loop:
        run_once()
        return
    cfg = load_config()
    print("loop: starting")
    try:
        announce_command_topic(cfg)
    except Exception as exc:
        print(f"command topic announce failed: {exc}", file=sys.stderr)
    while True:
        try:
            handle_control_commands(cfg)
            run_once()
        except SystemExit as exc:
            if exc.code == 2:
                raise
            print(f"run ended: {exc.code}", file=sys.stderr)
        except Exception as exc:
            print(f"run error: {exc}", file=sys.stderr)
        cfg = load_config()
        minutes = max(1, int(cfg["poll_interval_minutes"]))
        print(f"sleep: {minutes} minutes")
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(minutes * 60)


if __name__ == "__main__":
    main()
